package loader

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"testing"
)

type testKV struct {
	key   string
	typ   uint32
	value any
}

type testTensor struct {
	name   string
	kind   uint32
	dims   []uint64
	offset uint64
	size   uint64
}

func ggufString(b *bytes.Buffer, s string) {
	binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func buildGGUF(version uint32, kvs []testKV, tensors []testTensor, dataSize int) []byte {
	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, version)
	binary.Write(&b, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(&b, binary.LittleEndian, uint64(len(kvs)))
	for _, kv := range kvs {
		ggufString(&b, kv.key)
		binary.Write(&b, binary.LittleEndian, kv.typ)
		switch kv.typ {
		case 4:
			binary.Write(&b, binary.LittleEndian, kv.value.(uint32))
		case 8:
			ggufString(&b, kv.value.(string))
		case 9:
			switch items := kv.value.(type) {
			case []string:
				binary.Write(&b, binary.LittleEndian, uint32(8))
				binary.Write(&b, binary.LittleEndian, uint64(len(items)))
				for _, item := range items {
					ggufString(&b, item)
				}
			case []int32:
				binary.Write(&b, binary.LittleEndian, uint32(5))
				binary.Write(&b, binary.LittleEndian, uint64(len(items)))
				for _, item := range items {
					binary.Write(&b, binary.LittleEndian, item)
				}
			default:
				panic("unsupported test array type")
			}
		default:
			panic("unsupported test value type")
		}
	}
	for _, t := range tensors {
		ggufString(&b, t.name)
		binary.Write(&b, binary.LittleEndian, uint32(len(t.dims)))
		for _, d := range t.dims {
			binary.Write(&b, binary.LittleEndian, d)
		}
		binary.Write(&b, binary.LittleEndian, t.kind)
		binary.Write(&b, binary.LittleEndian, t.offset)
	}
	if pad := (32 - b.Len()%32) % 32; pad > 0 {
		b.Write(make([]byte, pad))
	}
	b.Write(make([]byte, dataSize))
	return b.Bytes()
}

func baseKVs() []testKV {
	return []testKV{
		{key: "general.architecture", typ: 8, value: "qwen3"},
		{key: "general.alignment", typ: 4, value: uint32(32)},
		{key: "qwen3.attention.head_count", typ: 4, value: uint32(16)},
		{key: "tokenizer.ggml.model", typ: 8, value: "gpt2"},
		{key: "tokenizer.ggml.pre", typ: 8, value: "qwen2"},
		{key: "tokenizer.ggml.tokens", typ: 9, value: []string{"a", "b", "<|im_end|>"}},
		{key: "tokenizer.ggml.token_type", typ: 9, value: []int32{1, 1, 3}},
		{key: "tokenizer.ggml.merges", typ: 9, value: []string{"a b"}},
		{key: "tokenizer.ggml.bos_token_id", typ: 4, value: uint32(1)},
		{key: "tokenizer.ggml.eos_token_id", typ: 4, value: uint32(2)},
		{key: "tokenizer.chat_template", typ: 8, value: "template-bytes"},
	}
}

func baseTensors() []testTensor {
	return []testTensor{
		{name: "token_embd.weight", kind: 0, dims: []uint64{4, 2}, offset: 0, size: 32},
		{name: "blk.0.ffn_gate.weight", kind: 12, dims: []uint64{256, 2}, offset: 32, size: 288},
	}
}

func TestParseValid(t *testing.T) {
	for _, version := range []uint32{2, 3} {
		raw := buildGGUF(version, baseKVs(), baseTensors(), 320)
		m, err := Parse(raw)
		if err != nil {
			t.Fatalf("v%d Parse: %v", version, err)
		}
		if m.Architecture != "qwen3" {
			t.Fatalf("v%d architecture = %q", version, m.Architecture)
		}
		if len(m.Tensors) != 2 {
			t.Fatalf("v%d tensors = %d", version, len(m.Tensors))
		}

		dataStart := uint64(len(raw) - 320)
		if got := m.Tensors[0].Offset; got != dataStart {
			t.Fatalf("v%d tensor[0] offset = %d, want %d", version, got, dataStart)
		}
		if got := m.Tensors[1].Offset; got != dataStart+32 {
			t.Fatalf("v%d tensor[1] offset = %d, want %d", version, got, dataStart+32)
		}
		if m.Tensors[0].Size != 32 || m.Tensors[1].Size != 288 {
			t.Fatalf("v%d sizes = %d, %d", version, m.Tensors[0].Size, m.Tensors[1].Size)
		}
		if m.Tensors[1].Kind != 12 {
			t.Fatalf("v%d kind = %d", version, m.Tensors[1].Kind)
		}

		if heads, ok := m.Uint("attention.head_count"); !ok || heads != 16 {
			t.Fatalf("v%d attention.head_count = %d, %v", version, heads, ok)
		}
		if arch, ok := m.String("general.architecture"); !ok || arch != "qwen3" {
			t.Fatalf("v%d general.architecture = %q, %v", version, arch, ok)
		}

		tk := m.Tokenizer
		if tk.Model != "gpt2" || tk.Pre != "qwen2" {
			t.Fatalf("v%d tokenizer model = %q pre = %q", version, tk.Model, tk.Pre)
		}
		if len(tk.Tokens) != 3 || tk.Tokens[2] != "<|im_end|>" {
			t.Fatalf("v%d tokens = %v", version, tk.Tokens)
		}
		if len(tk.TokenTypes) != 3 || tk.TokenTypes[2] != 3 {
			t.Fatalf("v%d token types = %v", version, tk.TokenTypes)
		}
		if len(tk.Merges) != 1 || tk.Merges[0] != "a b" {
			t.Fatalf("v%d merges = %v", version, tk.Merges)
		}
		if tk.BOSTokenID != 1 || tk.EOSTokenID != 2 {
			t.Fatalf("v%d special ids = %d, %d", version, tk.BOSTokenID, tk.EOSTokenID)
		}
		if tk.EOTTokenID != -1 || tk.UnknownTokenID != -1 {
			t.Fatalf("v%d missing ids = %d, %d", version, tk.EOTTokenID, tk.UnknownTokenID)
		}
		if tk.Template != "template-bytes" {
			t.Fatalf("v%d template = %q", version, tk.Template)
		}
	}
}

func TestParseRejectsBadHeader(t *testing.T) {
	raw := buildGGUF(3, baseKVs(), baseTensors(), 320)

	if _, err := Parse(raw[:8]); err == nil {
		t.Fatal("short file parsed successfully")
	}

	bad := bytes.Clone(raw)
	copy(bad[:4], "XXXX")
	if _, err := Parse(bad); !errors.Is(err, ErrInvalidFormat) {
		t.Fatalf("bad magic err = %v", err)
	}

	be := bytes.Clone(raw)
	copy(be[:4], "FUGG")
	if _, err := Parse(be); !errors.Is(err, ErrUnsupportedEndian) {
		t.Fatalf("big endian err = %v", err)
	}

	for _, version := range []uint32{1, 4} {
		v := bytes.Clone(raw)
		binary.LittleEndian.PutUint32(v[4:8], version)
		if _, err := Parse(v); !errors.Is(err, ErrUnsupportedVersion) {
			t.Fatalf("version %d err = %v", version, err)
		}
	}
}

func TestParseRejectsDuplicateTensor(t *testing.T) {
	tensors := baseTensors()
	tensors[1].name = tensors[0].name
	tensors[1].offset = 32
	raw := buildGGUF(3, baseKVs(), tensors, 320)
	if _, err := Parse(raw); !errors.Is(err, ErrDuplicateTensor) {
		t.Fatalf("err = %v", err)
	}
}

func TestParseRejectsOverlap(t *testing.T) {
	tensors := baseTensors()
	tensors[1].offset = 16
	raw := buildGGUF(3, baseKVs(), tensors, 320)
	if _, err := Parse(raw); err == nil {
		t.Fatal("overlapping tensors parsed successfully")
	}
}

func TestParseRejectsMisaligned(t *testing.T) {
	tensors := baseTensors()
	tensors[1].offset = 40
	raw := buildGGUF(3, baseKVs(), tensors, 328)
	if _, err := Parse(raw); !errors.Is(err, ErrMisalignedTensor) {
		t.Fatalf("err = %v", err)
	}
}

func TestParseRejectsUnknownKind(t *testing.T) {
	tensors := baseTensors()
	tensors[1].kind = 99
	raw := buildGGUF(3, baseKVs(), tensors, 320)
	if _, err := Parse(raw); err == nil {
		t.Fatal("unknown tensor kind parsed successfully")
	}
}

func TestParseRejectsTruncated(t *testing.T) {
	raw := buildGGUF(3, baseKVs(), baseTensors(), 320)
	if _, err := Parse(raw[:len(raw)-1]); err == nil {
		t.Fatal("truncated file parsed successfully")
	}
}

func TestParseRealModel(t *testing.T) {
	path := os.Getenv("MANBOML_TEST_GGUF")
	if path == "" {
		t.Skip("MANBOML_TEST_GGUF not set")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Parse(raw)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if m.Architecture != "qwen3" {
		t.Fatalf("architecture = %q", m.Architecture)
	}
	if len(m.Tensors) != 311 {
		t.Fatalf("tensors = %d, want 311", len(m.Tensors))
	}
	hist := map[uint32]int{}
	for _, tensor := range m.Tensors {
		hist[tensor.Kind]++
	}
	if hist[12] != 169 || hist[14] != 29 || hist[0] != 113 {
		t.Fatalf("kind histogram = %v", hist)
	}
	if got := len(m.Tokenizer.Tokens); got != 151936 {
		t.Fatalf("vocab = %d, want 151936", got)
	}
}
