package tokenizer

import (
	"os"
	"slices"
	"testing"

	"github.com/manboster/manboml/internal/loader"
)

func testMeta() loader.Tokenizer {
	return loader.Tokenizer{
		Model:      "gpt2",
		Pre:        "qwen2",
		Tokens:     []string{"a", "b", "x", "ab", "<|im_start|>", "<|im_end|>", "abc", "Ġ"},
		TokenTypes: []int32{1, 1, 1, 1, 3, 3, 1, 1},
		Merges:     []string{"a b"},
		Template:   qwen3GuardTemplate,
		BOSTokenID: 4,
		EOSTokenID: 5,
		EOTTokenID: -1,
		EOMTokenID: -1,
	}
}

func TestEncodeShortCircuit(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("ab")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{3}) {
		t.Fatalf("ids = %v", ids)
	}
}

func TestEncodeBPEMerge(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("abx")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{3, 2}) {
		t.Fatalf("ids = %v, want [3 2]", ids)
	}
}

func TestEncodeSpecialTokens(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("x<|im_end|>")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{2, 5}) {
		t.Fatalf("ids = %v, want [2 5]", ids)
	}
}

func TestDecodeRoundTrip(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("abx")
	if err != nil {
		t.Fatal(err)
	}
	text, err := tk.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if text != "abx" {
		t.Fatalf("decode = %q", text)
	}
}

func TestEOGAndMetadata(t *testing.T) {
	tk, err := New(testMeta())
	if err != nil {
		t.Fatal(err)
	}
	if !tk.IsEOG(5) || tk.IsEOG(4) || tk.IsEOG(0) {
		t.Fatal("EOG set wrong")
	}
	if tk.BOS() != 4 {
		t.Fatalf("BOS = %d", tk.BOS())
	}
	if tk.Size() != 8 {
		t.Fatalf("Size = %d", tk.Size())
	}
}

func TestNewValidation(t *testing.T) {
	meta := testMeta()
	meta.Model = "llama"
	if _, err := New(meta); err == nil {
		t.Fatal("sentencepiece model accepted")
	}
	meta = testMeta()
	meta.Tokens = nil
	if _, err := New(meta); err == nil {
		t.Fatal("empty vocabulary accepted")
	}
}

func TestSynthesizedTokenTypes(t *testing.T) {
	meta := testMeta()
	meta.TokenTypes = nil
	tk, err := New(meta)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tk.Encode("x<|im_end|>")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ids, []int32{2, 5}) {
		t.Fatalf("ids = %v, want [2 5]", ids)
	}
}

func TestRealModelTokenizer(t *testing.T) {
	path := os.Getenv("MANBOML_TEST_GGUF")
	if path == "" {
		t.Skip("MANBOML_TEST_GGUF not set")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	m, err := loader.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	tk, err := New(m.Tokenizer)
	if err != nil {
		t.Fatal(err)
	}

	ids, err := tk.Encode("<|im_start|>user\nhello<|im_end|>")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 || ids[0] != 151644 {
		t.Fatalf("first id = %v, want 151644", ids)
	}
	if !slices.Contains(ids, 151645) {
		t.Fatalf("ids %v missing 151645", ids)
	}
	if !tk.IsEOG(151645) {
		t.Fatal("EOG set missing eos token 151645")
	}
	if tk.BOS() != 151643 {
		t.Fatalf("BOS = %d, want 151643", tk.BOS())
	}
	if !RecognizeTemplate(m.Tokenizer.Template) {
		t.Fatal("real Qwen3Guard template not recognized")
	}
}
