package qwen3

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// fixture describes a tiny project-owned Qwen3 model used for offline tests.
type fixture struct {
	layers     int
	hidden     int
	ffn        int
	queryHeads int
	kvHeads    int
	headDim    int
	vocab      int
	context    int
	seed       uint64
}

func defaultFixture() fixture {
	return fixture{
		layers:     2,
		hidden:     256,
		ffn:        256,
		queryHeads: 4,
		kvHeads:    2,
		headDim:    64,
		vocab:      8,
		context:    32,
		seed:       20260719,
	}
}

func lcg(seed *uint64) uint32 {
	*seed = *seed*6364136223846793005 + 1442695040888963407
	return uint32(*seed >> 32)
}

func q4kMatrixBytes(seed *uint64, rows, cols int) []byte {
	blocks := cols / 256
	data := make([]byte, rows*blocks*144)
	for i := range data {
		data[i] = byte(lcg(seed))
	}
	for off := 0; off+144 <= len(data); off += 144 {
		binary.LittleEndian.PutUint16(data[off:], uint16(0x2C00+lcg(seed)%0x0800))
		binary.LittleEndian.PutUint16(data[off+2:], uint16(0x2000+lcg(seed)%0x0800))
	}
	return data
}

func f32VectorBytes(seed *uint64, n int) []byte {
	data := make([]byte, n*4)
	for i := 0; i < n; i++ {
		v := 0.5 + float32(lcg(seed)%2048)/1024
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	return data
}

type kv struct {
	key   string
	typ   uint32
	value any
}

type tensorDef struct {
	name string
	kind uint32
	dims []uint64
	data []byte
}

func ggufString(b *bytes.Buffer, s string) {
	binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

func buildTinyGGUF(f fixture) []byte {
	seed := f.seed
	queryWidth := f.queryHeads * f.headDim
	kvWidth := f.kvHeads * f.headDim

	tokens := make([]string, f.vocab)
	types := make([]int32, f.vocab)
	for i := range tokens {
		tokens[i] = string(rune('a' + i))
		types[i] = 1
	}
	tokens[f.vocab-1] = "<|im_end|>"
	types[f.vocab-1] = 3

	kvs := []kv{
		{"general.architecture", 8, "qwen3"},
		{"general.alignment", 4, uint32(32)},
		{"qwen3.block_count", 4, uint32(f.layers)},
		{"qwen3.context_length", 4, uint32(f.context)},
		{"qwen3.embedding_length", 4, uint32(f.hidden)},
		{"qwen3.feed_forward_length", 4, uint32(f.ffn)},
		{"qwen3.attention.head_count", 4, uint32(f.queryHeads)},
		{"qwen3.attention.head_count_kv", 4, uint32(f.kvHeads)},
		{"qwen3.attention.key_length", 4, uint32(f.headDim)},
		{"qwen3.attention.value_length", 4, uint32(f.headDim)},
		{"qwen3.rope.freq_base", 6, float32(10000)},
		{"qwen3.attention.layer_norm_rms_epsilon", 6, float32(1e-6)},
		{"tokenizer.ggml.model", 8, "gpt2"},
		{"tokenizer.ggml.pre", 8, "qwen2"},
		{"tokenizer.ggml.tokens", 9, tokens},
		{"tokenizer.ggml.token_type", 9, types},
		{"tokenizer.ggml.eos_token_id", 4, uint32(f.vocab - 1)},
	}

	tensors := []tensorDef{
		{"token_embd.weight", 12, []uint64{uint64(f.hidden), uint64(f.vocab)}, q4kMatrixBytes(&seed, f.vocab, f.hidden)},
		{"output.weight", 14, []uint64{uint64(f.hidden), uint64(f.vocab)}, nil},
		{"output_norm.weight", 0, []uint64{uint64(f.hidden)}, f32VectorBytes(&seed, f.hidden)},
	}
	tensors[1].data = make([]byte, f.vocab*f.hidden/256*210)
	for i := range tensors[1].data {
		tensors[1].data[i] = byte(lcg(&seed))
	}
	for off := 0; off+210 <= len(tensors[1].data); off += 210 {
		binary.LittleEndian.PutUint16(tensors[1].data[off+208:], uint16(0x2C00+lcg(&seed)%0x0800))
	}

	for l := 0; l < f.layers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		tensors = append(tensors,
			tensorDef{p + "attn_norm.weight", 0, []uint64{uint64(f.hidden)}, f32VectorBytes(&seed, f.hidden)},
			tensorDef{p + "attn_q.weight", 12, []uint64{uint64(f.hidden), uint64(queryWidth)}, q4kMatrixBytes(&seed, queryWidth, f.hidden)},
			tensorDef{p + "attn_k.weight", 12, []uint64{uint64(f.hidden), uint64(kvWidth)}, q4kMatrixBytes(&seed, kvWidth, f.hidden)},
			tensorDef{p + "attn_v.weight", 12, []uint64{uint64(f.hidden), uint64(kvWidth)}, q4kMatrixBytes(&seed, kvWidth, f.hidden)},
			tensorDef{p + "attn_q_norm.weight", 0, []uint64{uint64(f.headDim)}, f32VectorBytes(&seed, f.headDim)},
			tensorDef{p + "attn_k_norm.weight", 0, []uint64{uint64(f.headDim)}, f32VectorBytes(&seed, f.headDim)},
			tensorDef{p + "attn_output.weight", 12, []uint64{uint64(queryWidth), uint64(f.hidden)}, q4kMatrixBytes(&seed, f.hidden, queryWidth)},
			tensorDef{p + "ffn_norm.weight", 0, []uint64{uint64(f.hidden)}, f32VectorBytes(&seed, f.hidden)},
			tensorDef{p + "ffn_gate.weight", 12, []uint64{uint64(f.hidden), uint64(f.ffn)}, q4kMatrixBytes(&seed, f.ffn, f.hidden)},
			tensorDef{p + "ffn_up.weight", 12, []uint64{uint64(f.hidden), uint64(f.ffn)}, q4kMatrixBytes(&seed, f.ffn, f.hidden)},
			tensorDef{p + "ffn_down.weight", 12, []uint64{uint64(f.ffn), uint64(f.hidden)}, q4kMatrixBytes(&seed, f.hidden, f.ffn)},
		)
	}

	var b bytes.Buffer
	b.WriteString("GGUF")
	binary.Write(&b, binary.LittleEndian, uint32(3))
	binary.Write(&b, binary.LittleEndian, uint64(len(tensors)))
	binary.Write(&b, binary.LittleEndian, uint64(len(kvs)))
	for _, item := range kvs {
		ggufString(&b, item.key)
		binary.Write(&b, binary.LittleEndian, item.typ)
		switch item.typ {
		case 4:
			binary.Write(&b, binary.LittleEndian, item.value.(uint32))
		case 6:
			binary.Write(&b, binary.LittleEndian, item.value.(float32))
		case 8:
			ggufString(&b, item.value.(string))
		case 9:
			switch items := item.value.(type) {
			case []string:
				binary.Write(&b, binary.LittleEndian, uint32(8))
				binary.Write(&b, binary.LittleEndian, uint64(len(items)))
				for _, s := range items {
					ggufString(&b, s)
				}
			case []int32:
				binary.Write(&b, binary.LittleEndian, uint32(5))
				binary.Write(&b, binary.LittleEndian, uint64(len(items)))
				for _, n := range items {
					binary.Write(&b, binary.LittleEndian, n)
				}
			}
		}
	}

	offset := uint64(0)
	type placed struct {
		tensorDef
		off uint64
	}
	placedTensors := make([]placed, len(tensors))
	for i, t := range tensors {
		placedTensors[i] = placed{t, offset}
		offset += uint64(len(t.data))
		if offset%32 != 0 {
			offset += 32 - offset%32
		}
	}
	for _, t := range placedTensors {
		ggufString(&b, t.name)
		binary.Write(&b, binary.LittleEndian, uint32(len(t.dims)))
		for _, d := range t.dims {
			binary.Write(&b, binary.LittleEndian, d)
		}
		binary.Write(&b, binary.LittleEndian, t.kind)
		binary.Write(&b, binary.LittleEndian, t.off)
	}
	if b.Len()%32 != 0 {
		b.Write(make([]byte, 32-b.Len()%32))
	}
	for _, t := range placedTensors {
		b.Write(t.data)
		if b.Len()%32 != 0 {
			b.Write(make([]byte, 32-b.Len()%32))
		}
	}
	return b.Bytes()
}
