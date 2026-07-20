package manboml

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// tinyGGUF builds a minimal valid Qwen3 GGUF for root API tests.
func tinyGGUF() []byte {
	const (
		layers     = 2
		hidden     = 256
		ffn        = 256
		queryHeads = 4
		kvHeads    = 2
		headDim    = 64
		vocab      = 8
		context    = 32
	)
	seed := uint64(20260720)
	lcg := func() uint32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return uint32(seed >> 32)
	}
	q4k := func(rows, cols int) []byte {
		data := make([]byte, rows*cols/256*144)
		for i := range data {
			data[i] = byte(lcg())
		}
		for off := 0; off+144 <= len(data); off += 144 {
			binary.LittleEndian.PutUint16(data[off:], uint16(0x2C00+lcg()%0x0800))
			binary.LittleEndian.PutUint16(data[off+2:], uint16(0x2000+lcg()%0x0800))
		}
		return data
	}
	q6k := func(rows, cols int) []byte {
		data := make([]byte, rows*cols/256*210)
		for i := range data {
			data[i] = byte(lcg())
		}
		for off := 0; off+210 <= len(data); off += 210 {
			binary.LittleEndian.PutUint16(data[off+208:], uint16(0x2C00+lcg()%0x0800))
		}
		return data
	}
	f32 := func(n int) []byte {
		data := make([]byte, n*4)
		for i := 0; i < n; i++ {
			v := 0.5 + float32(lcg()%2048)/1024
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
		return data
	}

	tokens := []string{"a", "b", "c", "d", "e", "f", "g", "<|im_end|>"}
	types := []int32{1, 1, 1, 1, 1, 1, 1, 3}

	type kv struct {
		key   string
		typ   uint32
		value any
	}
	kvs := []kv{
		{"general.architecture", 8, "qwen3"},
		{"general.alignment", 4, uint32(32)},
		{"qwen3.block_count", 4, uint32(layers)},
		{"qwen3.context_length", 4, uint32(context)},
		{"qwen3.embedding_length", 4, uint32(hidden)},
		{"qwen3.feed_forward_length", 4, uint32(ffn)},
		{"qwen3.attention.head_count", 4, uint32(queryHeads)},
		{"qwen3.attention.head_count_kv", 4, uint32(kvHeads)},
		{"qwen3.attention.key_length", 4, uint32(headDim)},
		{"qwen3.rope.freq_base", 6, float32(10000)},
		{"qwen3.attention.layer_norm_rms_epsilon", 6, float32(1e-6)},
		{"tokenizer.ggml.model", 8, "gpt2"},
		{"tokenizer.ggml.pre", 8, "qwen2"},
		{"tokenizer.ggml.tokens", 9, tokens},
		{"tokenizer.ggml.token_type", 9, types},
		{"tokenizer.ggml.eos_token_id", 4, uint32(vocab - 1)},
	}

	type tensor struct {
		name string
		kind uint32
		dims []uint64
		data []byte
	}
	tensors := []tensor{
		{"token_embd.weight", 12, []uint64{hidden, vocab}, q4k(vocab, hidden)},
		{"output.weight", 14, []uint64{hidden, vocab}, q6k(vocab, hidden)},
		{"output_norm.weight", 0, []uint64{hidden}, f32(hidden)},
	}
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("blk.%d.", l)
		qw, kw := uint64(queryHeads*headDim), uint64(kvHeads*headDim)
		tensors = append(tensors,
			tensor{p + "attn_norm.weight", 0, []uint64{hidden}, f32(hidden)},
			tensor{p + "attn_q.weight", 12, []uint64{hidden, qw}, q4k(int(qw), hidden)},
			tensor{p + "attn_k.weight", 12, []uint64{hidden, kw}, q4k(int(kw), hidden)},
			tensor{p + "attn_v.weight", 12, []uint64{hidden, kw}, q4k(int(kw), hidden)},
			tensor{p + "attn_q_norm.weight", 0, []uint64{headDim}, f32(headDim)},
			tensor{p + "attn_k_norm.weight", 0, []uint64{headDim}, f32(headDim)},
			tensor{p + "attn_output.weight", 12, []uint64{qw, hidden}, q4k(hidden, int(qw))},
			tensor{p + "ffn_norm.weight", 0, []uint64{hidden}, f32(hidden)},
			tensor{p + "ffn_gate.weight", 12, []uint64{hidden, ffn}, q4k(ffn, hidden)},
			tensor{p + "ffn_up.weight", 12, []uint64{hidden, ffn}, q4k(ffn, hidden)},
			tensor{p + "ffn_down.weight", 12, []uint64{ffn, hidden}, q4k(hidden, ffn)},
		)
	}

	ggufString := func(b *bytes.Buffer, s string) {
		binary.Write(b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
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
	offsets := make([]uint64, len(tensors))
	var off uint64
	for i, t := range tensors {
		offsets[i] = off
		off += uint64(len(t.data))
		if off%32 != 0 {
			off += 32 - off%32
		}
	}
	for i, t := range tensors {
		ggufString(&b, t.name)
		binary.Write(&b, binary.LittleEndian, uint32(len(t.dims)))
		for _, d := range t.dims {
			binary.Write(&b, binary.LittleEndian, d)
		}
		binary.Write(&b, binary.LittleEndian, t.kind)
		binary.Write(&b, binary.LittleEndian, offsets[i])
	}
	for b.Len()%32 != 0 {
		b.WriteByte(0)
	}
	for _, t := range tensors {
		b.Write(t.data)
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
	}
	return b.Bytes()
}
