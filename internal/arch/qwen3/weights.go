package qwen3

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/manboster/manboml/internal/loader"
	"github.com/manboster/manboml/internal/ops"
)

// LayerWeights holds one decoder layer's bound tensors.
type LayerWeights struct {
	AttnNorm []float32
	Q        ops.Matrix
	K        ops.Matrix
	V        ops.Matrix
	QNorm    []float32
	KNorm    []float32
	AttnOut  ops.Matrix
	FFNNorm  []float32
	Gate     ops.Matrix
	Up       ops.Matrix
	Down     ops.Matrix
}

// Weights is the immutable, validated tensor set of a Qwen3 model. Packed
// matrices point into the model backing; norm vectors are decoded F32.
type Weights struct {
	Embedding  ops.Matrix
	Output     ops.Matrix
	OutputNorm []float32
	Layers     []LayerWeights
}

// BindWeights validates the required Qwen3 tensor set and binds it to the
// backing bytes. Missing tensors, shape mismatches and unsupported encodings
// are rejected.
func BindWeights(data []byte, tensors []loader.Tensor, cfg Config) (*Weights, error) {
	byName := make(map[string]loader.Tensor, len(tensors))
	for _, t := range tensors {
		byName[t.Name] = t
	}
	matrix := func(name string, rows, cols int) (ops.Matrix, error) {
		t, ok := byName[name]
		if !ok {
			return ops.Matrix{}, fmt.Errorf("%w: missing tensor %q", ErrInvalidWeights, name)
		}
		if t.Kind != ops.KindQ4K && t.Kind != ops.KindQ6K && t.Kind != ops.KindF32 {
			return ops.Matrix{}, fmt.Errorf("%w: tensor %q kind %d", ErrInvalidWeights, name, t.Kind)
		}
		if len(t.Shape) != 2 || int(t.Shape[0]) != cols || int(t.Shape[1]) != rows {
			return ops.Matrix{}, fmt.Errorf("%w: tensor %q shape %v, want [%d %d]",
				ErrInvalidWeights, name, t.Shape, cols, rows)
		}
		if t.Offset > uint64(len(data)) || t.Size > uint64(len(data))-t.Offset {
			return ops.Matrix{}, fmt.Errorf("%w: tensor %q outside backing", ErrInvalidWeights, name)
		}
		m := ops.Matrix{
			Data: data[t.Offset : t.Offset+t.Size],
			Rows: rows,
			Cols: cols,
			Kind: t.Kind,
		}
		if err := m.Validate(); err != nil {
			return ops.Matrix{}, fmt.Errorf("%w: tensor %q: %v", ErrInvalidWeights, name, err)
		}
		return m, nil
	}
	vector := func(name string, length int) ([]float32, error) {
		t, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("%w: missing tensor %q", ErrInvalidWeights, name)
		}
		if t.Kind != ops.KindF32 {
			return nil, fmt.Errorf("%w: norm tensor %q kind %d, want F32", ErrInvalidWeights, name, t.Kind)
		}
		if len(t.Shape) != 1 || int(t.Shape[0]) != length {
			return nil, fmt.Errorf("%w: norm tensor %q shape %v, want [%d]",
				ErrInvalidWeights, name, t.Shape, length)
		}
		if t.Size != uint64(length*4) || t.Offset > uint64(len(data)) || t.Size > uint64(len(data))-t.Offset {
			return nil, fmt.Errorf("%w: norm tensor %q outside backing", ErrInvalidWeights, name)
		}
		raw := data[t.Offset : t.Offset+t.Size]
		out := make([]float32, length)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	}

	w := &Weights{}
	vocab, err := vocabSize(byName)
	if err != nil {
		return nil, err
	}

	if w.Embedding, err = matrix("token_embd.weight", vocab, cfg.Hidden); err != nil {
		return nil, err
	}
	if w.Output, err = matrix("output.weight", vocab, cfg.Hidden); err != nil {
		return nil, err
	}
	if w.OutputNorm, err = vector("output_norm.weight", cfg.Hidden); err != nil {
		return nil, err
	}

	w.Layers = make([]LayerWeights, cfg.Layers)
	for i := 0; i < cfg.Layers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		lw := &w.Layers[i]
		if lw.AttnNorm, err = vector(prefix+"attn_norm.weight", cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.Q, err = matrix(prefix+"attn_q.weight", cfg.QueryWidth, cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.K, err = matrix(prefix+"attn_k.weight", cfg.KVWidth, cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.V, err = matrix(prefix+"attn_v.weight", cfg.KVWidth, cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.QNorm, err = vector(prefix+"attn_q_norm.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if lw.KNorm, err = vector(prefix+"attn_k_norm.weight", cfg.HeadDim); err != nil {
			return nil, err
		}
		if lw.AttnOut, err = matrix(prefix+"attn_output.weight", cfg.Hidden, cfg.QueryWidth); err != nil {
			return nil, err
		}
		if lw.FFNNorm, err = vector(prefix+"ffn_norm.weight", cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.Gate, err = matrix(prefix+"ffn_gate.weight", cfg.FFN, cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.Up, err = matrix(prefix+"ffn_up.weight", cfg.FFN, cfg.Hidden); err != nil {
			return nil, err
		}
		if lw.Down, err = matrix(prefix+"ffn_down.weight", cfg.Hidden, cfg.FFN); err != nil {
			return nil, err
		}
	}
	return w, nil
}

func vocabSize(byName map[string]loader.Tensor) (int, error) {
	t, ok := byName["token_embd.weight"]
	if !ok {
		return 0, fmt.Errorf("%w: missing tensor %q", ErrInvalidWeights, "token_embd.weight")
	}
	if len(t.Shape) != 2 {
		return 0, fmt.Errorf("%w: embedding shape %v", ErrInvalidWeights, t.Shape)
	}
	vocab := t.Shape[1]
	if vocab == 0 || vocab > uint64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("%w: invalid vocabulary size %d", ErrInvalidWeights, vocab)
	}
	return int(vocab), nil
}
