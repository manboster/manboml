// Package qwen3 implements the Qwen3 decoder-only Transformer architecture
// over ManboML's loader and operator packages. All dimensions come from GGUF
// metadata; nothing is hard-coded to a specific model size.
package qwen3

import (
	"errors"
	"fmt"

	"github.com/manboster/manboml/internal/loader"
)

var (
	ErrUnsupportedModel = errors.New("qwen3: unsupported model")
	ErrInvalidMetadata  = errors.New("qwen3: invalid metadata")
	ErrInvalidWeights   = errors.New("qwen3: invalid weights")
	ErrContextLimit     = errors.New("qwen3: context limit exceeded")
)

// Config describes a Qwen3 model's architecture, derived entirely from GGUF
// metadata.
type Config struct {
	Layers        int
	Hidden        int
	FFN           int
	QueryHeads    int
	KVHeads       int
	HeadDim       int
	RopeDim       int
	RopeBase      float64
	NormEps       float32
	ContextLength int

	QueryWidth int
	KVWidth    int
}

// NewConfig extracts and validates the Qwen3 architecture configuration.
func NewConfig(m *loader.Model) (Config, error) {
	if m.Architecture != "qwen3" {
		return Config{}, fmt.Errorf("%w: architecture %q", ErrUnsupportedModel, m.Architecture)
	}
	cfg := Config{
		RopeBase: 1e6,
		NormEps:  1e-6,
	}
	var err error
	if cfg.Layers, err = requiredInt(m, "block_count"); err != nil {
		return Config{}, err
	}
	if cfg.Hidden, err = requiredInt(m, "embedding_length"); err != nil {
		return Config{}, err
	}
	if cfg.FFN, err = requiredInt(m, "feed_forward_length"); err != nil {
		return Config{}, err
	}
	if cfg.QueryHeads, err = requiredInt(m, "attention.head_count"); err != nil {
		return Config{}, err
	}
	if cfg.KVHeads, err = requiredInt(m, "attention.head_count_kv"); err != nil {
		return Config{}, err
	}
	if cfg.HeadDim, err = requiredInt(m, "attention.key_length"); err != nil {
		return Config{}, err
	}
	if cfg.ContextLength, err = requiredInt(m, "context_length"); err != nil {
		return Config{}, err
	}

	if v, ok := m.Uint("attention.value_length"); ok {
		if int(v) != cfg.HeadDim {
			return Config{}, fmt.Errorf("%w: value_length %d differs from key_length %d",
				ErrUnsupportedModel, v, cfg.HeadDim)
		}
	}
	cfg.RopeDim = cfg.HeadDim
	if v, ok := m.Uint("rope.dimension_count"); ok {
		cfg.RopeDim = int(v)
	}
	if v, ok := m.Float("rope.freq_base"); ok {
		cfg.RopeBase = v
	}
	if v, ok := m.Float("attention.layer_norm_rms_epsilon"); ok {
		cfg.NormEps = float32(v)
	}
	if _, ok := m.Uint("rope.scaling.type"); ok {
		return Config{}, fmt.Errorf("%w: RoPE scaling", ErrUnsupportedModel)
	}

	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	cfg.QueryWidth = cfg.QueryHeads * cfg.HeadDim
	cfg.KVWidth = cfg.KVHeads * cfg.HeadDim
	return cfg, nil
}

func (c Config) validate() error {
	if c.Layers <= 0 || c.Hidden <= 0 || c.FFN <= 0 {
		return fmt.Errorf("%w: non-positive dimension", ErrInvalidMetadata)
	}
	if c.QueryHeads <= 0 || c.KVHeads <= 0 || c.HeadDim <= 0 || c.ContextLength <= 0 {
		return fmt.Errorf("%w: non-positive attention parameter", ErrInvalidMetadata)
	}
	if c.QueryHeads%c.KVHeads != 0 {
		return fmt.Errorf("%w: %d query heads not divisible by %d KV heads",
			ErrInvalidMetadata, c.QueryHeads, c.KVHeads)
	}
	if c.RopeDim <= 0 || c.RopeDim%2 != 0 || c.RopeDim > c.HeadDim {
		return fmt.Errorf("%w: invalid RoPE dimension %d", ErrInvalidMetadata, c.RopeDim)
	}
	if c.NormEps <= 0 || c.RopeBase <= 0 {
		return fmt.Errorf("%w: invalid norm epsilon or RoPE base", ErrInvalidMetadata)
	}
	return nil
}

func requiredInt(m *loader.Model, key string) (int, error) {
	v, ok := m.Uint(key)
	if !ok {
		return 0, fmt.Errorf("%w: missing %q", ErrInvalidMetadata, key)
	}
	if v > uint64(int(^uint(0)>>1)) {
		return 0, fmt.Errorf("%w: %q overflows int", ErrInvalidMetadata, key)
	}
	return int(v), nil
}
