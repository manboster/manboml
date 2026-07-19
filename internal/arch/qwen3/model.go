package qwen3

import (
	"fmt"
	"runtime"

	"github.com/manboster/manboml/internal/loader"
	"github.com/manboster/manboml/internal/ops"
)

// DefaultContextSize is the default runtime context window.
const DefaultContextSize = 2048

// Model is an immutable Qwen3 model: validated configuration, packed weights,
// a shared RoPE table and a worker executor. It is safe for concurrent use;
// mutable inference state lives in Sessions.
type Model struct {
	cfg         Config
	weights     *Weights
	rope        *ops.RoPETable
	executor    *ops.Executor
	vocab       int
	contextSize int
}

// NewModel binds a Qwen3 model from parsed GGUF data. data must remain valid
// for the model's lifetime. contextSize of 0 selects DefaultContextSize;
// workers of 0 captures the current GOMAXPROCS without changing it.
func NewModel(data []byte, lm *loader.Model, contextSize, workers int) (*Model, error) {
	cfg, err := NewConfig(lm)
	if err != nil {
		return nil, err
	}
	if contextSize == 0 {
		contextSize = DefaultContextSize
	}
	if contextSize < 0 || contextSize > cfg.ContextLength {
		return nil, fmt.Errorf("%w: context size %d (model maximum %d)",
			ErrContextLimit, contextSize, cfg.ContextLength)
	}
	if cfg.Hidden%ops.Q8KBlockSize != 0 ||
		cfg.QueryHeads*cfg.HeadDim%ops.Q8KBlockSize != 0 ||
		cfg.FFN%ops.Q8KBlockSize != 0 {
		return nil, fmt.Errorf("%w: dimensions must be multiples of %d for Q8_K activations",
			ErrUnsupportedModel, ops.Q8KBlockSize)
	}
	if _, err := kvBytes(cfg, contextSize); err != nil {
		return nil, err
	}

	weights, err := BindWeights(data, lm.Tensors, cfg)
	if err != nil {
		return nil, err
	}
	rope, err := ops.NewRoPETable(contextSize, cfg.HeadDim, cfg.RopeDim, cfg.RopeBase)
	if err != nil {
		return nil, err
	}
	if workers == 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	return &Model{
		cfg:         cfg,
		weights:     weights,
		rope:        rope,
		executor:    ops.NewExecutor(workers),
		vocab:       weights.Embedding.Rows,
		contextSize: contextSize,
	}, nil
}

// Config returns the model's architecture configuration.
func (m *Model) Config() Config { return m.cfg }

// ContextSize returns the configured runtime context window.
func (m *Model) ContextSize() int { return m.contextSize }

// VocabSize returns the vocabulary size from the embedding tensor.
func (m *Model) VocabSize() int { return m.vocab }

// SessionBytes returns the exact per-session payload in bytes: F16 KV cache,
// numerical scratch, logits and Q8_K activation storage.
func (m *Model) SessionBytes() uint64 {
	kv, _ := kvBytes(m.cfg, m.contextSize)
	f32 := func(n int) uint64 { return uint64(n) * 4 }
	scratch := f32(m.cfg.Hidden*2) + // hidden, norm
		f32(m.cfg.QueryWidth*2) + // q, attn
		f32(m.cfg.KVWidth*2) + // k, v
		f32(m.cfg.Hidden) + // proj
		f32(m.cfg.FFN*2) + // gate, up
		f32(m.vocab) + // logits
		f32(m.contextSize) // scores
	maxAct := max(m.cfg.Hidden, max(m.cfg.QueryWidth, m.cfg.FFN))
	q8 := f32(maxAct/ops.Q8KBlockSize) + // scales
		uint64(maxAct) + // qs int8
		uint64(maxAct/ops.Q8KBlockSize*ops.Q8KGroupCount*2) // bsums int16
	return kv + scratch + q8
}

// NewSession allocates an independent inference session with its own KV
// cache and scratch space. A session is not safe for concurrent use.
func (m *Model) NewSession() (*Session, error) {
	kv, err := newKVCache(m.cfg, m.contextSize)
	if err != nil {
		return nil, err
	}
	maxAct := max(m.cfg.Hidden, max(m.cfg.QueryWidth, m.cfg.FFN))
	return &Session{
		model:    m,
		kv:       kv,
		capacity: m.contextSize,
		hidden:   make([]float32, m.cfg.Hidden),
		norm:     make([]float32, m.cfg.Hidden),
		q:        make([]float32, m.cfg.QueryWidth),
		k:        make([]float32, m.cfg.KVWidth),
		v:        make([]float32, m.cfg.KVWidth),
		attn:     make([]float32, m.cfg.QueryWidth),
		proj:     make([]float32, m.cfg.Hidden),
		gate:     make([]float32, m.cfg.FFN),
		up:       make([]float32, m.cfg.FFN),
		logits:   make([]float32, m.vocab),
		scores:   make([]float32, m.contextSize),
		q8:       ops.NewQ8K(maxAct),
	}, nil
}

// Session holds mutable per-request inference state: KV cache, scratch
// buffers and the committed position.
type Session struct {
	model    *Model
	kv       []kvLayer
	pos      int
	capacity int

	hidden []float32
	norm   []float32
	q      []float32
	k      []float32
	v      []float32
	attn   []float32
	proj   []float32
	gate   []float32
	up     []float32
	logits []float32
	scores []float32
	q8     ops.Q8K
}

// Reset makes the session reusable for an independent request. It only
// resets the logical position; cached bytes are never read past the
// committed position, so clearing them is unnecessary.
func (s *Session) Reset() {
	s.pos = 0
}

// Position returns the committed sequence length.
func (s *Session) Position() int { return s.pos }
