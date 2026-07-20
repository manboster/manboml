package manboml

import (
	"fmt"

	"github.com/manboster/manboml/internal/arch/qwen3"
	"github.com/manboster/manboml/internal/backing"
	"github.com/manboster/manboml/internal/loader"
)

// MemoryEstimate breaks down the conservative memory plan for a model and
// option set. It counts ManboML-managed payloads, not exact OS RSS.
type MemoryEstimate struct {
	// ModelBytes is the full GGUF file size. With mmap it is mapped rather
	// than heap-resident, but it is counted conservatively either way.
	ModelBytes uint64
	// SessionBytes is the per-session payload: F16 KV cache, numerical
	// scratch, logits and activation storage.
	SessionBytes uint64
	// Sessions is the number of eagerly allocated sessions.
	Sessions int
	// TotalBytes is ModelBytes plus Sessions*SessionBytes, excluding
	// tokenizer tables and Go runtime overhead.
	TotalBytes uint64
}

// Estimate computes the memory plan for path and opts without allocating
// sessions. It performs the same structural validation as Open.
func Estimate(path string, opts Options) (MemoryEstimate, error) {
	opts = opts.withDefaults()
	b, err := backing.Open(path)
	if err != nil {
		return MemoryEstimate{}, err
	}
	defer b.Close()

	lm, err := loader.Parse(b.Bytes())
	if err != nil {
		return MemoryEstimate{}, fmt.Errorf("%w: %v", ErrInvalidModel, err)
	}
	if lm.Architecture != "qwen3" {
		return MemoryEstimate{}, fmt.Errorf("%w: architecture %q", ErrUnsupportedModel, lm.Architecture)
	}
	arch, err := qwen3.NewModel(b.Bytes(), lm, opts.ContextSize, 1)
	if err != nil {
		return MemoryEstimate{}, err
	}
	session := arch.SessionBytes()
	total := uint64(len(b.Bytes())) + session*uint64(opts.MaxConcurrent)
	return MemoryEstimate{
		ModelBytes:   uint64(len(b.Bytes())),
		SessionBytes: session,
		Sessions:     opts.MaxConcurrent,
		TotalBytes:   total,
	}, nil
}
