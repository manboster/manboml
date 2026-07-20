package manboml

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/manboster/manboml/internal/arch/qwen3"
	"github.com/manboster/manboml/internal/backing"
	"github.com/manboster/manboml/internal/loader"
	"github.com/manboster/manboml/internal/tokenizer"
)

// Open opens a supported GGUF model file. It validates the file structure,
// tokenizer, and architecture contract, verifies the optional checksum,
// admits the memory plan, and eagerly allocates all configured sessions.
// The caller must Close the model.
func Open(path string, opts Options) (*Model, error) {
	opts = opts.withDefaults()
	if opts.ContextSize < 0 || opts.MaxConcurrent < 0 || opts.Workers < 0 {
		return nil, fmt.Errorf("%w: negative option", ErrInvalidRequest)
	}

	b, err := backing.Open(path)
	if err != nil {
		return nil, err
	}
	cleanup := true
	defer func() {
		if cleanup {
			b.Close()
		}
	}()
	if err := b.CheckUnchanged(path); err != nil {
		return nil, err
	}

	if opts.ExpectedSHA256 != "" {
		want, err := hex.DecodeString(strings.ToLower(strings.TrimSpace(opts.ExpectedSHA256)))
		if err != nil || len(want) != sha256.Size {
			return nil, fmt.Errorf("%w: ExpectedSHA256 is not a hex SHA-256 digest", ErrInvalidRequest)
		}
		sum := sha256.Sum256(b.Bytes())
		if !strings.EqualFold(hex.EncodeToString(sum[:]), hex.EncodeToString(want)) {
			return nil, fmt.Errorf("%w: file digest does not match ExpectedSHA256", ErrChecksumMismatch)
		}
	}

	lm, err := loader.Parse(b.Bytes())
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidModel, err)
	}
	if lm.Architecture != "qwen3" {
		return nil, fmt.Errorf("%w: architecture %q", ErrUnsupportedModel, lm.Architecture)
	}
	tok, err := tokenizer.New(lm.Tokenizer)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidModel, err)
	}
	arch, err := qwen3.NewModel(b.Bytes(), lm, opts.ContextSize, opts.Workers)
	if err != nil {
		return nil, err
	}

	sessionBytes := arch.SessionBytes()
	total := uint64(len(b.Bytes())) + sessionBytes*uint64(opts.MaxConcurrent)
	if opts.MemoryLimit != 0 && total > opts.MemoryLimit {
		return nil, fmt.Errorf("%w: model and %d sessions need %d bytes, limit is %d",
			ErrMemoryLimit, opts.MaxConcurrent, total, opts.MemoryLimit)
	}

	m := &Model{
		backing:  b,
		arch:     arch,
		tok:      tok,
		sessions: make(chan *qwen3.Session, opts.MaxConcurrent),
		done:     make(chan struct{}),
		info: ModelInfo{
			Architecture:  lm.Architecture,
			ContextSize:   arch.ContextSize(),
			MaxConcurrent: opts.MaxConcurrent,
			VocabSize:     arch.VocabSize(),
			ModelBytes:    uint64(len(b.Bytes())),
			SessionBytes:  sessionBytes,
		},
	}
	for i := 0; i < opts.MaxConcurrent; i++ {
		s, err := arch.NewSession()
		if err != nil {
			return nil, err
		}
		m.sessions <- s
	}
	cleanup = false
	return m, nil
}
