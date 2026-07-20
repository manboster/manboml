package manboml

import (
	"context"
	"sync"

	"github.com/manboster/manboml/internal/arch/qwen3"
	"github.com/manboster/manboml/internal/backing"
	"github.com/manboster/manboml/internal/tokenizer"
)

// Model is an opened GGUF model. It owns the file backing, immutable weights,
// tokenizer and a bounded pool of inference sessions. Generate, Chat and Info
// are safe for concurrent use.
type Model struct {
	backing  *backing.Backing
	arch     *qwen3.Model
	tok      *tokenizer.Tokenizer
	sessions chan *qwen3.Session
	done     chan struct{}
	info     ModelInfo

	mu     sync.Mutex
	closed bool
	active sync.WaitGroup
}

// ModelInfo describes an opened model.
type ModelInfo struct {
	Architecture  string
	ContextSize   int
	MaxConcurrent int
	VocabSize     int
	ModelBytes    uint64
	SessionBytes  uint64
}

// Info returns information about the opened model.
func (m *Model) Info() ModelInfo { return m.info }

// Close rejects new work, waits for active requests, and releases all
// resources. It is idempotent.
func (m *Model) Close() error {
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	close(m.done)
	m.mu.Unlock()

	m.active.Wait()
	return m.backing.Close()
}

// begin registers an active request or reports that the model is closed.
func (m *Model) begin() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return ErrClosed
	}
	m.active.Add(1)
	return nil
}

func (m *Model) end() { m.active.Done() }

// acquire leases a session from the pool, honoring cancellation and Close.
func (m *Model) acquire(ctx context.Context) (*qwen3.Session, error) {
	select {
	case s := <-m.sessions:
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-m.done:
		return nil, ErrClosed
	}
}

// release returns a session to the pool.
func (m *Model) release(s *qwen3.Session) {
	s.Reset()
	m.sessions <- s
}
