package ops

import (
	"context"
	"sync"
)

// Executor runs row-partitioned numerical work with a bounded number of
// goroutines. Partitioning only affects which goroutine computes an output;
// every output row is computed entirely by one worker in increasing block
// order, so results are identical for any worker count.
type Executor struct {
	workers int
}

// NewExecutor returns an Executor. workers <= 1 runs everything inline.
func NewExecutor(workers int) *Executor {
	if workers < 1 {
		workers = 1
	}
	return &Executor{workers: workers}
}

// Workers returns the configured worker count.
func (e *Executor) Workers() int { return e.workers }

// run executes fn over contiguous [start, end) partitions of [0, n).
func (e *Executor) run(ctx context.Context, n int, fn func(ctx context.Context, start, end int) error) error {
	if n <= 0 {
		return nil
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	parts := e.workers
	if parts > n {
		parts = n
	}
	if parts <= 1 {
		return fn(ctx, 0, n)
	}

	var wg sync.WaitGroup
	errs := make([]error, parts)
	base, extra := n/parts, n%parts
	start := 0
	for p := 0; p < parts; p++ {
		size := base
		if p < extra {
			size++
		}
		s := start
		start += size
		if size == 0 {
			continue
		}
		wg.Add(1)
		go func(p, s, size int) {
			defer wg.Done()
			errs[p] = fn(ctx, s, s+size)
		}(p, s, size)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}
