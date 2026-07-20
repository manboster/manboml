package manboml

import (
	"context"
	"fmt"
)

// GenerateRequest is a raw-prompt generation request.
type GenerateRequest struct {
	// Prompt is an already formatted prompt.
	Prompt string
	// MaxTokens bounds the number of new tokens. It must be positive.
	MaxTokens int
}

// Generate tokenizes the raw prompt and generates at most req.MaxTokens new
// tokens with greedy decoding.
func (m *Model) Generate(ctx context.Context, req GenerateRequest) (Result, error) {
	if req.MaxTokens <= 0 {
		return Result{}, fmt.Errorf("%w: MaxTokens must be positive", ErrInvalidRequest)
	}
	if err := m.begin(); err != nil {
		return Result{}, err
	}
	defer m.end()

	s, err := m.acquire(ctx)
	if err != nil {
		return Result{}, err
	}
	defer m.release(s)

	ids, err := m.tok.Encode(req.Prompt)
	if err != nil {
		return Result{}, err
	}
	res, err := m.arch.GenerateTokens(ctx, m.tok, s, ids, req.MaxTokens)
	if err != nil {
		return Result{}, err
	}
	return Result{
		Text:            res.Text,
		GeneratedTokens: res.Tokens,
		FinishReason:    FinishReason(res.Finish),
	}, nil
}
