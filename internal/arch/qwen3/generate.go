package qwen3

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/manboster/manboml/internal/ops"
	"github.com/manboster/manboml/internal/tokenizer"
)

// FinishReason describes why generation stopped.
type FinishReason string

const (
	// FinishEOG means an end-of-generation token was selected.
	FinishEOG FinishReason = "eog"
	// FinishMaxTokens means the caller-provided token bound was reached.
	FinishMaxTokens FinishReason = "max_tokens"
)

// ErrInvalidRequest describes an invalid generation request.
var ErrInvalidRequest = errors.New("qwen3: invalid request")

// Result is the outcome of one generation.
type Result struct {
	Text   string
	Tokens int
	Finish FinishReason
}

// Generate tokenizes prompt, prefills it and greedily generates at most
// maxTokens new tokens. Generation stops early at an end-of-generation
// token. The session is reset before use.
func (m *Model) Generate(ctx context.Context, tok *tokenizer.Tokenizer, prompt string, maxTokens int) (Result, error) {
	if maxTokens <= 0 {
		return Result{}, fmt.Errorf("%w: maxTokens must be positive", ErrInvalidRequest)
	}
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	ids, err := tok.Encode(prompt)
	if err != nil {
		return Result{}, err
	}
	if len(ids) == 0 {
		return Result{}, fmt.Errorf("%w: prompt produced no tokens", ErrInvalidRequest)
	}
	if len(ids)+maxTokens > m.contextSize {
		return Result{}, fmt.Errorf("%w: %d prompt tokens plus %d new tokens exceed context %d",
			ErrContextLimit, len(ids), maxTokens, m.contextSize)
	}

	s, err := m.NewSession()
	if err != nil {
		return Result{}, err
	}
	s.Reset()
	for i, id := range ids {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		if err := s.Forward(ctx, id, i == len(ids)-1); err != nil {
			return Result{}, err
		}
	}
	return m.generate(ctx, tok, s, maxTokens)
}

// generate runs the greedy sampling loop over a session whose logits reflect
// the last evaluated token.
func (m *Model) generate(ctx context.Context, tok *tokenizer.Tokenizer, s *Session, maxTokens int) (Result, error) {
	var sb strings.Builder
	generated := 0
	finish := FinishMaxTokens
	for generated < maxTokens {
		next := int32(ops.Argmax(s.Logits()))
		if next < 0 {
			return Result{}, fmt.Errorf("%w: no logits available", ErrInvalidMetadata)
		}
		if tok.IsEOG(next) {
			finish = FinishEOG
			break
		}
		piece, err := tok.Decode([]int32{next})
		if err != nil {
			return Result{}, err
		}
		sb.WriteString(piece)
		generated++
		if generated < maxTokens {
			if err := s.Forward(ctx, next, true); err != nil {
				return Result{}, err
			}
		}
	}
	return Result{Text: sb.String(), Tokens: generated, Finish: finish}, nil
}
