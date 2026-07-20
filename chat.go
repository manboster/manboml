package manboml

import (
	"context"
	"fmt"

	"github.com/manboster/manboml/internal/tokenizer"
)

// Role is a semantic chat message role.
type Role = tokenizer.Role

const (
	RoleSystem    = tokenizer.RoleSystem
	RoleUser      = tokenizer.RoleUser
	RoleAssistant = tokenizer.RoleAssistant
)

// Message is a semantic chat message.
type Message struct {
	Role    Role
	Content string
}

// ChatRequest is a semantic-message generation request.
type ChatRequest struct {
	// Messages are rendered with the model's recognized chat template.
	Messages []Message
	// MaxTokens bounds the number of new tokens. It must be positive.
	MaxTokens int
}

// Chat formats messages with the model's recognized chat template and
// generates a response. An unrecognized embedded template returns
// ErrUnsupportedTemplate; arbitrary Jinja is never executed.
func (m *Model) Chat(ctx context.Context, req ChatRequest) (Result, error) {
	if req.MaxTokens <= 0 {
		return Result{}, fmt.Errorf("%w: MaxTokens must be positive", ErrInvalidRequest)
	}
	messages := make([]tokenizer.Message, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = tokenizer.Message{Role: msg.Role, Content: msg.Content}
	}
	prompt, err := m.tok.Format(messages)
	if err != nil {
		return Result{}, err
	}
	return m.Generate(ctx, GenerateRequest{Prompt: prompt, MaxTokens: req.MaxTokens})
}
