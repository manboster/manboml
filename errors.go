package manboml

import (
	"errors"

	"github.com/manboster/manboml/internal/arch/qwen3"
	"github.com/manboster/manboml/internal/tokenizer"
)

var (
	// ErrClosed reports use of a closed model.
	ErrClosed = errors.New("manboml: model closed")
	// ErrInvalidRequest reports an invalid generation or chat request.
	ErrInvalidRequest = errors.New("manboml: invalid request")
	// ErrChecksumMismatch reports an ExpectedSHA256 mismatch.
	ErrChecksumMismatch = errors.New("manboml: model checksum mismatch")
	// ErrMemoryLimit reports that the configured memory budget is too small.
	ErrMemoryLimit = errors.New("manboml: memory limit exceeded")
	// ErrInvalidModel reports a malformed or inconsistent model file.
	ErrInvalidModel = errors.New("manboml: invalid model")

	// ErrUnsupportedModel reports an unsupported architecture or model
	// variant.
	ErrUnsupportedModel = qwen3.ErrUnsupportedModel
	// ErrContextLimit reports that the request does not fit the context
	// window.
	ErrContextLimit = qwen3.ErrContextLimit
	// ErrUnsupportedTemplate reports an unrecognized embedded chat template.
	ErrUnsupportedTemplate = tokenizer.ErrUnsupportedTemplate
)
