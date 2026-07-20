package manboml

// FinishReason describes why generation stopped.
type FinishReason string

const (
	// FinishEOG means an end-of-generation token was selected.
	FinishEOG FinishReason = "eog"
	// FinishMaxTokens means the request's MaxTokens bound was reached.
	FinishMaxTokens FinishReason = "max_tokens"
)

// Result is one completed generation.
type Result struct {
	// Text is the decoded generated text, excluding special tokens that
	// terminated generation.
	Text string
	// GeneratedTokens is the number of new tokens produced.
	GeneratedTokens int
	// FinishReason reports why generation stopped.
	FinishReason FinishReason
}
