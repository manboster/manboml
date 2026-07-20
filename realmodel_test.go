package manboml

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestRealModelEndToEnd exercises the pinned Qwen3Guard-Gen-0.6B Q4_K_M
// artifact through the public API. It requires MANBOML_TEST_GGUF to point at
// the exact pinned file; it never downloads anything.
func TestRealModelEndToEnd(t *testing.T) {
	path := realModelPath(t)
	m, err := Open(path, Options{
		ContextSize:    2048,
		MaxConcurrent:  1,
		ExpectedSHA256: realModelSHA256,
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer m.Close()

	info := m.Info()
	if info.ContextSize != 2048 || info.VocabSize != 151936 {
		t.Fatalf("info = %+v", info)
	}
	if info.ModelBytes != 484219904 {
		t.Fatalf("model bytes = %d", info.ModelBytes)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	res, err := m.Chat(ctx, ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "How are you?"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	t.Logf("moderation output: %q", res.Text)
	if !strings.Contains(res.Text, "Safety:") {
		t.Fatalf("output lacks Safety verdict: %q", res.Text)
	}
	if res.FinishReason != FinishEOG && res.FinishReason != FinishMaxTokens {
		t.Fatalf("finish = %q", res.FinishReason)
	}
}

func TestRealModelEstimate(t *testing.T) {
	path := realModelPath(t)
	est, err := Estimate(path, Options{ContextSize: 2048, MaxConcurrent: 1})
	if err != nil {
		t.Fatal(err)
	}
	if est.ModelBytes != 484219904 {
		t.Fatalf("model bytes = %d", est.ModelBytes)
	}
	if est.SessionBytes < 230_000_000 || est.SessionBytes > 240_000_000 {
		t.Fatalf("session bytes = %d", est.SessionBytes)
	}
}
