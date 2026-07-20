package manboml

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func writeTinyModel(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, tinyGGUF(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func openTiny(t *testing.T, opts Options) *Model {
	t.Helper()
	if opts.ContextSize == 0 {
		opts.ContextSize = 32
	}
	m, err := Open(writeTinyModel(t), opts)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func TestOpenMissingFile(t *testing.T) {
	if _, err := Open(filepath.Join(t.TempDir(), "missing.gguf"), Options{}); err == nil {
		t.Fatal("missing file opened")
	}
}

func TestOpenInfoAndGenerate(t *testing.T) {
	m := openTiny(t, Options{})
	info := m.Info()
	if info.Architecture != "qwen3" || info.ContextSize != 32 {
		t.Fatalf("info = %+v", info)
	}
	if info.VocabSize != 8 || info.MaxConcurrent != 1 {
		t.Fatalf("info = %+v", info)
	}
	if info.ModelBytes == 0 || info.SessionBytes == 0 {
		t.Fatalf("info = %+v", info)
	}

	r1, err := m.Generate(context.Background(), GenerateRequest{Prompt: "ab", MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if r1.GeneratedTokens > 4 {
		t.Fatalf("tokens = %d", r1.GeneratedTokens)
	}
	r2, err := m.Generate(context.Background(), GenerateRequest{Prompt: "ab", MaxTokens: 4})
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Fatalf("nondeterministic: %+v != %+v", r1, r2)
	}
}

func TestGenerateValidation(t *testing.T) {
	m := openTiny(t, Options{})
	if _, err := m.Generate(context.Background(), GenerateRequest{Prompt: "ab"}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("err = %v", err)
	}
	long := strings.Repeat("a", 40)
	if _, err := m.Generate(context.Background(), GenerateRequest{Prompt: long, MaxTokens: 1}); !errors.Is(err, ErrContextLimit) {
		t.Fatalf("err = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate(ctx, GenerateRequest{Prompt: "ab", MaxTokens: 1}); err == nil {
		t.Fatal("canceled context accepted")
	}
}

func TestChatUnsupportedTemplate(t *testing.T) {
	m := openTiny(t, Options{})
	_, err := m.Chat(context.Background(), ChatRequest{
		Messages:  []Message{{Role: RoleUser, Content: "hi"}},
		MaxTokens: 1,
	})
	if !errors.Is(err, ErrUnsupportedTemplate) {
		t.Fatalf("err = %v", err)
	}
}

func TestConcurrentGenerate(t *testing.T) {
	m := openTiny(t, Options{MaxConcurrent: 2})
	var wg sync.WaitGroup
	errs := make([]error, 6)
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, errs[i] = m.Generate(context.Background(), GenerateRequest{Prompt: "ab", MaxTokens: 2})
		}(i)
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			t.Fatalf("request %d: %v", i, err)
		}
	}
}

func TestClose(t *testing.T) {
	m := openTiny(t, Options{})
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal("second Close failed")
	}
	if _, err := m.Generate(context.Background(), GenerateRequest{Prompt: "ab", MaxTokens: 1}); !errors.Is(err, ErrClosed) {
		t.Fatalf("err = %v", err)
	}
}

func TestChecksum(t *testing.T) {
	path := writeTinyModel(t)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	good := hex.EncodeToString(sum[:])

	m, err := Open(path, Options{ContextSize: 32, ExpectedSHA256: strings.ToUpper(good)})
	if err != nil {
		t.Fatalf("valid checksum rejected: %v", err)
	}
	m.Close()

	wrong := strings.Repeat("00", 32)
	if _, err := Open(path, Options{ContextSize: 32, ExpectedSHA256: wrong}); !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("err = %v", err)
	}
	if _, err := Open(path, Options{ContextSize: 32, ExpectedSHA256: "nothex"}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("err = %v", err)
	}
}

func TestMemoryLimit(t *testing.T) {
	path := writeTinyModel(t)
	if _, err := Open(path, Options{ContextSize: 32, MemoryLimit: 1}); !errors.Is(err, ErrMemoryLimit) {
		t.Fatalf("err = %v", err)
	}
}

func TestEstimate(t *testing.T) {
	path := writeTinyModel(t)
	est, err := Estimate(path, Options{ContextSize: 32, MaxConcurrent: 3})
	if err != nil {
		t.Fatal(err)
	}
	if est.ModelBytes == 0 || est.SessionBytes == 0 || est.Sessions != 3 {
		t.Fatalf("estimate = %+v", est)
	}
	if est.TotalBytes != est.ModelBytes+3*est.SessionBytes {
		t.Fatalf("total = %d", est.TotalBytes)
	}
}
