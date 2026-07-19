package qwen3

import (
	"context"
	"strings"
	"testing"

	"github.com/manboster/manboml/internal/tokenizer"
)

func newTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	m := parseFixture(t, defaultFixture())
	tk, err := tokenizer.New(m.Tokenizer)
	if err != nil {
		t.Fatal(err)
	}
	return tk
}

func TestGenerateMaxTokens(t *testing.T) {
	m := newTestModel(t)
	tk := newTestTokenizer(t)
	res, err := m.Generate(context.Background(), tk, "ab", 3)
	if err != nil {
		t.Fatal(err)
	}
	if res.Tokens != 3 || res.Finish != FinishMaxTokens {
		t.Fatalf("got %+v", res)
	}
	if len(res.Text) == 0 {
		t.Fatal("empty output text")
	}
}

func TestGenerateDeterminism(t *testing.T) {
	m := newTestModel(t)
	tk := newTestTokenizer(t)
	r1, err := m.Generate(context.Background(), tk, "abc", 4)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := m.Generate(context.Background(), tk, "abc", 4)
	if err != nil {
		t.Fatal(err)
	}
	if r1 != r2 {
		t.Fatalf("nondeterministic: %+v != %+v", r1, r2)
	}
}

func TestGenerateEOG(t *testing.T) {
	m := newTestModel(t)
	tk := newTestTokenizer(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	s.logits[m.VocabSize()-1] = 1e9
	res, err := m.generate(context.Background(), tk, s, 8)
	if err != nil {
		t.Fatal(err)
	}
	if res.Finish != FinishEOG || res.Tokens != 0 || res.Text != "" {
		t.Fatalf("got %+v", res)
	}
}

func TestGenerateThenEOGAfterTokens(t *testing.T) {
	m := newTestModel(t)
	tk := newTestTokenizer(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	s.logits[2] = 1e9
	first := append([]float32(nil), s.logits...)
	res, err := m.generate(context.Background(), tk, s, 1)
	if err != nil {
		t.Fatal(err)
	}
	if res.Finish != FinishMaxTokens || res.Tokens != 1 || res.Text != "c" {
		t.Fatalf("got %+v", res)
	}
	_ = first
}

func TestGenerateValidation(t *testing.T) {
	m := newTestModel(t)
	tk := newTestTokenizer(t)
	if _, err := m.Generate(context.Background(), tk, "ab", 0); err == nil {
		t.Fatal("zero maxTokens accepted")
	}
	if _, err := m.Generate(context.Background(), tk, "", 1); err == nil {
		t.Fatal("empty prompt accepted")
	}
	long := strings.Repeat("a", 40)
	if _, err := m.Generate(context.Background(), tk, long, 1); err == nil {
		t.Fatal("overlong prompt accepted")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.Generate(ctx, tk, "ab", 1); err == nil {
		t.Fatal("canceled context accepted")
	}
}
