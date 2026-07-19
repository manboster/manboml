package qwen3

import (
	"context"
	"math"
	"testing"
)

func forwardTokens(t *testing.T, s *Session, tokens []int32) {
	t.Helper()
	for i, token := range tokens {
		if err := s.Forward(context.Background(), token, i == len(tokens)-1); err != nil {
			t.Fatalf("forward token %d: %v", i, err)
		}
	}
}

func TestForwardDeterminism(t *testing.T) {
	m := newTestModel(t)
	tokens := []int32{1, 2, 3}

	s1, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s1, tokens)
	want := append([]float32(nil), s1.Logits()...)

	s2, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s2, tokens)
	for i := range want {
		if math.Float32bits(s2.Logits()[i]) != math.Float32bits(want[i]) {
			t.Fatalf("logit %d differs: %v != %v", i, s2.Logits()[i], want[i])
		}
	}
}

func TestForwardWorkerDeterminism(t *testing.T) {
	f := defaultFixture()
	raw := buildTinyGGUF(f)
	lm := parseFixture(t, f)
	tokens := []int32{1, 2, 3}

	m1, err := NewModel(raw, lm, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	s1, err := m1.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s1, tokens)
	want := append([]float32(nil), s1.Logits()...)

	m4, err := NewModel(raw, lm, 32, 4)
	if err != nil {
		t.Fatal(err)
	}
	s4, err := m4.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s4, tokens)
	for i := range want {
		if math.Float32bits(s4.Logits()[i]) != math.Float32bits(want[i]) {
			t.Fatalf("worker mismatch at logit %d: %v != %v", i, s4.Logits()[i], want[i])
		}
	}
}

func TestForwardResetIndependence(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s, []int32{1, 2, 3})
	s.Reset()
	forwardTokens(t, s, []int32{5})
	got := append([]float32(nil), s.Logits()...)

	fresh, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, fresh, []int32{5})
	for i := range got {
		if math.Float32bits(fresh.Logits()[i]) != math.Float32bits(got[i]) {
			t.Fatalf("reset leaked state at logit %d", i)
		}
	}
}

func TestForwardContextLimit(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if err := s.Forward(context.Background(), 1, false); err != nil {
			t.Fatalf("token %d: %v", i, err)
		}
	}
	if err := s.Forward(context.Background(), 1, true); err == nil {
		t.Fatal("position beyond capacity accepted")
	}
}

func TestForwardTokenValidation(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Forward(context.Background(), -1, false); err == nil {
		t.Fatal("negative token accepted")
	}
	if err := s.Forward(context.Background(), 8, false); err == nil {
		t.Fatal("out-of-vocabulary token accepted")
	}
}

func TestForwardCancellation(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := s.Forward(ctx, 1, false); err == nil {
		t.Fatal("canceled context accepted")
	}
	if s.Position() != 0 {
		t.Fatalf("position advanced after cancellation: %d", s.Position())
	}
}

func TestForwardLogitsFinite(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	forwardTokens(t, s, []int32{1, 2, 3, 4})
	for i, v := range s.Logits() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit %d not finite: %v", i, v)
		}
	}
}
