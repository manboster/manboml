package qwen3

import (
	"testing"
)

func TestKVBytesFormula(t *testing.T) {
	cfg := Config{
		Layers:        28,
		KVHeads:       8,
		HeadDim:       128,
		ContextLength: 32768,
	}
	for context, want := range map[int]uint64{
		1024: 117440512,
		2048: 234881024,
		4096: 469762048,
		8192: 939524096,
	} {
		got, err := kvBytes(cfg, context)
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Fatalf("context %d: got %d, want %d", context, got, want)
		}
	}
	if _, err := kvBytes(cfg, 32769); err == nil {
		t.Fatal("context above model maximum accepted")
	}
	if _, err := kvBytes(cfg, 0); err == nil {
		t.Fatal("zero context accepted")
	}
}

func newTestModel(t *testing.T) *Model {
	t.Helper()
	f := defaultFixture()
	raw := buildTinyGGUF(f)
	lm := parseFixture(t, f)
	m, err := NewModel(raw, lm, 32, 1)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNewModelDefaults(t *testing.T) {
	m := newTestModel(t)
	if m.ContextSize() != 32 {
		t.Fatalf("context = %d", m.ContextSize())
	}
	if m.VocabSize() != 8 {
		t.Fatalf("vocab = %d", m.VocabSize())
	}
	if m.Config().Layers != 2 {
		t.Fatalf("layers = %d", m.Config().Layers)
	}
}

func TestSessionLifecycle(t *testing.T) {
	m := newTestModel(t)
	s, err := m.NewSession()
	if err != nil {
		t.Fatal(err)
	}
	if s.Position() != 0 {
		t.Fatalf("initial position = %d", s.Position())
	}
	s.pos = 7
	s.Reset()
	if s.Position() != 0 {
		t.Fatalf("position after reset = %d", s.Position())
	}
	if len(s.kv) != 2 {
		t.Fatalf("kv layers = %d", len(s.kv))
	}
	wantKV := 2 * 32 * 64
	if len(s.kv[0].keys) != wantKV || len(s.kv[0].values) != wantKV {
		t.Fatalf("kv slices = %d, %d", len(s.kv[0].keys), len(s.kv[0].values))
	}
}

func TestSessionBytes(t *testing.T) {
	m := newTestModel(t)
	total := m.SessionBytes()
	kv, err := kvBytes(m.cfg, m.ContextSize())
	if err != nil {
		t.Fatal(err)
	}
	if total <= kv {
		t.Fatalf("session bytes %d not above KV %d", total, kv)
	}
}
