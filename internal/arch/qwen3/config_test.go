package qwen3

import (
	"testing"

	"github.com/manboster/manboml/internal/loader"
)

func parseFixture(t *testing.T, f fixture) *loader.Model {
	t.Helper()
	m, err := loader.Parse(buildTinyGGUF(f))
	if err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return m
}

func TestNewConfig(t *testing.T) {
	m := parseFixture(t, defaultFixture())
	cfg, err := NewConfig(m)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Layers != 2 || cfg.Hidden != 256 || cfg.FFN != 256 {
		t.Fatalf("dims = %v", cfg)
	}
	if cfg.QueryHeads != 4 || cfg.KVHeads != 2 || cfg.HeadDim != 64 {
		t.Fatalf("heads = %v", cfg)
	}
	if cfg.QueryWidth != 256 || cfg.KVWidth != 128 {
		t.Fatalf("widths = %d, %d", cfg.QueryWidth, cfg.KVWidth)
	}
	if cfg.RopeDim != 64 || cfg.RopeBase != 10000 {
		t.Fatalf("rope = %d, %v", cfg.RopeDim, cfg.RopeBase)
	}
	if cfg.ContextLength != 32 {
		t.Fatalf("context = %d", cfg.ContextLength)
	}
}

func TestNewConfigRejectsWrongArchitecture(t *testing.T) {
	f := defaultFixture()
	raw := buildTinyGGUF(f)
	m, err := loader.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	m.Architecture = "llama"
	if _, err := NewConfig(m); err == nil {
		t.Fatal("llama architecture accepted")
	}
}

func TestBindWeights(t *testing.T) {
	f := defaultFixture()
	m := parseFixture(t, f)
	cfg, err := NewConfig(m)
	if err != nil {
		t.Fatal(err)
	}
	raw := buildTinyGGUF(f)
	w, err := BindWeights(raw, m.Tensors, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(w.Layers) != 2 {
		t.Fatalf("layers = %d", len(w.Layers))
	}
	if w.Embedding.Rows != 8 || w.Embedding.Cols != 256 {
		t.Fatalf("embedding = %dx%d", w.Embedding.Rows, w.Embedding.Cols)
	}
	if w.Output.Rows != 8 || w.Output.Kind != 14 {
		t.Fatalf("output = %dx%d kind %d", w.Output.Rows, w.Output.Cols, w.Output.Kind)
	}
	lw := w.Layers[0]
	if lw.Q.Rows != 256 || lw.Q.Cols != 256 {
		t.Fatalf("q = %dx%d", lw.Q.Rows, lw.Q.Cols)
	}
	if lw.K.Rows != 128 || lw.AttnOut.Cols != 256 || lw.Down.Rows != 256 {
		t.Fatalf("layer shapes wrong")
	}
	if len(lw.AttnNorm) != 256 || len(lw.QNorm) != 64 {
		t.Fatalf("norm lengths wrong")
	}
}

func TestBindWeightsRejectsMissingTensor(t *testing.T) {
	f := defaultFixture()
	m := parseFixture(t, f)
	cfg, err := NewConfig(m)
	if err != nil {
		t.Fatal(err)
	}
	tensors := m.Tensors[:len(m.Tensors)-1]
	raw := buildTinyGGUF(f)
	if _, err := BindWeights(raw, tensors, cfg); err == nil {
		t.Fatal("missing tensor accepted")
	}
}

func TestBindWeightsRejectsShapeMismatch(t *testing.T) {
	f := defaultFixture()
	m := parseFixture(t, f)
	cfg, err := NewConfig(m)
	if err != nil {
		t.Fatal(err)
	}
	for i := range m.Tensors {
		if m.Tensors[i].Name == "token_embd.weight" {
			m.Tensors[i].Shape[1] = 99
		}
	}
	raw := buildTinyGGUF(f)
	if _, err := BindWeights(raw, m.Tensors, cfg); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}
