package ops

import (
	"math"
	"testing"
)

func TestRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	dst := make([]float32, 4)
	if err := RMSNorm(dst, x, w, 1e-6); err != nil {
		t.Fatal(err)
	}
	mean := float32(1+4+9+16) / 4
	scale := float32(1) / float32(math.Sqrt(float64(mean+1e-6)))
	for i := range dst {
		want := x[i] * scale
		if dst[i] != want {
			t.Fatalf("index %d: got %v want %v", i, dst[i], want)
		}
	}

	weighted := []float32{0.5, 2, -1, 3}
	if err := RMSNorm(dst, x, weighted, 1e-6); err != nil {
		t.Fatal(err)
	}
	for i := range dst {
		want := x[i] * scale * weighted[i]
		if dst[i] != want {
			t.Fatalf("weighted index %d: got %v want %v", i, dst[i], want)
		}
	}

	zero := make([]float32, 4)
	if err := RMSNorm(dst, zero, w, 1e-6); err != nil {
		t.Fatal(err)
	}
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("zero input index %d = %v", i, v)
		}
	}

	if err := RMSNorm(dst[:2], x, w, 1e-6); err == nil {
		t.Fatal("length mismatch accepted")
	}
	if err := RMSNorm(dst, nil, nil, 1e-6); err == nil {
		t.Fatal("empty input accepted")
	}
}

func TestRMSNormHeads(t *testing.T) {
	x := []float32{1, 2, 3, 4, 2, 0, 0, 2}
	w := []float32{1, 1, 1, 1}
	if err := RMSNormHeads(x, w, 2, 4, 1e-6); err != nil {
		t.Fatal(err)
	}
	mean0 := float32(1+4+9+16) / 4
	scale0 := float32(1) / float32(math.Sqrt(float64(mean0+1e-6)))
	if x[0] != 1*scale0 || x[3] != 4*scale0 {
		t.Fatalf("head 0 = %v", x[:4])
	}
	mean1 := float32(4+0+0+4) / 4
	scale1 := float32(1) / float32(math.Sqrt(float64(mean1+1e-6)))
	if x[4] != 2*scale1 || x[7] != 2*scale1 {
		t.Fatalf("head 1 = %v", x[4:])
	}
	if err := RMSNormHeads(x, w[:2], 2, 4, 1e-6); err == nil {
		t.Fatal("weight length mismatch accepted")
	}
	if err := RMSNormHeads(x[:4], w, 2, 4, 1e-6); err == nil {
		t.Fatal("input length mismatch accepted")
	}
}

func TestRoPETable(t *testing.T) {
	if _, err := NewRoPETable(0, 4, 4, 10000); err == nil {
		t.Fatal("zero positions accepted")
	}
	if _, err := NewRoPETable(8, 4, 3, 10000); err == nil {
		t.Fatal("odd ropeDim accepted")
	}
	if _, err := NewRoPETable(8, 4, 6, 10000); err == nil {
		t.Fatal("ropeDim > headDim accepted")
	}

	rt, err := NewRoPETable(16, 4, 4, 10000)
	if err != nil {
		t.Fatal(err)
	}

	x := []float32{1, 2, 3, 4}
	orig := append([]float32(nil), x...)
	if err := rt.Apply(x, 0, 1, 4); err != nil {
		t.Fatal(err)
	}
	for i := range x {
		if x[i] != orig[i] {
			t.Fatalf("position 0 changed value %d: %v != %v", i, x[i], orig[i])
		}
	}

	pos := 5
	freq0 := 1.0
	freq1 := math.Pow(10000, -2.0/4.0)
	s0, c0 := math.Sincos(float64(pos) * freq0)
	s1, c1 := math.Sincos(float64(pos) * freq1)
	if err := rt.Apply(x, pos, 1, 4); err != nil {
		t.Fatal(err)
	}
	want := []float32{
		orig[0]*float32(c0) - orig[2]*float32(s0),
		orig[1]*float32(c1) - orig[3]*float32(s1),
		orig[0]*float32(s0) + orig[2]*float32(c0),
		orig[1]*float32(s1) + orig[3]*float32(c1),
	}
	for i := range x {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 {
			t.Fatalf("index %d: got %v want %v", i, x[i], want[i])
		}
	}

	if err := rt.Apply(x, 16, 1, 4); err == nil {
		t.Fatal("position outside table accepted")
	}
	if err := rt.Apply(x, 1, 1, 8); err == nil {
		t.Fatal("headDim mismatch accepted")
	}
	if err := rt.Apply(x[:3], 1, 1, 4); err == nil {
		t.Fatal("length mismatch accepted")
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 1, 1, 1}
	Softmax(x)
	for i, v := range x {
		if v != 0.25 {
			t.Fatalf("uniform index %d = %v", i, v)
		}
	}

	one := []float32{-7}
	Softmax(one)
	if one[0] != 1 {
		t.Fatalf("single element = %v", one[0])
	}

	big := []float32{1000, 1001, 999}
	Softmax(big)
	var sum float32
	for _, v := range big {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite probability %v", v)
		}
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum = %v", sum)
	}
	if !(big[1] > big[0] && big[0] > big[2]) {
		t.Fatalf("ordering = %v", big)
	}

	empty := []float32{}
	Softmax(empty)
}

func TestSwiGLU(t *testing.T) {
	gate := []float32{0, 1, -1, 10}
	up := []float32{1, 2, 3, 0.5}
	dst := make([]float32, 4)
	if err := SwiGLU(dst, gate, up); err != nil {
		t.Fatal(err)
	}
	for i := range dst {
		want := SiLU(gate[i]) * up[i]
		if dst[i] != want {
			t.Fatalf("index %d: got %v want %v", i, dst[i], want)
		}
	}
	if got := SiLU(0); got != 0 {
		t.Fatalf("SiLU(0) = %v", got)
	}
	if err := SwiGLU(dst[:2], gate, up); err == nil {
		t.Fatal("length mismatch accepted")
	}
}

func TestAttentionHead(t *testing.T) {
	headDim := 4
	seqLen := 2
	query := []float32{1, 0, 0, 0}
	keys := make([]uint16, seqLen*headDim)
	values := make([]uint16, seqLen*headDim)
	F32SliceToF16(keys[:4], []float32{1, 0, 0, 0})
	F32SliceToF16(keys[4:], []float32{0, 1, 0, 0})
	F32SliceToF16(values[:4], []float32{10, 0, 0, 0})
	F32SliceToF16(values[4:], []float32{20, 0, 0, 0})

	scores := make([]float32, seqLen)
	dst := make([]float32, headDim)
	if err := AttentionHead(dst, query, keys, values, scores, headDim, seqLen, 1); err != nil {
		t.Fatal(err)
	}
	e1 := float32(math.Exp(1))
	w0 := e1 / (e1 + 1)
	w1 := float32(1) / (e1 + 1)
	want := w0*10 + w1*20
	if math.Abs(float64(dst[0]-want)) > 1e-4 {
		t.Fatalf("dst[0] = %v, want %v", dst[0], want)
	}
	for i := 1; i < headDim; i++ {
		if dst[i] != 0 {
			t.Fatalf("dst[%d] = %v", i, dst[i])
		}
	}

	if err := AttentionHead(dst, query, keys, values, scores, headDim, 1, 1); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 10 {
		t.Fatalf("single position dst[0] = %v", dst[0])
	}

	dst[0] = -999
	if err := AttentionHead(dst, query, keys, values, scores, headDim, 1, 1); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 10 {
		t.Fatalf("dst not overwritten: %v", dst[0])
	}

	if err := AttentionHead(dst, query, keys, values, scores, headDim, 0, 1); err == nil {
		t.Fatal("zero seqLen accepted")
	}
	if err := AttentionHead(dst[:2], query, keys, values, scores, headDim, 1, 1); err == nil {
		t.Fatal("short dst accepted")
	}
	if err := AttentionHead(dst, query, keys[:3], values, scores, headDim, 1, 1); err == nil {
		t.Fatal("short keys accepted")
	}
	if err := AttentionHead(dst, query, keys, values, scores[:0], headDim, 1, 1); err == nil {
		t.Fatal("short scores accepted")
	}
}

func TestArgmax(t *testing.T) {
	if got := Argmax([]float32{1, 3, 2}); got != 1 {
		t.Fatalf("got %d", got)
	}
	if got := Argmax([]float32{3, 3, 1}); got != 0 {
		t.Fatalf("tie should pick lowest index, got %d", got)
	}
	if got := Argmax([]float32{-5, -1, -3}); got != 1 {
		t.Fatalf("negative got %d", got)
	}
	if got := Argmax(nil); got != -1 {
		t.Fatalf("empty got %d", got)
	}
}
