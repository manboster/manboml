package ops

import (
	"errors"
	"math"
)

// RMSNorm computes dst = x * w / sqrt(mean(x^2) + eps), with the float32
// accumulation order used by ggml. x, w and dst must have equal length.
func RMSNorm(dst, x, w []float32, eps float32) error {
	if len(dst) != len(x) || len(x) != len(w) {
		return errors.New("ops: RMSNorm length mismatch")
	}
	if len(x) == 0 {
		return errors.New("ops: RMSNorm empty input")
	}
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	mean := sum / float32(len(x))
	scale := float32(1) / float32(math.Sqrt(float64(mean+eps)))
	for i := range dst {
		dst[i] = x[i] * scale * w[i]
	}
	return nil
}

// RMSNormHeads applies RMSNorm independently to each head of x in place,
// using the shared per-head weight w of length headDim. It implements Qwen3's
// per-head Q/K normalization.
func RMSNormHeads(x, w []float32, heads, headDim int, eps float32) error {
	if heads <= 0 || headDim <= 0 {
		return errors.New("ops: RMSNormHeads invalid shape")
	}
	if len(w) != headDim || len(x) != heads*headDim {
		return errors.New("ops: RMSNormHeads length mismatch")
	}
	for h := 0; h < heads; h++ {
		head := x[h*headDim : (h+1)*headDim]
		var sum float32
		for _, v := range head {
			sum += v * v
		}
		mean := sum / float32(headDim)
		scale := float32(1) / float32(math.Sqrt(float64(mean+eps)))
		for i := range head {
			head[i] = head[i] * scale * w[i]
		}
	}
	return nil
}
