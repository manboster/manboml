package ops

import (
	"errors"
	"math"
)

// SiLU returns x * sigmoid(x).
func SiLU(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

// SwiGLU computes dst = SiLU(gate) * up elementwise. dst may alias gate or
// up. All slices must have equal length.
func SwiGLU(dst, gate, up []float32) error {
	if len(dst) != len(gate) || len(gate) != len(up) {
		return errors.New("ops: SwiGLU length mismatch")
	}
	for i := range dst {
		dst[i] = SiLU(gate[i]) * up[i]
	}
	return nil
}
