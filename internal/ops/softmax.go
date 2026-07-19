package ops

import "math"

// Softmax normalizes x in place using the max-subtraction stable form.
// An empty slice is a no-op.
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}
	var sum float32
	for i, v := range x {
		e := float32(math.Exp(float64(v - max)))
		x[i] = e
		sum += e
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
}
