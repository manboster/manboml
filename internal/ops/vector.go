package ops

import "errors"

// Add computes dst = a + b elementwise. All slices must have equal length.
func Add(dst, a, b []float32) error {
	if len(dst) != len(a) || len(a) != len(b) {
		return errors.New("ops: Add length mismatch")
	}
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
	return nil
}

// AddInplace computes dst += src elementwise.
func AddInplace(dst, src []float32) error {
	if len(dst) != len(src) {
		return errors.New("ops: AddInplace length mismatch")
	}
	for i := range dst {
		dst[i] += src[i]
	}
	return nil
}

// Dot returns the float32 dot product of a and b in increasing index order.
func Dot(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float32
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
