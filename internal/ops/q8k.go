package ops

import "math"

// Q8_K constants: 256 values per super-block; values are grouped in 16 runs
// of 16 for block sums.
const (
	Q8KBlockSize  = 256
	Q8KGroupSize  = 16
	Q8KGroupCount = Q8KBlockSize / Q8KGroupSize
)

// Q8K is a quantized activation vector in the Q8_K layout. All slices are
// indexed by super-block.
type Q8K struct {
	Scales []float32
	Qs     []int8
	Bsums  []int16
}

// NewQ8K allocates Q8_K storage for n activation values. n must be a multiple
// of Q8KBlockSize.
func NewQ8K(n int) Q8K {
	if n <= 0 || n%Q8KBlockSize != 0 {
		panic("ops: Q8K size must be a positive multiple of 256")
	}
	blocks := n / Q8KBlockSize
	return Q8K{
		Scales: make([]float32, blocks),
		Qs:     make([]int8, n),
		Bsums:  make([]int16, blocks*Q8KGroupCount),
	}
}

// Blocks returns the number of super-blocks in q.
func (q Q8K) Blocks() int { return len(q.Scales) }

// QuantizeQ8K quantizes src into dst using the Q8_K scheme, mirroring ggml's
// quantize_row_q8_K_ref: per 256-value block, d = amax/127, qs = round(x/d),
// plus per-16-value int16 block sums. src length must match dst capacity.
func QuantizeQ8K(dst *Q8K, src []float32) {
	blocks := dst.Blocks()
	if len(src) != blocks*Q8KBlockSize {
		panic("ops: QuantizeQ8K length mismatch")
	}
	for i := 0; i < blocks; i++ {
		x := src[i*Q8KBlockSize : (i+1)*Q8KBlockSize]
		qs := dst.Qs[i*Q8KBlockSize : (i+1)*Q8KBlockSize]
		bs := dst.Bsums[i*Q8KGroupCount : (i+1)*Q8KGroupCount]

		var amax float32
		for _, v := range x {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		if amax == 0 {
			dst.Scales[i] = 0
			for j := range qs {
				qs[j] = 0
			}
			for j := range bs {
				bs[j] = 0
			}
			continue
		}

		d := amax / 127
		id := 1 / d
		dst.Scales[i] = d
		for j, v := range x {
			q := int32(math.Round(float64(v * id)))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			qs[j] = int8(q)
		}
		for g := 0; g < Q8KGroupCount; g++ {
			var sum int32
			run := qs[g*Q8KGroupSize : (g+1)*Q8KGroupSize]
			for _, v := range run {
				sum += int32(v)
			}
			bs[g] = int16(sum)
		}
	}
}
