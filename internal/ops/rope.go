package ops

import (
	"errors"
	"fmt"
	"math"
)

// RoPETable holds precomputed cosine and sine values for NeoX-style rotary
// position embeddings. It is shared by all sessions of one model.
type RoPETable struct {
	cos       []float32
	sin       []float32
	positions int
	freqDim   int
	headDim   int
}

// NewRoPETable precomputes rotation values for positions 0..positions-1.
// ropeDim must be even and no larger than headDim.
func NewRoPETable(positions, headDim, ropeDim int, theta float64) (*RoPETable, error) {
	if positions <= 0 || headDim <= 0 || ropeDim <= 0 {
		return nil, errors.New("ops: RoPE dimensions must be positive")
	}
	if ropeDim%2 != 0 || ropeDim > headDim {
		return nil, fmt.Errorf("ops: invalid RoPE dimension %d for head dimension %d", ropeDim, headDim)
	}
	if theta <= 0 {
		return nil, errors.New("ops: RoPE base must be positive")
	}
	freqDim := ropeDim / 2
	if positions > int(^uint(0)>>1)/freqDim {
		return nil, errors.New("ops: RoPE table size overflows")
	}
	t := &RoPETable{
		cos:       make([]float32, positions*freqDim),
		sin:       make([]float32, positions*freqDim),
		positions: positions,
		freqDim:   freqDim,
		headDim:   headDim,
	}
	for pos := 0; pos < positions; pos++ {
		for i := 0; i < freqDim; i++ {
			angle := float64(pos) * math.Pow(theta, -2*float64(i)/float64(ropeDim))
			sn, cs := math.Sincos(angle)
			t.cos[pos*freqDim+i] = float32(cs)
			t.sin[pos*freqDim+i] = float32(sn)
		}
	}
	return t, nil
}

// Positions returns the number of precomputed positions.
func (t *RoPETable) Positions() int { return t.positions }

// Apply rotates x in place using the NeoX layout: for each head, element i is
// paired with element i+ropeDim/2 for i < ropeDim/2. x must have length
// heads*headDim.
func (t *RoPETable) Apply(x []float32, position, heads, headDim int) error {
	if heads <= 0 || len(x) != heads*headDim {
		return errors.New("ops: RoPE input length mismatch")
	}
	if headDim != t.headDim {
		return fmt.Errorf("ops: RoPE head dimension %d, want %d", headDim, t.headDim)
	}
	if position < 0 || position >= t.positions {
		return fmt.Errorf("ops: RoPE position %d outside table", position)
	}
	freqDim := t.freqDim
	cosRow := t.cos[position*freqDim : (position+1)*freqDim]
	sinRow := t.sin[position*freqDim : (position+1)*freqDim]
	for h := 0; h < heads; h++ {
		head := x[h*headDim : (h+1)*headDim]
		for i := 0; i < freqDim; i++ {
			x0 := head[i]
			x1 := head[i+freqDim]
			c := cosRow[i]
			s := sinRow[i]
			head[i] = x0*c - x1*s
			head[i+freqDim] = x0*s + x1*c
		}
	}
	return nil
}
