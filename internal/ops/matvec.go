package ops

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
)

// MatVecF32 computes dst = m * x using float32 activations. It is the scalar
// reference path for correctness testing and debugging. ctx cancellation is
// checked at bounded row intervals.
func MatVecF32(ctx context.Context, ex *Executor, dst []float32, m Matrix, x []float32) error {
	if ex == nil {
		ex = NewExecutor(1)
	}
	if err := m.Validate(); err != nil {
		return err
	}
	if len(dst) != m.Rows || len(x) != m.Cols {
		return errors.New("ops: MatVecF32 length mismatch")
	}
	blocks := m.Cols / Q4KBlockSize
	return ex.run(ctx, m.Rows, func(ctx context.Context, start, end int) error {
		var scratch [Q4KBlockSize]float32
		for row := start; row < end; row++ {
			if (row-start)%64 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			dst[row] = dotRowF32(m.row(row), m.Kind, x, blocks, &scratch)
		}
		return nil
	})
}

// MatVecQ8K computes dst = m * x where x is a Q8_K-quantized activation. It is
// the production path for quantized K-quant weights. Only KindQ4K and KindQ6K
// are supported.
func MatVecQ8K(ctx context.Context, ex *Executor, dst []float32, m Matrix, x *Q8K) error {
	if ex == nil {
		ex = NewExecutor(1)
	}
	if m.Kind != KindQ4K && m.Kind != KindQ6K {
		return errors.New("ops: MatVecQ8K requires Q4_K or Q6_K weights")
	}
	if err := m.Validate(); err != nil {
		return err
	}
	if len(dst) != m.Rows || x.Blocks()*Q8KBlockSize != m.Cols {
		return errors.New("ops: MatVecQ8K length mismatch")
	}
	blocks := m.Cols / Q4KBlockSize
	return ex.run(ctx, m.Rows, func(ctx context.Context, start, end int) error {
		for row := start; row < end; row++ {
			if (row-start)%64 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			dst[row] = dotRowQ8K(m.row(row), m.Kind, x, blocks)
		}
		return nil
	})
}

func dotRowF32(row []byte, kind uint32, x []float32, blocks int, scratch *[Q4KBlockSize]float32) float32 {
	var sum float32
	switch kind {
	case KindF32:
		for i := 0; i < len(x); i++ {
			sum += math.Float32frombits(binary.LittleEndian.Uint32(row[i*4:])) * x[i]
		}
	case KindQ4K:
		for b := 0; b < blocks; b++ {
			DequantizeQ4KBlock(scratch[:], row[b*Q4KTypeSize:(b+1)*Q4KTypeSize])
			blk := x[b*Q4KBlockSize : (b+1)*Q4KBlockSize]
			var s float32
			for i := 0; i < Q4KBlockSize; i++ {
				s += scratch[i] * blk[i]
			}
			sum += s
		}
	case KindQ6K:
		for b := 0; b < blocks; b++ {
			DequantizeQ6KBlock(scratch[:], row[b*Q6KTypeSize:(b+1)*Q6KTypeSize])
			blk := x[b*Q6KBlockSize : (b+1)*Q6KBlockSize]
			var s float32
			for i := 0; i < Q6KBlockSize; i++ {
				s += scratch[i] * blk[i]
			}
			sum += s
		}
	}
	return sum
}

func dotRowQ8K(row []byte, kind uint32, x *Q8K, blocks int) float32 {
	var sum float32
	switch kind {
	case KindQ4K:
		for b := 0; b < blocks; b++ {
			sum += dotQ4KQ8KBlock(
				row[b*Q4KTypeSize:(b+1)*Q4KTypeSize],
				x.Scales[b],
				x.Qs[b*Q8KBlockSize:(b+1)*Q8KBlockSize],
				x.Bsums[b*Q8KGroupCount:(b+1)*Q8KGroupCount],
			)
		}
	case KindQ6K:
		for b := 0; b < blocks; b++ {
			sum += dotQ6KQ8KBlock(
				row[b*Q6KTypeSize:(b+1)*Q6KTypeSize],
				x.Scales[b],
				x.Qs[b*Q8KBlockSize:(b+1)*Q8KBlockSize],
			)
		}
	}
	return sum
}
