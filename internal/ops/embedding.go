package ops

import (
	"encoding/binary"
	"errors"
	"math"
)

// EmbeddingRow decodes the embedding row for token into dst. dst must have
// length m.Cols and token must be within m.Rows.
func EmbeddingRow(dst []float32, m Matrix, token uint32) error {
	if err := m.Validate(); err != nil {
		return err
	}
	if uint64(token) >= uint64(m.Rows) {
		return errors.New("ops: embedding token out of range")
	}
	if len(dst) != m.Cols {
		return errors.New("ops: embedding destination length mismatch")
	}
	row := m.row(int(token))
	switch m.Kind {
	case KindF32:
		for i := range dst {
			dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(row[i*4:]))
		}
	case KindQ4K:
		for b := 0; b < m.Cols/Q4KBlockSize; b++ {
			DequantizeQ4KBlock(dst[b*Q4KBlockSize:(b+1)*Q4KBlockSize], row[b*Q4KTypeSize:(b+1)*Q4KTypeSize])
		}
	case KindQ6K:
		for b := 0; b < m.Cols/Q6KBlockSize; b++ {
			DequantizeQ6KBlock(dst[b*Q6KBlockSize:(b+1)*Q6KBlockSize], row[b*Q6KTypeSize:(b+1)*Q6KTypeSize])
		}
	}
	return nil
}
