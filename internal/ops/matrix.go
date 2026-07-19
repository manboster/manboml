package ops

import "fmt"

// GGML tensor kinds supported by the operators.
const (
	KindF32 uint32 = 0
	KindQ4K uint32 = 12
	KindQ6K uint32 = 14
)

// Matrix is a packed weight matrix in GGUF layout: Cols is the contiguous
// input dimension and Rows is the output dimension. Data is a read-only view
// into the model backing.
type Matrix struct {
	Data []byte
	Rows int
	Cols int
	Kind uint32
}

// RowBytes returns the encoded size of one row. It returns 0 for unsupported
// kinds or invalid dimensions.
func (m Matrix) RowBytes() int {
	if m.Cols <= 0 {
		return 0
	}
	switch m.Kind {
	case KindF32:
		return m.Cols * 4
	case KindQ4K:
		if m.Cols%Q4KBlockSize != 0 {
			return 0
		}
		return m.Cols / Q4KBlockSize * Q4KTypeSize
	case KindQ6K:
		if m.Cols%Q6KBlockSize != 0 {
			return 0
		}
		return m.Cols / Q6KBlockSize * Q6KTypeSize
	}
	return 0
}

// Validate checks the matrix dimensions and backing size.
func (m Matrix) Validate() error {
	if m.Rows <= 0 || m.Cols <= 0 {
		return fmt.Errorf("ops: invalid matrix shape %dx%d", m.Rows, m.Cols)
	}
	rowBytes := m.RowBytes()
	if rowBytes == 0 {
		return fmt.Errorf("ops: unsupported matrix kind %d with %d columns", m.Kind, m.Cols)
	}
	if m.Rows > int(^uint(0)>>1)/rowBytes {
		return fmt.Errorf("ops: matrix size overflows")
	}
	if len(m.Data) != m.Rows*rowBytes {
		return fmt.Errorf("ops: matrix data is %d bytes, want %d", len(m.Data), m.Rows*rowBytes)
	}
	return nil
}

func (m Matrix) row(i int) []byte {
	rowBytes := m.RowBytes()
	return m.Data[i*rowBytes : (i+1)*rowBytes]
}
