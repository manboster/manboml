package ops

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
)

func buildMatrix(t *testing.T, kind uint32, rows, cols int, seed uint64) Matrix {
	t.Helper()
	var rowBytes int
	switch kind {
	case KindF32:
		rowBytes = cols * 4
	case KindQ4K:
		rowBytes = cols / Q4KBlockSize * Q4KTypeSize
	case KindQ6K:
		rowBytes = cols / Q6KBlockSize * Q6KTypeSize
	default:
		t.Fatalf("unknown kind %d", kind)
	}
	data := randomBytes(seed, rows*rowBytes)
	switch kind {
	case KindF32:
		for i := 0; i < rows*cols; i++ {
			v := float32(int(lcg(&seed)%4096)-2048) / 512
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
		}
	case KindQ4K:
		for off := 0; off+Q4KTypeSize <= len(data); off += Q4KTypeSize {
			saneF16(data, off+0, uint16(0x3400+lcg(&seed)%0x0400))
			saneF16(data, off+2, uint16(0x2C00+lcg(&seed)%0x0400))
		}
	case KindQ6K:
		for off := 0; off+Q6KTypeSize <= len(data); off += Q6KTypeSize {
			saneF16(data, off+208, uint16(0x3400+lcg(&seed)%0x0400))
		}
	}
	return Matrix{Data: data, Rows: rows, Cols: cols, Kind: kind}
}

func dequantRow(t *testing.T, m Matrix, row int) []float32 {
	t.Helper()
	out := make([]float32, m.Cols)
	raw := m.row(row)
	switch m.Kind {
	case KindF32:
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case KindQ4K:
		for b := 0; b < m.Cols/Q4KBlockSize; b++ {
			DequantizeQ4KBlock(out[b*Q4KBlockSize:(b+1)*Q4KBlockSize], raw[b*Q4KTypeSize:(b+1)*Q4KTypeSize])
		}
	case KindQ6K:
		for b := 0; b < m.Cols/Q6KBlockSize; b++ {
			DequantizeQ6KBlock(out[b*Q6KBlockSize:(b+1)*Q6KBlockSize], raw[b*Q6KTypeSize:(b+1)*Q6KTypeSize])
		}
	}
	return out
}

func refMatVec(m Matrix, x []float32) []float64 {
	out := make([]float64, m.Rows)
	for r := 0; r < m.Rows; r++ {
		row := make([]float32, m.Cols)
		raw := m.row(r)
		switch m.Kind {
		case KindF32:
			for i := range row {
				row[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
		case KindQ4K:
			for b := 0; b < m.Cols/Q4KBlockSize; b++ {
				DequantizeQ4KBlock(row[b*Q4KBlockSize:(b+1)*Q4KBlockSize], raw[b*Q4KTypeSize:(b+1)*Q4KTypeSize])
			}
		case KindQ6K:
			for b := 0; b < m.Cols/Q6KBlockSize; b++ {
				DequantizeQ6KBlock(row[b*Q6KBlockSize:(b+1)*Q6KBlockSize], raw[b*Q6KTypeSize:(b+1)*Q6KTypeSize])
			}
		}
		var sum float64
		for i := 0; i < m.Cols; i++ {
			sum += float64(row[i]) * float64(x[i])
		}
		out[r] = sum
	}
	return out
}

func TestMatVecF32AllKinds(t *testing.T) {
	for _, kind := range []uint32{KindF32, KindQ4K, KindQ6K} {
		m := buildMatrix(t, kind, 5, 512, 1000+uint64(kind))
		x := make([]float32, 512)
		seed := uint64(77)
		for i := range x {
			x[i] = float32(int(lcg(&seed)%2048)-1024) / 256
		}
		want := refMatVec(m, x)
		dst := make([]float32, m.Rows)
		if err := MatVecF32(context.Background(), nil, dst, m, x); err != nil {
			t.Fatalf("kind %d: %v", kind, err)
		}
		for r := range dst {
			if !almostEqual(float64(dst[r]), want[r], 1e-4) {
				t.Fatalf("kind %d row %d: got %v want %v", kind, r, dst[r], want[r])
			}
		}
	}
}

func TestMatVecF32WorkerDeterminism(t *testing.T) {
	m := buildMatrix(t, KindQ4K, 33, 1024, 555)
	x := make([]float32, 1024)
	seed := uint64(9)
	for i := range x {
		x[i] = float32(int(lcg(&seed)%2048)-1024) / 512
	}
	base := make([]float32, m.Rows)
	if err := MatVecF32(context.Background(), NewExecutor(1), base, m, x); err != nil {
		t.Fatal(err)
	}
	for _, workers := range []int{2, 3, 8, 64} {
		got := make([]float32, m.Rows)
		if err := MatVecF32(context.Background(), NewExecutor(workers), got, m, x); err != nil {
			t.Fatal(err)
		}
		for r := range got {
			if math.Float32bits(got[r]) != math.Float32bits(base[r]) {
				t.Fatalf("workers %d row %d: %v != %v", workers, r, got[r], base[r])
			}
		}
	}
}

func TestMatVecQ8KAgainstF32(t *testing.T) {
	for _, kind := range []uint32{KindQ4K, KindQ6K} {
		m := buildMatrix(t, kind, 4, 512, 4242+uint64(kind))
		src := make([]float32, 512)
		seed := uint64(31)
		for i := range src {
			src[i] = float32(int(lcg(&seed)%4096)-2048) / 1024
		}
		act := NewQ8K(512)
		QuantizeQ8K(&act, src)

		got := make([]float32, m.Rows)
		if err := MatVecQ8K(context.Background(), NewExecutor(2), got, m, &act); err != nil {
			t.Fatalf("kind %d: %v", kind, err)
		}

		qa := make([]float32, 512)
		for b := 0; b < act.Blocks(); b++ {
			for i := 0; i < Q8KBlockSize; i++ {
				qa[b*Q8KBlockSize+i] = act.Scales[b] * float32(act.Qs[b*Q8KBlockSize+i])
			}
		}
		want := refMatVec(m, qa)
		for r := range got {
			if !almostEqual(float64(got[r]), want[r], 1e-3) {
				t.Fatalf("kind %d row %d: got %v want %v", kind, r, got[r], want[r])
			}
		}
	}
}

func TestMatVecErrors(t *testing.T) {
	m := buildMatrix(t, KindQ4K, 2, 512, 1)
	x := make([]float32, 512)
	if err := MatVecF32(context.Background(), nil, make([]float32, 1), m, x); err == nil {
		t.Fatal("short dst accepted")
	}
	if err := MatVecF32(context.Background(), nil, make([]float32, 2), m, x[:256]); err == nil {
		t.Fatal("short x accepted")
	}
	bad := Matrix{Data: m.Data[:len(m.Data)-1], Rows: 2, Cols: 512, Kind: KindQ4K}
	if err := MatVecF32(context.Background(), nil, make([]float32, 2), bad, x); err == nil {
		t.Fatal("short data accepted")
	}
	odd := Matrix{Data: make([]byte, 100), Rows: 1, Cols: 100, Kind: KindQ4K}
	if err := MatVecF32(context.Background(), nil, make([]float32, 1), odd, make([]float32, 100)); err == nil {
		t.Fatal("non-multiple-of-256 columns accepted")
	}
	f32m := buildMatrix(t, KindF32, 2, 512, 7)
	act := NewQ8K(512)
	QuantizeQ8K(&act, x)
	if err := MatVecQ8K(context.Background(), nil, make([]float32, 2), f32m, &act); err == nil {
		t.Fatal("F32 weights accepted by MatVecQ8K")
	}
	unknown := Matrix{Data: make([]byte, 512), Rows: 1, Cols: 512, Kind: 99}
	if err := MatVecF32(context.Background(), nil, make([]float32, 1), unknown, x); err == nil {
		t.Fatal("unknown kind accepted")
	}
}

func TestMatVecCancellation(t *testing.T) {
	m := buildMatrix(t, KindQ4K, 256, 512, 3)
	x := make([]float32, 512)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := MatVecF32(ctx, NewExecutor(4), make([]float32, 256), m, x); err == nil {
		t.Fatal("canceled context accepted")
	}
}

func TestEmbeddingRow(t *testing.T) {
	for _, kind := range []uint32{KindF32, KindQ4K, KindQ6K} {
		m := buildMatrix(t, kind, 8, 512, 900+uint64(kind))
		dst := make([]float32, 512)
		if err := EmbeddingRow(dst, m, 3); err != nil {
			t.Fatalf("kind %d: %v", kind, err)
		}
		want := dequantRow(t, m, 3)
		for i := range dst {
			if dst[i] != want[i] {
				t.Fatalf("kind %d index %d: got %v want %v", kind, i, dst[i], want[i])
			}
		}
		if err := EmbeddingRow(dst, m, 8); err == nil {
			t.Fatalf("kind %d: out-of-range token accepted", kind)
		}
		if err := EmbeddingRow(dst[:100], m, 0); err == nil {
			t.Fatalf("kind %d: short dst accepted", kind)
		}
	}
}

func TestVectorOps(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dst := make([]float32, 3)
	if err := Add(dst, a, b); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 5 || dst[1] != 7 || dst[2] != 9 {
		t.Fatalf("Add = %v", dst)
	}
	if err := AddInplace(dst, a); err != nil {
		t.Fatal(err)
	}
	if dst[0] != 6 || dst[1] != 9 || dst[2] != 12 {
		t.Fatalf("AddInplace = %v", dst)
	}
	if got := Dot(a, b); got != 32 {
		t.Fatalf("Dot = %v", got)
	}
	if err := Add(dst[:2], a, b); err == nil {
		t.Fatal("mismatched Add accepted")
	}
	if err := AddInplace(dst, a[:2]); err == nil {
		t.Fatal("mismatched AddInplace accepted")
	}
}
