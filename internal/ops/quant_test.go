package ops

import (
	"encoding/binary"
	"math"
	"testing"
)

func lcg(seed *uint64) uint32 {
	*seed = *seed*6364136223846793005 + 1442695040888963407
	return uint32(*seed >> 32)
}

func randomBytes(seed uint64, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(lcg(&seed))
	}
	return b
}

func TestF16ToF32KnownValues(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x3C00, 1},
		{0xC000, -2},
		{0x0555, float32(float64(0x555) * math.Ldexp(1, -24))},
		{0x0001, float32(math.Ldexp(1, -24))},
		{0x7BFF, 65504},
		{0x7C00, float32(math.Inf(1))},
		{0xFC00, float32(math.Inf(-1))},
	}
	for _, c := range cases {
		if got := F16ToF32(c.bits); got != c.want {
			t.Errorf("F16ToF32(%#04x) = %v, want %v", c.bits, got, c.want)
		}
	}
	if got := F16ToF32(0x7E00); !math.IsNaN(float64(got)) {
		t.Errorf("F16ToF32(NaN) = %v", got)
	}
	if got := F16ToF32(0x0002); got != float32(2*math.Ldexp(1, -24)) {
		t.Errorf("subnormal 0x0002 = %v", got)
	}
}

func TestF32ToF16KnownValues(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3C00},
		{-2, 0xC000},
		{65504, 0x7BFF},
		{65519, 0x7BFF},
		{65520, 0x7C00},
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
		{float32(math.Ldexp(1, -24)), 0x0001},
		{float32(math.Ldexp(1, -25)), 0x0000},
		{float32(3 * math.Ldexp(1, -25)), 0x0002},
		{1 + float32(math.Ldexp(1, -11)), 0x3C00},
		{1 + 3*float32(math.Ldexp(1, -11)), 0x3C02},
	}
	for _, c := range cases {
		if got := F32ToF16(c.in); got != c.want {
			t.Errorf("F32ToF16(%v) = %#04x, want %#04x", c.in, got, c.want)
		}
	}
	if got := F32ToF16(float32(math.NaN())); got != 0x7E00 {
		t.Errorf("F32ToF16(NaN) = %#04x", got)
	}
}

func TestF16RoundTrip(t *testing.T) {
	for h := uint32(0); h < 65536; h++ {
		bits := uint16(h)
		f := F16ToF32(bits)
		if math.IsNaN(float64(f)) {
			continue
		}
		if got := F32ToF16(f); got != bits {
			t.Fatalf("round trip %#04x -> %v -> %#04x", bits, f, got)
		}
	}
}

func refQ4KDequant(out []float32, b []byte) {
	d := F16ToF32(binary.LittleEndian.Uint16(b))
	dmin := F16ToF32(binary.LittleEndian.Uint16(b[2:]))
	scales := b[4:16]
	qs := b[16:144]
	yi := 0
	for j := 0; j < 4; j++ {
		sc1, m1 := q4kScaleMin(2*j, scales)
		sc2, m2 := q4kScaleMin(2*j+1, scales)
		for l := 0; l < 32; l++ {
			out[yi] = d*float32(sc1)*float32(qs[j*32+l]&0x0F) - dmin*float32(m1)
			yi++
		}
		for l := 0; l < 32; l++ {
			out[yi] = d*float32(sc2)*float32(qs[j*32+l]>>4) - dmin*float32(m2)
			yi++
		}
	}
}

func TestDequantizeQ4KBlock(t *testing.T) {
	b := randomBytes(42, Q4KTypeSize)
	got := make([]float32, Q4KBlockSize)
	want := make([]float32, Q4KBlockSize)
	DequantizeQ4KBlock(got, b)
	refQ4KDequant(want, b)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestDequantizeQ4KBlockZero(t *testing.T) {
	b := make([]byte, Q4KTypeSize)
	out := make([]float32, Q4KBlockSize)
	DequantizeQ4KBlock(out, b)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("zero block index %d = %v", i, v)
		}
	}
}

func refQ6KDequant(out []float32, b []byte) {
	ql, qh, sc := b[:128], b[128:192], b[192:208]
	d := F16ToF32(binary.LittleEndian.Uint16(b[208:]))
	for chunk := 0; chunk < 2; chunk++ {
		n0 := chunk * 128
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((ql[chunk*64+l]&0x0F)|(((qh[chunk*32+l]>>0)&3)<<4)) - 32
			q2 := int8((ql[chunk*64+l+32]&0x0F)|(((qh[chunk*32+l]>>2)&3)<<4)) - 32
			q3 := int8((ql[chunk*64+l]>>4)|(((qh[chunk*32+l]>>4)&3)<<4)) - 32
			q4 := int8((ql[chunk*64+l+32]>>4)|(((qh[chunk*32+l]>>6)&3)<<4)) - 32
			out[n0+l] = d * float32(int8(sc[chunk*8+is+0])) * float32(q1)
			out[n0+l+32] = d * float32(int8(sc[chunk*8+is+2])) * float32(q2)
			out[n0+l+64] = d * float32(int8(sc[chunk*8+is+4])) * float32(q3)
			out[n0+l+96] = d * float32(int8(sc[chunk*8+is+6])) * float32(q4)
		}
	}
}

func TestDequantizeQ6KBlock(t *testing.T) {
	b := randomBytes(43, Q6KTypeSize)
	got := make([]float32, Q6KBlockSize)
	want := make([]float32, Q6KBlockSize)
	DequantizeQ6KBlock(got, b)
	refQ6KDequant(want, b)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("index %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestQuantizeQ8KZero(t *testing.T) {
	dst := NewQ8K(Q8KBlockSize)
	QuantizeQ8K(&dst, make([]float32, Q8KBlockSize))
	if dst.Scales[0] != 0 {
		t.Fatalf("scale = %v", dst.Scales[0])
	}
	for i, v := range dst.Qs {
		if v != 0 {
			t.Fatalf("qs[%d] = %d", i, v)
		}
	}
	for i, v := range dst.Bsums {
		if v != 0 {
			t.Fatalf("bsums[%d] = %d", i, v)
		}
	}
}

func TestQuantizeQ8KKnown(t *testing.T) {
	src := make([]float32, Q8KBlockSize)
	for i := range src {
		src[i] = 1.27
	}
	src[7] = -1.27
	src[100] = 0
	dst := NewQ8K(Q8KBlockSize)
	QuantizeQ8K(&dst, src)
	if dst.Scales[0] != 1.27/127 {
		t.Fatalf("scale = %v, want %v", dst.Scales[0], 1.27/127)
	}
	for i, v := range dst.Qs {
		want := int8(127)
		if i == 7 {
			want = -127
		}
		if i == 100 {
			want = 0
		}
		if v != want {
			t.Fatalf("qs[%d] = %d, want %d", i, v, want)
		}
	}
	var sum0 int32
	for _, v := range dst.Qs[:16] {
		sum0 += int32(v)
	}
	if got := dst.Bsums[0]; int32(got) != sum0 {
		t.Fatalf("bsums[0] = %d, want %d", got, sum0)
	}
	var total int32
	for g := 0; g < Q8KGroupCount; g++ {
		total += int32(dst.Bsums[g])
	}
	var direct int32
	for _, v := range dst.Qs {
		direct += int32(v)
	}
	if total != direct {
		t.Fatalf("bsum total %d != qs sum %d", total, direct)
	}
}

func TestQuantizeQ8KClamps(t *testing.T) {
	src := make([]float32, Q8KBlockSize)
	src[0] = 100
	src[1] = -100
	dst := NewQ8K(Q8KBlockSize)
	QuantizeQ8K(&dst, src)
	if dst.Qs[0] != 127 || dst.Qs[1] != -127 {
		t.Fatalf("clamp got %d, %d", dst.Qs[0], dst.Qs[1])
	}
	if dst.Scales[0] != 100.0/127 {
		t.Fatalf("scale = %v", dst.Scales[0])
	}
}

func makeActivation(seed uint64) []float32 {
	src := make([]float32, Q8KBlockSize)
	for i := range src {
		src[i] = (float32(int(lcg(&seed)%4096)-2048) / 512)
	}
	return src
}

func almostEqual(a, b, tol float64) bool {
	d := math.Abs(float64(a) - float64(b))
	m := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return d <= tol*math.Max(1, m)
}

func saneF16(b []byte, off int, bits uint16) {
	binary.LittleEndian.PutUint16(b[off:off+2], bits)
}

func TestDotQ4KQ8KAgainstDequant(t *testing.T) {
	for seed := uint64(1); seed <= 5; seed++ {
		b := randomBytes(seed*7919, Q4KTypeSize)
		saneF16(b, 0, 0x3C00)
		saneF16(b, 2, 0x3800)
		src := makeActivation(seed * 104729)
		act := NewQ8K(Q8KBlockSize)
		QuantizeQ8K(&act, src)

		weights := make([]float32, Q4KBlockSize)
		DequantizeQ4KBlock(weights, b)

		var ref float64
		for i := 0; i < Q8KBlockSize; i++ {
			ref += float64(weights[i]) * float64(act.Scales[0]*float32(act.Qs[i]))
		}
		got := dotQ4KQ8KBlock(b, act.Scales[0], act.Qs, act.Bsums)
		if !almostEqual(float64(got), ref, 1e-4) {
			t.Fatalf("seed %d: got %v, ref %v", seed, got, ref)
		}
	}
}

func TestDotQ6KQ8KAgainstDequant(t *testing.T) {
	for seed := uint64(11); seed <= 15; seed++ {
		b := randomBytes(seed*7919, Q6KTypeSize)
		saneF16(b, 208, 0x3C00)
		src := makeActivation(seed * 104729)
		act := NewQ8K(Q8KBlockSize)
		QuantizeQ8K(&act, src)

		weights := make([]float32, Q6KBlockSize)
		DequantizeQ6KBlock(weights, b)

		var ref float64
		for i := 0; i < Q6KBlockSize; i++ {
			ref += float64(weights[i]) * float64(act.Scales[0]*float32(act.Qs[i]))
		}
		got := dotQ6KQ8KBlock(b, act.Scales[0], act.Qs)
		if !almostEqual(float64(got), ref, 1e-4) {
			t.Fatalf("seed %d: got %v, ref %v", seed, got, ref)
		}
	}
}
