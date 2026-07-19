package ops

import "encoding/binary"

// Q4_K constants: 256 values per super-block, 144 bytes per super-block.
const (
	Q4KBlockSize = 256
	Q4KTypeSize  = 144
)

// DequantizeQ4KBlock decodes one Q4_K super-block from b (144 bytes) into
// out (256 values): y = d*scale*q - dmin*min, mirroring ggml's
// dequantize_row_q4_K.
func DequantizeQ4KBlock(out []float32, b []byte) {
	_ = out[255]
	_ = b[143]
	d := F16ToF32(binary.LittleEndian.Uint16(b))
	dmin := F16ToF32(binary.LittleEndian.Uint16(b[2:]))
	scales := b[4:16]
	qs := b[16:144]

	yi := 0
	for j := 0; j < 4; j++ {
		is := 2 * j
		sc1, m1 := q4kScaleMin(is, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)
		d1, off1 := d*float32(sc1), dmin*float32(m1)
		d2, off2 := d*float32(sc2), dmin*float32(m2)
		q := qs[j*32 : j*32+32]
		for l := 0; l < 32; l++ {
			out[yi] = d1*float32(q[l]&0x0F) - off1
			yi++
		}
		for l := 0; l < 32; l++ {
			out[yi] = d2*float32(q[l]>>4) - off2
			yi++
		}
	}
}

// q4kScaleMin unpacks the j-th 6-bit scale and min from the packed 12-byte
// scales array of a Q4_K/Q5_K super-block (ggml's get_scale_min_k4).
func q4kScaleMin(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	scale = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
	min = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
	return scale, min
}

// dotQ4KQ8KBlock computes the dot product of one Q4_K weight super-block and
// one Q8_K activation super-block, matching ggml_vec_dot_q4_K_q8_K's integer
// accumulation order.
func dotQ4KQ8KBlock(b []byte, yd float32, q8 []int8, bsums []int16) float32 {
	_ = b[143]
	_ = q8[255]
	_ = bsums[15]
	d := F16ToF32(binary.LittleEndian.Uint16(b)) * yd
	dmin := F16ToF32(binary.LittleEndian.Uint16(b[2:])) * yd
	scales := b[4:16]
	qs := b[16:144]

	var isum, imin int32
	for j := 0; j < 4; j++ {
		is := 2 * j
		sc1, m1 := q4kScaleMin(is, scales)
		sc2, m2 := q4kScaleMin(is+1, scales)

		q := qs[j*32 : j*32+32]
		a := q8[j*64 : j*64+64]

		var dot1, dot2 int32
		for l := 0; l < 32; l++ {
			v := q[l]
			dot1 += int32(v&0x0F) * int32(a[l])
			dot2 += int32(v>>4) * int32(a[l+32])
		}
		isum += int32(sc1)*dot1 + int32(sc2)*dot2
		imin += int32(m1)*(int32(bsums[4*j])+int32(bsums[4*j+1])) +
			int32(m2)*(int32(bsums[4*j+2])+int32(bsums[4*j+3]))
	}
	return d*float32(isum) - dmin*float32(imin)
}
