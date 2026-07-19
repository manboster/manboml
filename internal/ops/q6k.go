package ops

import "encoding/binary"

// Q6_K constants: 256 values per super-block, 210 bytes per super-block.
const (
	Q6KBlockSize = 256
	Q6KTypeSize  = 210
)

// DequantizeQ6KBlock decodes one Q6_K super-block from b (210 bytes) into
// out (256 values), mirroring ggml's dequantize_row_q6_K.
func DequantizeQ6KBlock(out []float32, b []byte) {
	_ = out[255]
	_ = b[209]
	ql := b[:128]
	qh := b[128:192]
	sc := b[192:208]
	d := F16ToF32(binary.LittleEndian.Uint16(b[208:]))

	for chunk := 0; chunk < 2; chunk++ {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		sco := sc[chunk*8:]
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32
			q2 := int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32
			q3 := int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32
			q4 := int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32
			out[n0+l] = d * float32(int8(sco[is+0])) * float32(q1)
			out[n0+l+32] = d * float32(int8(sco[is+2])) * float32(q2)
			out[n0+l+64] = d * float32(int8(sco[is+4])) * float32(q3)
			out[n0+l+96] = d * float32(int8(sco[is+6])) * float32(q4)
		}
	}
}

// dotQ6KQ8KBlock computes the dot product of one Q6_K weight super-block and
// one Q8_K activation super-block, matching ggml_vec_dot_q6_K_q8_K's integer
// accumulation order.
func dotQ6KQ8KBlock(b []byte, yd float32, q8 []int8) float32 {
	_ = b[209]
	_ = q8[255]
	ql := b[:128]
	qh := b[128:192]
	sc := b[192:208]
	d := F16ToF32(binary.LittleEndian.Uint16(b[208:])) * yd

	var isum int32
	for chunk := 0; chunk < 2; chunk++ {
		n0 := chunk * 128
		qlo := ql[chunk*64:]
		qho := qh[chunk*32:]
		sco := sc[chunk*8:]
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int32(int8((qlo[l]&0x0F)|(((qho[l]>>0)&3)<<4)) - 32)
			q2 := int32(int8((qlo[l+32]&0x0F)|(((qho[l]>>2)&3)<<4)) - 32)
			q3 := int32(int8((qlo[l]>>4)|(((qho[l]>>4)&3)<<4)) - 32)
			q4 := int32(int8((qlo[l+32]>>4)|(((qho[l]>>6)&3)<<4)) - 32)
			isum += int32(q8[n0+l])*int32(int8(sco[is+0]))*q1 +
				int32(q8[n0+l+32])*int32(int8(sco[is+2]))*q2 +
				int32(q8[n0+l+64])*int32(int8(sco[is+4]))*q3 +
				int32(q8[n0+l+96])*int32(int8(sco[is+6]))*q4
		}
	}
	return d * float32(isum)
}
