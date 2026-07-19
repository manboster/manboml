package ops

import "math"

// F16ToF32 converts an IEEE-754 binary16 bit pattern to float32, preserving
// subnormals, infinities and NaNs.
func F16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3FF
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1F:
		return math.Float32frombits(sign | 0xFF<<23 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// F32ToF16 converts a float32 to an IEEE-754 binary16 bit pattern using
// round-to-nearest-even, matching llama.cpp's default F32 to F16 conversion.
func F32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23) & 0xFF
	mant := b & 0x7FFFFF
	if exp == 0xFF {
		if mant != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	e := exp - 127 + 15
	switch {
	case e >= 31:
		return sign | 0x7C00
	case e <= 0:
		if e < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - e)
		half := mant >> shift
		rem := mant & (1<<shift - 1)
		if rem > 1<<(shift-1) || (rem == 1<<(shift-1) && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	default:
		half := uint16(e)<<10 | uint16(mant>>13)
		rem := mant & 0x1FFF
		if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
			half++
		}
		return sign | half
	}
}

// F16SliceToF32 converts len(dst) binary16 values from src to float32.
// src and dst must have the same length.
func F16SliceToF32(dst []float32, src []uint16) {
	for i, h := range src {
		dst[i] = F16ToF32(h)
	}
}

// F32SliceToF16 converts len(dst) float32 values from src to binary16.
// src and dst must have the same length.
func F32SliceToF16(dst []uint16, src []float32) {
	for i, f := range src {
		dst[i] = F32ToF16(f)
	}
}
