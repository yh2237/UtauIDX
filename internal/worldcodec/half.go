package worldcodec

import "math"

// Float32ToHalf converts a float32 to an IEEE-754 binary16 bit pattern.
func Float32ToHalf(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits>>16) & 0x8000
	exponent := int((bits>>23)&0xff) - 127 + 15
	mantissa := bits & 0x7fffff
	switch {
	case exponent <= 0:
		if exponent < -10 {
			return sign
		}
		mantissa |= 0x800000
		shift := uint(14 - exponent)
		half := uint16(mantissa >> shift)
		if (mantissa>>(shift-1))&1 != 0 {
			half++
		}
		return sign | half
	case exponent >= 31:
		return sign | 0x7c00
	}
	half := sign | uint16(exponent)<<10 | uint16(mantissa>>13)
	if mantissa&0x1000 != 0 {
		half++
	}
	return half
}

// HalfToFloat32 converts an IEEE-754 binary16 bit pattern back to float32.
func HalfToFloat32(value uint16) float32 {
	sign := uint32(value&0x8000) << 16
	exponent := uint32(value>>10) & 0x1f
	mantissa := uint32(value & 0x3ff)
	switch {
	case exponent == 0:
		subnormal := float32(mantissa) / 1024 / 16384
		if sign != 0 {
			subnormal = -subnormal
		}
		return subnormal
	case exponent == 31:
		return math.Float32frombits(sign | 0x7f800000 | mantissa<<13)
	}
	return math.Float32frombits(sign | (exponent+112)<<23 | mantissa<<13)
}

// RoundToHalf quantizes each value through binary16 and decodes back to float32.
func RoundToHalf(values []float32) []float32 {
	out := make([]float32, len(values))
	for i, value := range values {
		out[i] = HalfToFloat32(Float32ToHalf(value))
	}
	return out
}
