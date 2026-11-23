// Package worldcodec provides internal WORLD spectral/aperiodicity coding, IEEE-754 float16 conversion,
// signal math helpers, and spectral Mouth cepstral feature extraction shared by UtauIDX v2 packages.
package worldcodec

import (
	"math"
	"sort"
)

const (
	// FrameMS is the 20ms feature/search frame period.
	FrameMS = 20.0
	// SubFrames is the number of 5ms WORLD frames per 20ms feature frame.
	SubFrames = 4
	// WorldFrameMS is the 5ms WORLD analysis/synthesis period.
	WorldFrameMS = FrameMS / SubFrames
	// SpectrumDims is the number of coded WORLD spectral envelope coefficients.
	SpectrumDims = 60

	melM0             = 1127.01048
	melF0             = 700.0
	codecFloorHz      = 40.0
	codecCeilHz       = 20000.0
	apIntervalHz      = 3000.0
	apUpperLimitHz    = 15000.0
	worldSafeGuardMin = 0.000000000001
)

// Codec matches WORLD's CodeSpectralEnvelope, DecodeSpectralEnvelope,
// CodeAperiodicity, and DecodeAperiodicity. It is not safe for concurrent use
// because scratch buffers are reused across calls.
type Codec struct {
	rate, fftSize, dims int
	codeMelAxis         []float64
	codeFreqAxis        []float64
	codeBasis           []float64
	decodeMelAxis       []float64
	decodeFreqAxis      []float64
	decodeBasis         []float64
	apBands             int

	logSpectrum []float64
	codeMel     []float64
	codeV       []float64
	decodeOut   []float64
	decodeMel   []float64
	decodeValue []float64
	apLog       []float64
	apAxis      []float64
	apValues    []float64
	apFreq      []float64
	apInterp    []float64
}

// New constructs a Codec for the given sample rate, FFT size, and spectral coding dimensions.
func New(rate, fftSize, dims int) *Codec {
	half := fftSize / 2
	toMel := func(hz float64) float64 { return melM0 * math.Log(hz/melF0+1) }
	floor, ceil := toMel(codecFloorHz), toMel(math.Min(float64(rate)/2, codecCeilHz))
	apBands := max(0, int(math.Min(apUpperLimitHz, float64(rate)/2-apIntervalHz)/apIntervalHz))
	c := &Codec{
		rate:           rate,
		fftSize:        fftSize,
		dims:           dims,
		codeMelAxis:    make([]float64, half),
		codeFreqAxis:   make([]float64, half+1),
		codeBasis:      make([]float64, dims*half),
		decodeMelAxis:  make([]float64, half+2),
		decodeFreqAxis: make([]float64, half+1),
		decodeBasis:    make([]float64, dims*half),
		apBands:        apBands,
		logSpectrum:    make([]float64, half+1),
		codeMel:        make([]float64, half),
		codeV:          make([]float64, half),
		decodeOut:      make([]float64, half),
		decodeMel:      make([]float64, half+2),
		decodeValue:    make([]float64, half+1),
		apLog:          make([]float64, half+1),
		apAxis:         make([]float64, apBands+2),
		apValues:       make([]float64, apBands+2),
		apFreq:         make([]float64, half+1),
		apInterp:       make([]float64, half+1),
	}
	for i := 0; i < half; i++ {
		mel := (ceil-floor)*float64(i)/float64(half) + floor
		c.codeMelAxis[i] = mel
		c.decodeMelAxis[i+1] = melF0 * (math.Exp(mel/melM0) - 1)
	}
	c.decodeMelAxis[half+1] = float64(rate) / 2
	for i := 0; i <= half; i++ {
		hz := float64(i) * float64(rate) / float64(fftSize)
		c.codeFreqAxis[i] = toMel(hz)
		c.decodeFreqAxis[i] = hz
	}
	codeScale := 2 / math.Sqrt(float64(fftSize)) / math.Sqrt(float64(half))
	decodeScale := math.Sqrt(float64(fftSize)) * math.Sqrt(float64(half))
	for k := 0; k < dims; k++ {
		theta := math.Pi * float64(k) / float64(fftSize)
		cs, ds := codeScale, decodeScale
		if k == 0 {
			cs /= math.Sqrt2
			ds /= math.Sqrt2
		}
		for p := 0; p < half; p++ {
			angle := 2 * math.Pi * float64(k*p%half) / float64(half)
			c.codeBasis[k*half+p] = cs * math.Cos(theta-angle)
			c.decodeBasis[k*half+p] = ds * math.Cos(angle+theta)
		}
	}
	return c
}

// APBands returns the number of coded aperiodicity sub-bands for this Codec.
func (c *Codec) APBands() int { return c.apBands }

// Dims returns the number of coded spectral envelope dimensions.
func (c *Codec) Dims() int { return c.dims }

// FFTSize returns the FFT size configured on this Codec.
func (c *Codec) FFTSize() int { return c.fftSize }

// Rate returns the sample rate configured on this Codec.
func (c *Codec) Rate() int { return c.rate }

// CodeSpectrum encodes a linear WORLD spectral envelope row into coded mel-cepstral coefficients.
func (c *Codec) CodeSpectrum(spectrum, coded []float64) {
	half := c.fftSize / 2
	logSpectrum := c.logSpectrum[:half+1]
	for i, v := range spectrum[:half+1] {
		logSpectrum[i] = math.Log(v)
	}
	mel := interp1Into(c.codeFreqAxis, logSpectrum, c.codeMelAxis, c.codeMel[:half])
	v := c.codeV[:half]
	if half%2 != 0 {
		clear(v)
	}
	for i := 0; i < half/2; i++ {
		v[i] = mel[i*2]
		v[i+half/2] = mel[half-i*2-1]
	}
	for k := 0; k < c.dims; k++ {
		basis := c.codeBasis[k*half : (k+1)*half]
		var sum float64
		for p, x := range v {
			sum += x * basis[p]
		}
		coded[k] = sum
	}
}

// DecodeSpectrum reconstructs a linear WORLD spectral envelope row from coded coefficients.
func (c *Codec) DecodeSpectrum(coded, spectrum []float64) {
	half := c.fftSize / 2
	out := c.decodeOut[:half]
	clear(out)
	for k := 0; k < c.dims; k++ {
		basis := c.decodeBasis[k*half : (k+1)*half]
		for p := range out {
			out[p] += coded[k] * basis[p]
		}
	}
	mel := c.decodeMel[:half+2]
	if half%2 != 0 {
		clear(mel)
	}
	for i := 0; i < half/2; i++ {
		mel[1+i*2] = out[i]
		mel[2+i*2] = out[half-i-1]
	}
	mel[0], mel[half+1] = mel[1], mel[half]
	values := interp1Into(c.decodeMelAxis, mel, c.decodeFreqAxis, c.decodeValue[:half+1])
	for i, v := range values {
		spectrum[i] = math.Exp(v / float64(half))
	}
}

// CodeAperiodicity encodes a linear WORLD aperiodicity row into band dB values.
func (c *Codec) CodeAperiodicity(aperiodicity, coded []float64) {
	bins := c.fftSize/2 + 1
	step := float64(c.rate) / float64(c.fftSize)
	logAP := c.apLog[:bins]
	for i, v := range aperiodicity[:bins] {
		logAP[i] = 20 * math.Log10(v)
	}
	for band := 0; band < c.apBands; band++ {
		position := apIntervalHz * float64(band+1) / step
		base := int(position)
		fraction := position - float64(base)
		delta := 0.0
		if base+1 < bins {
			delta = logAP[base+1] - logAP[base]
		}
		coded[band] = logAP[base] + delta*fraction
	}
}

// DecodeAperiodicity reconstructs a linear WORLD aperiodicity row from coded band dB values.
func (c *Codec) DecodeAperiodicity(coded, aperiodicity []float64) {
	bins := c.fftSize/2 + 1
	for i := range aperiodicity[:bins] {
		aperiodicity[i] = 1 - worldSafeGuardMin
	}
	bands := max(0, min(c.apBands, len(coded)))
	if bands > 0 {
		var mean float64
		for _, v := range coded[:bands] {
			mean += v
		}
		if mean/float64(bands) > -0.5 {
			return
		}
	}
	axis := c.apAxis[:bands+2]
	values := c.apValues[:bands+2]
	for i := 0; i <= bands; i++ {
		axis[i] = float64(i) * apIntervalHz
	}
	axis[bands+1] = float64(c.rate) / 2
	values[0] = -60
	copy(values[1:], coded[:bands])
	values[bands+1] = -worldSafeGuardMin
	freq := c.apFreq[:bins]
	for i := range freq {
		freq[i] = float64(c.rate) / float64(c.fftSize) * float64(i)
	}
	out := interp1Into(axis, values, freq, c.apInterp[:bins])
	for i, v := range out {
		aperiodicity[i] = math.Pow(10, v/20)
	}
}

func interp1Into(x, y, xi, out []float64) []float64 {
	out = out[:len(xi)]
	for i, v := range xi {
		k := sort.Search(len(x), func(j int) bool { return x[j] > v })
		k = min(max(k, 1), len(x)-1)
		out[i] = y[k-1] + (v-x[k-1])/(x[k]-x[k-1])*(y[k]-y[k-1])
	}
	return out
}
