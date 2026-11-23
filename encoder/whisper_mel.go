package encoder

import (
	"math"

	"github.com/yh2237/gograd/dsp"
)

const (
	whisperFFT  = 400
	whisperHop  = 160
	whisperBins = whisperFFT/2 + 1
)

// Slaney filters are area-normalized triangles with breakpoints spaced evenly
// in mel. The triangle slopes are measured in hertz.
func slaneyMelFilters(mels int) [][]float64 {
	const linearStep = 200.0 / 3
	logStep := math.Log(6.4) / 27
	melMax := 15 + math.Log(8)/logStep
	edges := make([]float64, mels+2)
	for i := range edges {
		mel := melMax * float64(i) / float64(mels+1)
		edges[i] = linearStep * mel
		if mel >= 15 {
			edges[i] = 1000 * math.Exp(logStep*(mel-15))
		}
	}

	filters := make([][]float64, whisperBins)
	for bin := range filters {
		row := make([]float64, mels)
		hz := float64(bin) * SampleRate / whisperFFT
		for j := range row {
			left, center, right := edges[j], edges[j+1], edges[j+2]
			rise := (hz - left) / (center - left)
			fall := (right - hz) / (right - center)
			row[j] = max(0, min(rise, fall)) * 2 / (right - left)
		}
		filters[bin] = row
	}
	return filters
}

type whisperSTFT struct {
	fft    *dsp.FFT
	window []float32
	chirp  [whisperFFT]complex128
	kernel []float64
}

var whisperSpectrum = newWhisperSTFT()

// Bluestein's algorithm computes the 400-point transform using gograd's
// radix-2 FFT. The convolution needs at least 2*400-1 points.
func newWhisperSTFT() *whisperSTFT {
	const size = 1024
	fft, err := dsp.NewFFT(size)
	if err != nil {
		panic(err)
	}
	s := &whisperSTFT{fft: fft, window: dsp.HannPeriodic(whisperFFT), kernel: make([]float64, size*2)}
	for i := range s.chirp {
		angle := math.Pi * float64((i*i)%(2*whisperFFT)) / whisperFFT
		sin, cos := math.Sincos(angle)
		s.chirp[i] = complex(cos, -sin)
		s.kernel[2*i], s.kernel[2*i+1] = cos, sin
		if i > 0 {
			s.kernel[2*(size-i)], s.kernel[2*(size-i)+1] = cos, sin
		}
	}
	fft.Transform(s.kernel, false)
	return s
}

func whisperSample(seg []float32, index, sampleCount int) float32 {
	if index < 0 {
		index = -index
	} else if index >= sampleCount {
		index = 2*(sampleCount-1) - index
	}
	if index >= 0 && index < len(seg) {
		return seg[index]
	}
	return 0
}

func (s *whisperSTFT) power(seg []float32, frame, sampleCount int, scratch []float64, out []float64) {
	clear(scratch)
	left := frame*whisperHop - whisperFFT/2
	for i, window := range s.window {
		sample := whisperSample(seg, left+i, sampleCount)
		value := complex(float64(sample*window), 0) * s.chirp[i]
		scratch[i*2], scratch[i*2+1] = real(value), imag(value)
	}
	s.fft.Transform(scratch, false)
	for i := 0; i < len(scratch); i += 2 {
		z := complex(scratch[i], scratch[i+1]) * complex(s.kernel[i], s.kernel[i+1])
		scratch[i], scratch[i+1] = real(z), imag(z)
	}
	s.fft.Transform(scratch, true)
	for i := range out {
		// The output chirp has unit magnitude, so it drops out of the power.
		re, im := scratch[2*i], scratch[2*i+1]
		out[i] = re*re + im*im
	}
}

// whisperLogMel returns time-major log-mel rows. The waveform is conceptually
// zero-padded to frames*whisperHop before the centered, reflect-padded STFT.
func whisperLogMel(seg []float32, frames int, filters [][]float64) []float32 {
	mels := len(filters[0])
	sampleCount := frames * whisperHop
	seg = seg[:min(len(seg), sampleCount)]
	signalFrames := min(frames, (len(seg)+whisperFFT/2+whisperHop-1)/whisperHop)
	logMel := make([]float32, frames*mels)
	scratch := make([]float64, whisperSpectrum.fft.Len()*2)
	var power [whisperBins]float64
	row := make([]float64, mels)
	const floor = -10.0
	peak := floor
	for frame := 0; frame < signalFrames; frame++ {
		whisperSpectrum.power(seg, frame, sampleCount, scratch, power[:])
		clear(row)
		for bin, value := range power {
			for mel, weight := range filters[bin] {
				row[mel] += value * weight
			}
		}
		for mel, value := range row {
			logValue := math.Log10(max(value, 1e-10))
			logMel[frame*mels+mel] = float32(logValue)
			peak = max(peak, logValue)
		}
	}

	threshold := float32(peak - 8)
	for i := range logMel[:signalFrames*mels] {
		logMel[i] = (max(logMel[i], threshold) + 4) / 4
	}
	silence := (max(float32(floor), threshold) + 4) / 4
	for i := signalFrames * mels; i < len(logMel); i++ {
		logMel[i] = silence
	}
	return logMel
}
