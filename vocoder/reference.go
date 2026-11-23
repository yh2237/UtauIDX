package vocoder

import (
	"fmt"
	"math"
	"slices"

	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/gograd/dsp"
)

const analysisRate = 44100

type ReferenceAnalysis struct {
	Frames      int
	SampleRate  int
	FFTSize     int
	F0          []float64 // 5 ms F0 in Hz
	SafeRefAP   []float32 // 5 ms sibilant detection
	Levels      []float64 // 5 ms dB relative to p95
	MouthStatic []float32 // 12 values per 20 ms frame
	MouthDelta  []float32 // 12 values per 20 ms frame
	IsUV20      []bool    // 20 ms unvoiced mask derived from spectral energy
}

// Voiced20ms uses the same F0 gating as synthesis.
func (a ReferenceAnalysis) Voiced20ms(useSafeRefAP bool) []bool {
	f0 := slices.Clone(a.F0)
	if useSafeRefAP {
		for row, ap := range a.SafeRefAP[:min(len(a.SafeRefAP), len(f0))] {
			if ap >= 0.5 {
				f0[row] = 0
			}
		}
	}
	return worldcodec.FrameVoicing(f0)
}

// AnalyzeReference extracts levels, pitch, sibilant cues, and Mouth features.
func AnalyzeReference(world index.World, refSamples []float64, refRate int, frames int, externalF0 []float32, externalF0FrameMS float64) (ReferenceAnalysis, error) {
	if world == nil || len(refSamples) == 0 || refRate <= 0 || frames <= 0 ||
		frames > math.MaxInt/(worldcodec.SubFrames*worldcodec.MouthDims) {
		return ReferenceAnalysis{}, fmt.Errorf("vocoder: invalid reference analysis input")
	}
	if len(externalF0) > 0 && (!finite(externalF0FrameMS) || externalF0FrameMS <= 0) {
		return ReferenceAnalysis{}, fmt.Errorf("vocoder: invalid external F0 frame period")
	}
	rows := frames * worldcodec.SubFrames
	levels := ReferenceLevels(refSamples, refRate, rows)
	x44 := refSamples
	if refRate != analysisRate {
		x := make([]float32, len(refSamples))
		for i, v := range refSamples {
			x[i] = float32(v)
		}
		resampled := dsp.Resample(x, refRate, analysisRate)
		x44 = make([]float64, len(resampled))
		for i, v := range resampled {
			x44[i] = float64(v)
		}
	}
	harvestF0, spectrum, ap, fftSize, err := world.Analyze(x44, analysisRate, worldcodec.WorldFrameMS, index.D4CThreshold)
	if err != nil {
		return ReferenceAnalysis{}, fmt.Errorf("vocoder: analyze reference: %w", err)
	}
	bins := fftSize/2 + 1
	available := min(rows, len(harvestF0))
	if fftSize < 2 || fftSize%2 != 0 || available == 0 || available > len(spectrum)/bins || available > len(ap)/bins {
		return ReferenceAnalysis{}, fmt.Errorf("vocoder: invalid WORLD reference analysis shape")
	}

	type bandMask struct{ ap, low, high bool }
	masks := make([]bandMask, bins)
	var apBins int
	for b := range bins {
		hz := float64(b) * analysisRate / float64(fftSize)
		masks[b] = bandMask{hz >= 100 && hz < 3000, hz >= 100 && hz < 900, hz >= 2800 && hz < 10000}
		if masks[b].ap {
			apBins++
		}
	}
	codec := worldcodec.New(analysisRate, fftSize, worldcodec.SpectrumDims)
	var code [worldcodec.SpectrumDims]float64
	spRow := make([]float64, bins)
	hiDB := make([]float64, rows)
	uvRaw := make([]bool, rows)
	safeAP := make([]float32, rows)
	mouth := make([]float64, frames*worldcodec.MouthDims)
	for row := range rows {
		src := min(row, available-1) * bins
		low, high, ap03 := 1e-12, 1e-12, 0.0
		for b, mask := range masks {
			power := spectrum[src+b]
			spRow[b] = max(power, 1e-30)
			if mask.low {
				low += power
			}
			if mask.high {
				high += power
			}
			if mask.ap {
				ap03 += ap[src+b]
			}
		}
		ap03 /= float64(max(apBins, 1))
		hiDB[row] = 10 * math.Log10(high/low)
		uvRaw[row] = hiDB[row] > 4 && levels[row] > -32 && ap03 > 0.65
		safeAP[row] = 0.10
		if ap03 > 0.50 {
			safeAP[row] = 0.45
		}

		codec.CodeSpectrum(spRow, code[:])
		offset := (row / worldcodec.SubFrames) * worldcodec.MouthDims
		for d := range worldcodec.MouthDims {
			mouth[offset+d] += code[d+2] / worldcodec.SubFrames
		}
	}
	for row, uv := range uvRaw {
		if uv && ((row > 0 && uvRaw[row-1]) || (row+1 < rows && uvRaw[row+1])) {
			safeAP[row] = 0.90
		}
	}
	isUV20 := make([]bool, frames)
	voiced20 := make([]bool, frames)
	for frame := range frames {
		start := frame * worldcodec.SubFrames
		slope := sum(hiDB[start:start+worldcodec.SubFrames]) / worldcodec.SubFrames
		level := sum(levels[start:start+worldcodec.SubFrames]) / worldcodec.SubFrames
		isUV20[frame] = slope > 2 && level > -32
		voiced20[frame] = !isUV20[frame]
	}
	static, delta := worldcodec.ComputeQueryMouth(mouth, frames, voiced20)
	f0 := make([]float64, rows)
	if len(externalF0) > 0 {
		f0 = ReferenceF0(externalF0, externalF0FrameMS, frames)
	} else {
		copy(f0, harvestF0)
	}
	FillVoicedByLevel(f0, levels, -35)
	return ReferenceAnalysis{
		Frames: frames, SampleRate: analysisRate, FFTSize: fftSize,
		F0: f0, SafeRefAP: safeAP, Levels: levels,
		MouthStatic: static, MouthDelta: delta, IsUV20: isUV20,
	}, nil
}

// ReferenceLevels computes p95-relative energy over three 5 ms hops.
func ReferenceLevels(samples []float64, sampleRate, rows int) []float64 {
	if rows <= 0 {
		return nil
	}
	levels := make([]float64, rows)
	hop := sampleRate / (1000 / int(worldcodec.WorldFrameMS))
	if hop <= 0 {
		return levels
	}
	for row := range rows {
		var energy float64
		first, last := max(0, (row-1)*hop), min(len(samples), (row+2)*hop)
		for i := first; i < last; i++ {
			energy += samples[i] * samples[i]
		}
		levels[row] = 10 * math.Log10(energy/(3*float64(hop))+1e-12)
	}
	peak := worldcodec.Percentile(levels, 95)
	for i := range levels {
		levels[i] -= peak
	}
	return levels
}

// ReferenceF0 smooths log pitch over 20 ms and interpolates it to the WORLD grid.
func ReferenceF0(f0 []float32, f0FrameMS float64, frames int) []float64 {
	if frames <= 0 || frames > math.MaxInt/worldcodec.SubFrames {
		return nil
	}
	out := make([]float64, frames*worldcodec.SubFrames)
	if f0FrameMS <= 0 || !finite(f0FrameMS) {
		return out
	}
	var times, logF0 []float64
	voicedness := make([]float64, len(f0))
	axis := make([]float64, len(f0))
	for i, v := range f0 {
		axis[i] = float64(i) * f0FrameMS
		if v > 60 {
			voicedness[i] = 1
			times = append(times, axis[i])
			logF0 = append(logF0, math.Log(float64(v)))
		}
	}
	if len(times) == 0 {
		return out
	}
	filled := make([]float64, len(f0))
	for i, t := range axis {
		filled[i] = worldcodec.InterpClamped(times, logF0, t)
	}
	smoothed := worldcodec.GaussianFilter(filled, 20/f0FrameMS)
	for i := range out {
		t := float64(i) * worldcodec.WorldFrameMS
		if worldcodec.InterpClamped(axis, voicedness, t) > 0.5 {
			out[i] = math.Exp(worldcodec.InterpClamped(axis, smoothed, t))
		}
	}
	return out
}

// FillVoicedByLevel fills pitch gaps above thresholdDB using neighboring voiced rows.
func FillVoicedByLevel(f0, levels []float64, thresholdDB float64) {
	if len(levels) < len(f0) {
		return
	}
	previous := -1
	for i := range f0 {
		if f0[i] <= 0 {
			continue
		}
		if previous >= 0 {
			for k := previous + 1; k < i; k++ {
				if levels[k] > thresholdDB {
					f0[k] = f0[previous] + (f0[i]-f0[previous])*float64(k-previous)/float64(i-previous)
				}
			}
		}
		previous = i
	}
	if previous >= 0 {
		for k := previous + 1; k < len(f0) && levels[k] > thresholdDB; k++ {
			f0[k] = f0[previous]
		}
	}
}

// OutputF0 shifts the median to centerHz and scales deviations in log frequency.
func OutputF0(reference []float64, defaultCenterHz, centerHz, scale float64) []float64 {
	if centerHz <= 0 {
		centerHz = defaultCenterHz
	}
	var voiced []float64
	for _, v := range reference {
		if v > 0 {
			voiced = append(voiced, v)
		}
	}
	out := make([]float64, len(reference))
	if len(voiced) == 0 {
		return out
	}
	median := worldcodec.Percentile(voiced, 50)
	for i, v := range reference {
		if v > 0 {
			out[i] = centerHz * math.Exp(scale*math.Log(math.Max(v, 1)/median))
		}
	}
	return out
}
