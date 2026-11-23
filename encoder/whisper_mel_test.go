package encoder

import (
	"math"
	"testing"

	"github.com/yh2237/gograd/dsp"
)

func TestSlaneyMelFilters(t *testing.T) {
	filters := slaneyMelFilters(80)
	for j := 0; j < 80; j++ {
		var area float64
		for _, row := range filters {
			if row[j] < 0 {
				t.Fatalf("negative weight in filter %d", j)
			}
			area += 40 * row[j]
		}
		if math.Abs(area-1) > 0.34 {
			t.Errorf("filter %d: approximate area %.8f", j, area)
		}
	}
	// The first filter is in Slaney's linear region. Its continuous triangle
	// is symmetric about this center, and its discrete samples follow its slopes.
	center := (200.0 / 3) * (15 + 27*math.Log(8)/math.Log(6.4)) / 81
	for i, row := range filters {
		want := max(0, 1-math.Abs(float64(i)*40-center)/center) / center
		if math.Abs(row[0]-want) > 1e-10 {
			t.Fatalf("first filter bin %d: %.12g, want %.12g", i, row[0], want)
		}
	}
}

func naiveWhisperLogMel(seg []float32, frames int, filters [][]float64) []float32 {
	wave := make([]float32, frames*whisperHop)
	copy(wave, seg)
	padded := make([]float32, len(wave)+whisperFFT)
	half := whisperFFT / 2
	copy(padded[half:], wave)
	for i := 0; i < half; i++ {
		padded[i] = wave[half-i]
		padded[len(padded)-1-i] = wave[len(wave)-1-half+i]
	}
	window := dsp.HannPeriodic(whisperFFT)
	cosines := make([][whisperFFT]float64, whisperBins)
	sines := make([][whisperFFT]float64, whisperBins)
	for k := range cosines {
		for i, weight := range window {
			s, c := math.Sincos(2 * math.Pi * float64(k*i) / whisperFFT)
			cosines[k][i] = c * float64(weight)
			sines[k][i] = s * float64(weight)
		}
	}

	mels := len(filters[0])
	output := make([]float32, frames*mels)
	peak := -10.0
	row := make([]float64, mels)
	for frame := 0; frame < frames; frame++ {
		clear(row)
		input := padded[frame*whisperHop:][:whisperFFT]
		for k := 0; k < whisperBins; k++ {
			var real, imaginary float64
			for i, sample := range input {
				real += float64(sample) * cosines[k][i]
				imaginary -= float64(sample) * sines[k][i]
			}
			power := real*real + imaginary*imaginary
			for j, weight := range filters[k] {
				row[j] += power * weight
			}
		}
		for j, power := range row {
			logPower := math.Log10(max(1e-10, power))
			output[frame*mels+j] = float32(logPower)
			peak = max(peak, logPower)
		}
	}
	for i, value := range output {
		output[i] = (max(value, float32(peak-8)) + 4) / 4
	}
	return output
}

func TestWhisperLogMelSilencePadding(t *testing.T) {
	const frames = 3000
	filters := slaneyMelFilters(80)
	wave := testWave(32000)
	got := whisperLogMel(wave, frames, filters)
	want := naiveWhisperLogMel(wave, frames, filters)
	var maxDiff float64
	for i, value := range got {
		maxDiff = max(maxDiff, math.Abs(float64(value)-float64(want[i])))
	}
	t.Logf("full STFT maxDiff=%.9g", maxDiff)
	if maxDiff >= 1e-5 {
		t.Errorf("fast log-mel differs by %.9g from full STFT", maxDiff)
	}
	silence := got[203*80]
	for _, v := range got[203*80:] {
		if v != silence {
			t.Fatal("tail differs from the clipped silence floor")
		}
	}
}
