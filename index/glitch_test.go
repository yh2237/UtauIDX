package index

import (
	"math"
	"slices"
	"testing"
)

func TestComputeC35(t *testing.T) {
	spectrum := make([]float32, 2*SubFrames*SpectrumDims)
	for s := range SubFrames {
		base := s * SpectrumDims
		spectrum[base+2] = 900
		copy(spectrum[base+3:], []float32{1, 2, 3, 900})
		copy(spectrum[base+SubFrames*SpectrumDims+3:], []float32{4, 5, 6})
	}
	for _, tc := range []struct {
		voiced []bool
		meanSq float64
	}{
		{[]bool{true, false}, 14.0 / 3},
		{[]bool{false, true}, 77.0 / 3},
		{[]bool{false, false}, 91.0 / 6},
		{nil, 91.0 / 6},
	} {
		if got, want := ComputeC35(spectrum, tc.voiced), math.Sqrt(tc.meanSq); got != want {
			t.Fatalf("C35 for %v = %g, want %g", tc.voiced, got, want)
		}
	}
	if got := ComputeC35(nil, nil); got != 0 {
		t.Fatalf("empty C35 = %g", got)
	}
}

func TestComputeGlitchPenalties(t *testing.T) {
	f0 := []float32{
		100, 100, 100, 100,
		200, 100, 100, 100,
		100, 100, 0, 0,
		100, 100, 100, 100,
		200, 0, 0, 0,
		160, 160, 0, 0,
		0, 0, 0, 0,
	}
	ap := make([]float32, len(f0)*2)
	for row := range f0 {
		copy(ap[row*2:], []float32{-14, -16})
	}
	for row := 8; row < 12; row++ {
		copy(ap[row*2:], []float32{-0.25, -0.25})
	}
	for row := 12; row < 16; row++ {
		copy(ap[row*2:], []float32{-3, -16})
	}
	voiced := []bool{true, true, true, true, false, true, true}
	ids := []int32{0, 0, 0, 0, 0, 1, 1}
	want := []float32{0, 0.10, 0.12, 0.06, 0, 0.10, 0}
	if got := ComputeGlitchPenalties(f0, ap, 2, voiced, ids, 200); !slices.Equal(got, want) {
		t.Fatalf("penalties = %v, want %v", got, want)
	}
	want = []float32{0, 0.10, 0, 0, 0, 0.10, 0}
	if got := ComputeGlitchPenalties(f0, nil, 0, voiced, ids, 0); !slices.Equal(got, want) {
		t.Fatalf("penalties without AP = %v, want %v", got, want)
	}
}
