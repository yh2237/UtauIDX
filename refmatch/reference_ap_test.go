package refmatch

import (
	"math"
	"reflect"
	"slices"
	"testing"
)

func TestReferenceAPParameters(t *testing.T) {
	index := &Index{Header: Header{Frames: 1, Dims: 1, SampleRate: 16000, FFTSize: 512, APBands: 1, F0Median: 200}, Features: []float32{1}, Spectrum: make([]float32, 4*60), Aperiodicity: []float32{-40, -40, -40, -40}, F0: []float32{200, 200, 200, 200}, File: []int32{0}}
	request := Request{Frames: 1, Features: []float32{1}, ReferenceF0: []float64{200, 200, 200, 0}, OutputF0: []float64{220, 220, 220, 220}}
	baseline, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	request.ReferenceAP = []float32{0.1, 0.9, 1, 0.1}
	got, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.F0, []float64{220, 0, 0, 0}) {
		t.Fatalf("F0 = %v", got.F0)
	}
	if !slices.Equal(request.OutputF0, []float64{220, 220, 220, 220}) || !slices.Equal(request.ReferenceF0, []float64{200, 200, 200, 0}) {
		t.Fatal("caller F0 modified")
	}
	bins := index.FFTSize/2 + 1
	if got.Aperiodicity[bins] < 0.9 || got.Aperiodicity[2*bins] != 1 {
		t.Fatal("high AP was not preserved")
	}
	want := 0.95*baseline.Aperiodicity[0] + 0.05*float64(request.ReferenceAP[0])
	if math.Abs(got.Aperiodicity[0]-want) > 1e-12 {
		t.Fatalf("voiced AP = %v want %v", got.Aperiodicity[0], want)
	}
	request.ReferenceAP = nil
	again, err := index.Parameters(request)
	if err != nil || !reflect.DeepEqual(baseline, again) {
		t.Fatal("nil AP changed baseline")
	}
	for _, ap := range [][]float32{{}, {1}, {0, 0, 0, float32(math.NaN())}, {0, 0, 0, float32(math.Inf(1))}, {0, 0, 0, -0.1}, {0, 0, 0, 1.1}} {
		request.ReferenceAP = ap
		if _, err := index.Parameters(request); err == nil {
			t.Fatalf("accepted invalid AP %v", ap)
		}
	}
}
