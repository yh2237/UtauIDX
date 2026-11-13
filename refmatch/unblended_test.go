package refmatch

import (
	"math"
	"reflect"
	"testing"
)

func TestUnblendedUsesTopCandidateAsIs(t *testing.T) {
	const rate, fft = 16000, 1024
	index := &Index{
		Header:   Header{Frames: 2, Dims: 2, SampleRate: rate, FFTSize: fft, APBands: 1},
		Features: []float32{1, 0, 0.8, 0.6}, File: []int32{0, 1}, Position: []int32{0, 0},
		F0: make([]float32, 2*subFrames), Spectrum: make([]float32, 2*subFrames*spectrumDims),
		Aperiodicity: make([]float32, 2*subFrames),
	}
	for i := range index.Spectrum {
		index.Spectrum[i] = float32(i%7)*0.05 - float32(i%3)*0.02
	}
	request := Request{Features: []float32{1, 0}, Frames: 1, ReferenceF0: make([]float64, subFrames), Unblended: true}
	got, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	c := newCodec(rate, fft, spectrumDims)
	bins := fft/2 + 1
	coded, want := make([]float64, spectrumDims), make([]float64, bins)
	for row := range subFrames {
		for d := range coded {
			coded[d] = float64(index.Spectrum[row*spectrumDims+d])
		}
		c.decodeSpectrum(coded, want)
		for i, value := range got.Spectrum[row*bins : (row+1)*bins] {
			if math.Abs(value-want[i]) > 1e-9*math.Abs(want[i]) {
				t.Fatalf("row %d bin %d: got %g want %g", row, i, value, want[i])
			}
		}
	}
	request.Unblended = false
	blended, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(blended.Spectrum, got.Spectrum) {
		t.Fatal("blended output equals the top candidate")
	}
}
