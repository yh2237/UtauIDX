package refmatch

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"testing"
)

func TestCepstrumExtractionAndReference(t *testing.T) {
	const rate, fft, frames = 16000, 1024, 2
	bins := fft/2 + 1
	spectrum := make([]float64, frames*subFrames*bins)
	codes := make([]float32, frames*subFrames*spectrumDims)
	codec := newCodec(rate, fft, spectrumDims)
	code := make([]float64, spectrumDims)
	for row := range frames * subFrames {
		for b := range bins {
			spectrum[row*bins+b] = 1 + math.Exp(-math.Pow((float64(b)-float64(30+row*3))/15, 2))*10
		}
		codec.codeSpectrum(spectrum[row*bins:(row+1)*bins], code)
		for d, v := range code {
			codes[row*spectrumDims+d] = halfToFloat32(float32ToHalf(float32(v)))
		}
	}
	want := cepstrumFromCodes(codes, frames)
	got, n, err := ReferenceCepstrum(spectrum, rate, fft, worldFrameMS)
	if err != nil || n != frames || !slices.Equal(got, want) {
		t.Fatalf("reference and index differ: frames=%d err=%v", n, err)
	}
	for row := range frames * subFrames {
		codes[row*spectrumDims] += 1000
	}
	if !slices.Equal(want, cepstrumFromCodes(codes, frames)) {
		t.Fatal("C0 affected search features")
	}
	for frame := range frames {
		var norm float64
		for _, v := range got[frame*CepstrumDims : (frame+1)*CepstrumDims] {
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-6 {
			t.Fatalf("norm=%g", norm)
		}
	}
	for _, ms := range []float64{0, 10} {
		if _, _, err := ReferenceCepstrum(spectrum, rate, fft, ms); err == nil {
			t.Fatal("invalid period accepted")
		}
	}
}

func cepstrumTestIndex() *Index {
	const frames = 8
	index := &Index{
		Header:   Header{Version: indexVersion, Model: "test", SampleRate: 16000, FFTSize: 1024, APBands: 1, Dims: 2, Frames: frames},
		Features: make([]float32, frames*2), Spectrum: make([]float32, frames*subFrames*spectrumDims),
		Aperiodicity: make([]float32, frames*subFrames), F0: make([]float32, frames*subFrames), File: make([]int32, frames), Position: make([]int32, frames),
	}
	for i := range frames {
		index.Features[i*2] = 1
		index.File[i] = int32(i)
		for sub := range subFrames {
			index.Spectrum[(i*subFrames+sub)*spectrumDims+1+i] = 1
		}
	}
	index.Cepstrum = cepstrumFromCodes(index.Spectrum, frames)
	return index
}

func TestCepstrumModeAndWavLMDefault(t *testing.T) {
	index := cepstrumTestIndex()
	query := Request{Features: []float32{1, 0}, Frames: 1, ReferenceF0: make([]float64, subFrames), SelectStats: &SelectStats{}}
	baseline, err := index.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	query.FeatureMode = FeatureWavLM
	explicit, err := index.Parameters(query)
	if err != nil || !reflect.DeepEqual(baseline, explicit) {
		t.Fatal("wavlm default changed", err)
	}
	query.FeatureMode = FeatureCepstrum
	query.Cepstrum = slices.Clone(index.Cepstrum[7*CepstrumDims : 8*CepstrumDims])
	other, err := index.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(baseline, other) {
		t.Fatal("cepstrum did not affect selection")
	}
	query.FeatureMode = "hybrid"
	if _, err := index.Parameters(query); err == nil {
		t.Fatal("unknown mode accepted")
	}
	query.FeatureMode = FeatureCepstrum
	query.Cepstrum = nil
	if _, err := index.Parameters(query); err == nil {
		t.Fatal("missing cepstrum accepted")
	}
}

func TestCepstrumIndexVersions(t *testing.T) {
	index := cepstrumTestIndex()
	path := filepath.Join(t.TempDir(), "v7.idx")
	if err := index.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(index.Cepstrum, loaded.Cepstrum) {
		t.Fatal("cepstrum did not round trip")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data[:len(data)-1], 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadIndex(path); err == nil {
		t.Fatal("truncated cepstrum accepted")
	}
	index.Version = 6
	legacy := filepath.Join(t.TempDir(), "v6.idx")
	if err := index.Save(legacy); err != nil {
		t.Fatal(err)
	}
	old, err := LoadIndex(legacy)
	if err != nil {
		t.Fatal(err)
	}
	if len(old.Cepstrum) != 0 {
		t.Fatal("legacy index unexpectedly has cepstrum")
	}
	query := Request{Features: []float32{1, 0}, Frames: 1, ReferenceF0: make([]float64, subFrames)}
	a, err := old.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	b, err := loaded.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("legacy wavlm results changed")
	}
	query.FeatureMode = FeatureCepstrum
	query.Cepstrum = slices.Clone(index.Cepstrum[7*CepstrumDims:])
	a, err = old.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	b, err = loaded.Parameters(query)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("legacy cepstrum fallback differs")
	}
}
