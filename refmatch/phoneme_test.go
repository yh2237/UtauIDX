package refmatch

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func phonemeFixture() (*Index, Request) {
	index := &Index{Header: Header{Version: Version(), Frames: 9, Dims: 3, SampleRate: 16000, FFTSize: 512, APBands: 1, F0Median: 200, Files: []string{"a.wav", "ka.wav", "u.wav"}}, Features: make([]float32, 27), Spectrum: make([]float32, 9*4*60), Aperiodicity: make([]float32, 36), F0: make([]float32, 36), File: make([]int32, 9)}
	for j := range 9 {
		index.File[j] = int32(j / 3)
		for d := range 3 {
			index.Features[j*3+d] = float32(math.Sin(float64(j*3 + d + 1)))
		}
		for sub := range 4 {
			index.F0[j*4+sub] = float32(180 + j*3)
			index.Aperiodicity[j*4+sub] = -20
			for d := range 60 {
				index.Spectrum[(j*4+sub)*60+d] = float32(math.Sin(float64(j+d+sub)) / float64(d+1))
			}
		}
	}
	request := Request{Frames: 5, Features: append([]float32(nil), index.Features[:15]...), ReferenceF0: make([]float64, 20), PitchScale: 1}
	for j := range request.ReferenceF0 {
		request.ReferenceF0[j] = 200
	}
	return index, request
}

func TestPhonemeCandidatesAndSynthesis(t *testing.T) {
	index, request := phonemeFixture()
	index.Phonemes = []string{"a", "ka", "u"}
	index.FilePhonemes = []int32{0, 1, 2}
	targets := []int32{1, 2, 0, -1, 99}
	baseline := index.nearest(request.Features, request.Frames, frameVoicing(request.ReferenceF0))
	for _, bonus := range []float64{0, 100} {
		got := index.nearestWithPhonemes(request.Features, request.Frames, frameVoicing(request.ReferenceF0), bonus, targets, false)
		for frame, target := range targets[:3] {
			if len(got[frame]) != 3 {
				t.Fatalf("frame %d candidates = %v", frame, got[frame])
			}
			for _, j := range got[frame] {
				if index.FilePhonemes[index.File[j]] != target {
					t.Fatalf("frame %d chose wrong phoneme: %d", frame, j)
				}
			}
		}
		if bonus == 0 && (!reflect.DeepEqual(got[3], baseline[3]) || !reflect.DeepEqual(got[4], baseline[4])) {
			t.Fatal("unspecified or absent phoneme did not fall back")
		}
	}
	request.TargetPhonemes = []int32{1, 1, 1, 1, 1}
	// A single-file index provides the same candidates as a phoneme-restricted index.
	expected := index.WithFeatures(index.Features[9:18])
	expected.Frames = 3
	expected.Spectrum = index.Spectrum[3*4*60 : 6*4*60]
	expected.Aperiodicity = index.Aperiodicity[12:24]
	expected.F0 = index.F0[12:24]
	expected.File = []int32{1, 1, 1}
	actual, err := index.Synthesize(parameterWorld{}, request)
	if err != nil {
		t.Fatal(err)
	}
	request.TargetPhonemes = nil
	want, err := expected.Synthesize(parameterWorld{}, request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actual, want) {
		t.Fatal("synthesis used units outside target phoneme")
	}
	request.TargetPhonemes = []int32{0}
	if _, err := index.Parameters(request); err == nil {
		t.Fatal("accepted wrong target length")
	}
	request.TargetPhonemes = []int32{-2, -1, -1, -1, -1}
	if _, err := index.Parameters(request); err == nil {
		t.Fatal("accepted invalid target ID")
	}
}

type parameterWorld struct{ fakeWorld }

func (parameterWorld) Synthesize(f0, spectrum, ap []float64, _ int, _ int, _ float64) ([]float64, error) {
	return append(append(append([]float64(nil), f0...), spectrum...), ap...), nil
}

func TestPhonemePersistenceAndLegacy(t *testing.T) {
	for _, version := range []int{6, 7, 8} {
		index, request := phonemeFixture()
		index.Version = version
		if version == 8 {
			index.Phonemes = []string{"a", "ka"}
			index.FilePhonemes = []int32{0, 1, -1}
		}
		path := filepath.Join(t.TempDir(), "index.idx")
		if err := index.Save(path); err != nil {
			t.Fatal(err)
		}
		loaded, err := LoadIndex(path)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(index.Header, loaded.Header) {
			t.Fatalf("version %d metadata changed", version)
		}
		copied := loaded.WithFeatures(loaded.Features)
		if !reflect.DeepEqual(copied.Phonemes, loaded.Phonemes) || !reflect.DeepEqual(copied.FilePhonemes, loaded.FilePhonemes) {
			t.Fatal("WithFeatures lost phonemes")
		}
		if version < 8 {
			baseline, err := loaded.Synthesize(parameterWorld{}, request)
			if err != nil {
				t.Fatal(err)
			}
			request.TargetPhonemes = []int32{0, 1, 0, 1, 0}
			got, err := loaded.Synthesize(parameterWorld{}, request)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, baseline) {
				t.Fatalf("legacy v%d did not ignore targets", version)
			}
		}
	}
}

// These hashes were captured before adding phoneme filtering.
func TestUnspecifiedPhonemeBaselineBytes(t *testing.T) {
	index, request := phonemeFixture()
	index.Phonemes = []string{"a", "ka", "u"}
	index.FilePhonemes = []int32{0, 1, 2}
	for _, bonus := range []float64{0, 0.15} {
		request.ContinuityBonus = bonus
		p, err := index.Parameters(request)
		if err != nil {
			t.Fatal(err)
		}
		var b bytes.Buffer
		for _, data := range [][]float64{p.F0, p.Spectrum, p.Aperiodicity} {
			if err := binary.Write(&b, binary.LittleEndian, data); err != nil {
				t.Fatal(err)
			}
		}
		want := "073e9e517aae452172e9fc69acbef18438fd2efd5ff2a89307629130a2f9aef1"
		if bonus != 0 {
			want = "d727f519198cdf7d33191fe9cb2c7a87ef3bcaeeb172ac377c0a75a375ded4a4"
		}
		if got := fmt.Sprintf("%x", sha256.Sum256(b.Bytes())); got != want {
			t.Fatalf("bonus %g baseline bytes changed: %s", bonus, got)
		}
	}
}
