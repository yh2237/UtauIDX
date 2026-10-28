package refmatch

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yh2237/UtauIDX/audio"
)

func TestOriginalPositions(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.wav")
	if err := audio.WriteWav(p, &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 16000)}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	f := func([]float64, int) ([]float32, int, int, error) { return make([]float32, 50), 50, 1, nil }
	got, err := analyzeBytes(data, fakeWorld{}, f, 16000, 20, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.position) != 49 {
		t.Fatalf("positions=%d, want 49", len(got.position))
	}
	cached, err := decodePart(encodePart(got))
	if err != nil || !reflect.DeepEqual(cached.position, got.position) {
		t.Fatalf("position cache round trip: %v", err)
	}
	for j, p := range got.position {
		if p != int32(j+1) {
			t.Fatalf("position[%d]=%d", j, p)
		}
	}
}

func TestSelectStats(t *testing.T) {
	index := &Index{File: []int32{0, 0, 0, 1, 1, 1}}
	for _, tt := range []struct {
		name   string
		chosen [][]int
		want   SelectStats
	}{
		{"empty", nil, SelectStats{}},
		{"one", [][]int{{0}}, SelectStats{Frames: 4, Runs: 1, UniqueUnits: 1, MeanRunFrames: 4}},
		{"continuous", [][]int{{0}, {1}, {2}}, SelectStats{Frames: 12, Runs: 1, UniqueUnits: 1, MeanRunFrames: 12}},
		{"recording boundary", [][]int{{2}, {3}}, SelectStats{Frames: 8, Runs: 2, Switches: 1, UniqueUnits: 2, MeanRunFrames: 4}},
		{"repeat", [][]int{{1}, {1}}, SelectStats{Frames: 8, Runs: 2, Switches: 1, UniqueUnits: 1, MeanRunFrames: 4}},
		{"skip", [][]int{{0}, {2}}, SelectStats{Frames: 8, Runs: 2, Switches: 1, UniqueUnits: 1, MeanRunFrames: 4}},
		{"reverse", [][]int{{2}, {1}}, SelectStats{Frames: 8, Runs: 2, Switches: 1, UniqueUnits: 1, MeanRunFrames: 4}},
		{"ignore lower ranks", [][]int{{0, 4}, {1, 3}}, SelectStats{Frames: 8, Runs: 1, UniqueUnits: 1, MeanRunFrames: 8}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := index.selectStats(tt.chosen); got != tt.want {
				t.Fatalf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestSelectStatsCountsRemovedQuietFramesAsGap(t *testing.T) {
	index := &Index{File: []int32{0, 0, 0}, Position: []int32{10, 11, 15}}
	got := index.selectStats([][]int{{0}, {1}, {2}})
	want := SelectStats{Frames: 12, Runs: 2, Switches: 1, UniqueUnits: 1, MeanRunFrames: 6}
	if got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
}

func TestSynthesizeSelectStatsDoesNotChangeParameters(t *testing.T) {
	index := &Index{
		Header:   Header{Frames: 1, Dims: 1, SampleRate: 16000, FFTSize: 1024, APBands: 1},
		Features: []float32{1}, File: []int32{0}, Position: []int32{10},
		F0: make([]float32, subFrames), Spectrum: make([]float32, subFrames*spectrumDims),
		Aperiodicity: make([]float32, subFrames),
	}
	request := Request{Features: []float32{1, 1}, Frames: 2, ReferenceF0: make([]float64, 2*subFrames)}
	plain, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	var stats SelectStats
	request.SelectStats = &stats
	observed, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plain, observed) {
		t.Fatal("statistics changed parameters")
	}
	want := SelectStats{Frames: 8, Runs: 2, Switches: 1, UniqueUnits: 1, MeanRunFrames: 4}
	if stats != want {
		t.Fatalf("got %+v, want %+v", stats, want)
	}
	stats = SelectStats{}
	if _, err := index.Synthesize(fakeWorld{}, request); err != nil {
		t.Fatal(err)
	}
	if stats != want {
		t.Fatalf("Synthesize got %+v, want %+v", stats, want)
	}
	clone := index.WithFeatures([]float32{1})
	if !reflect.DeepEqual(clone.Position, index.Position) || !reflect.DeepEqual(clone.File, index.File) || clone.packed != nil {
		t.Fatal("clone lost positions or inherited search cache")
	}
	var reported SelectStats
	stats = SelectStats{}
	_, err = clone.SynthesizeWithStats(fakeWorld{}, request, func(frames, runs, switches, units int, mean float64) {
		reported = SelectStats{Frames: frames, Runs: runs, Switches: switches, UniqueUnits: units, MeanRunFrames: mean}
	})
	if err != nil || reported != want || stats != want {
		t.Fatalf("callback got %+v, output %+v, error %v", reported, stats, err)
	}
}

func TestSelectStatsResetOnInvalidRequest(t *testing.T) {
	stats := SelectStats{Frames: 100, Runs: 2}
	index := &Index{}
	if _, err := index.Parameters(Request{SelectStats: &stats}); err == nil {
		t.Fatal("expected invalid request error")
	}
	if stats != (SelectStats{}) {
		t.Fatalf("stale stats: %+v", stats)
	}
}
