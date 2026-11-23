package index

import (
	"math"
	"reflect"
	"slices"
	"testing"

	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
)

func mergeFixture(layer speechpool.Layer, channels []ChannelMeta, f0, coeff float32, position int32) *Index {
	idx := &Index{
		Header: Header{Version: Version, SampleRate: 16000, FFTSize: 128, APBands: 1, Frames: 2,
			F0Median: 999, C35: 999, Source: "source", CacheKey: "key",
			Channels: channels, Files: []FileEntry{{Path: string(layer) + ".wav", Layer: layer}}},
		Channels:     make(map[string][]float32),
		Spectrum:     make([]float32, 2*SubFrames*SpectrumDims),
		Aperiodicity: make([]float32, 2*SubFrames),
		F0:           make([]float32, 2*SubFrames),
		FileID:       []int32{0, 0}, Position: []int32{position, position + 1},
		GlitchPenalty: []float32{1, 1},
	}
	for _, meta := range channels {
		idx.Channels[meta.ID] = normalizedChannel(testFeatures(idx.Frames, meta.Dims), idx.Frames, meta.Dims)
	}
	for row := range idx.F0 {
		idx.F0[row] = f0
		idx.Aperiodicity[row] = -14
		copy(idx.Spectrum[row*SpectrumDims+3:], []float32{coeff, coeff, coeff})
		idx.Spectrum[row*SpectrumDims+2] = float32(row / SubFrames)
	}
	return idx
}

func TestMergeIndexes(t *testing.T) {
	a := mergeFixture(speechpool.LayerRaw, []ChannelMeta{{"sharedB", 4}, {"onlyA", 6}, {"sharedA", 3}}, 200, 2, 0)
	b := mergeFixture(speechpool.LayerBabble, []ChannelMeta{{"sharedA", 3}, {"sharedB", 4}, {"onlyB", 8}}, 300, 4, 2)
	a.F0[0] = 400
	b.Aperiodicity[0], b.Aperiodicity[1], b.Aperiodicity[2], b.Aperiodicity[3] = 0, 0, 0, 0
	out, err := Merge(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if err := out.Validate(); err != nil {
		t.Fatal(err)
	}
	if out.Frames != 4 || !slices.Equal(out.Header.Channels, []ChannelMeta{{"sharedB", 4}, {"sharedA", 3}}) ||
		len(out.Channels) != 2 || !slices.Equal(out.Files, []FileEntry{a.Files[0], b.Files[0]}) ||
		!slices.Equal(out.FileID, []int32{0, 0, 1, 1}) || !slices.Equal(out.Position, []int32{0, 1, 2, 3}) {
		t.Fatalf("merged header = %+v, file IDs = %v, positions = %v", out.Header, out.FileID, out.Position)
	}
	if out.F0Median != 300 || out.C35 != math.Sqrt(10) || !slices.Equal(out.GlitchPenalty, []float32{0.10, 0, 0.12, 0}) {
		t.Fatalf("F0 median = %v, C35 = %v, penalties = %v", out.F0Median, out.C35, out.GlitchPenalty)
	}
	if !slices.Equal(out.Voiced, []bool{true, true, true, true}) {
		t.Fatalf("voicing = %v", out.Voiced)
	}
	static, delta := worldcodec.ComputeBankMouth(out.Spectrum, out.Frames, out.Voiced, out.FileID, out.Position)
	if !slices.Equal(out.MouthStatic, static) || !slices.Equal(out.MouthDelta, delta) {
		t.Fatal("merged Mouth features do not match recomputed features")
	}
	for _, meta := range out.Header.Channels {
		want := append(slices.Clone(a.Channels[meta.ID]), b.Channels[meta.ID]...)
		if !slices.Equal(out.Channels[meta.ID], want) {
			t.Fatalf("merged channel %q values differ", meta.ID)
		}
	}
	out.Channels["sharedA"][0] = 99
	if a.Channels["sharedA"][0] == 99 {
		t.Fatal("merged values alias an input index")
	}
	one, err := Merge(a)
	if err != nil || !reflect.DeepEqual(one.Channels, a.Channels) || one.Source != a.Source || one.CacheKey != a.CacheKey {
		t.Fatalf("Merge of one index returned %v", err)
	}
}

func TestMergeRejectsIncompatibleIndexes(t *testing.T) {
	a := mergeFixture(speechpool.LayerRaw, []ChannelMeta{{"a", 3}}, 200, 2, 0)
	if _, err := Merge(); err == nil {
		t.Fatal("Merge accepted no inputs")
	}
	if _, err := Merge(a, nil); err == nil {
		t.Fatal("Merge accepted a nil input")
	}
	for _, tc := range []struct {
		name   string
		change func(*Index)
	}{
		{"sample rate", func(b *Index) { b.SampleRate *= 2 }},
		{"FFT size", func(b *Index) { b.FFTSize *= 2 }},
		{"AP bands", func(b *Index) { b.APBands = 0; b.Aperiodicity = nil }},
		{"no shared dimensions", func(b *Index) {
			b.Header.Channels[0].Dims++
			b.Channels["a"] = testFeatures(b.Frames, b.Header.Channels[0].Dims)
		}},
		{"invalid position length", func(b *Index) { b.Position = nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := mergeFixture(speechpool.LayerBabble, []ChannelMeta{{"a", 3}}, 200, 2, 0)
			tc.change(b)
			if _, err := Merge(a, b); err == nil {
				t.Fatal("Merge accepted incompatible inputs")
			}
		})
	}
}
