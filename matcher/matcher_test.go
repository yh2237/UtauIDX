package matcher

import (
	"math"
	"slices"
	"testing"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
	"github.com/yh2237/gograd/tensor"
)

const wavlm = encoder.IDWavLMBasePlusL6

func testIndex(frames int, ids ...string) *index.Index {
	idx := &index.Index{
		Header: index.Header{Version: index.Version, Frames: frames, SampleRate: 44100, FFTSize: 128,
			Files: []index.FileEntry{{Layer: speechpool.LayerRaw}, {Layer: speechpool.LayerSpeech}}},
		Channels:      make(map[string][]float32),
		Spectrum:      make([]float32, frames*index.SubFrames*index.SpectrumDims),
		F0:            make([]float32, frames*index.SubFrames),
		FileID:        make([]int32, frames),
		Position:      make([]int32, frames),
		GlitchPenalty: make([]float32, frames),
	}
	for j := range frames {
		idx.Position[j] = int32(j)
	}
	for _, id := range ids {
		idx.Header.Channels = append(idx.Header.Channels, index.ChannelMeta{ID: id, Dims: 2})
		idx.Channels[id] = make([]float32, frames*2)
	}
	return idx
}

func checkNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tolerance {
		t.Fatalf("got %.12g, want %.12g", got, want)
	}
}

func TestComputeAutoProfile(t *testing.T) {
	es, wh := encoder.IDWav2Vec2EspeakL20, encoder.IDWhisperSmallL12PosInv
	for _, tc := range []struct {
		name                        string
		frames                      int
		wavCos, whCos               float32
		all, sinusoidal             bool
		gap, xling, context, glitch float64
		want                        map[string]float64
	}{
		{"large bank with Whisper", 10000, 0.55, 0.53, true, false, 0.0952380952, 0.5, 0.030952381, 0.727891156,
			map[string]float64{wavlm: 0.713114800, es: 0.031382453, wh: 0.249633150, ChannelMouthStatic: 0.004402198, ChannelMouthDelta: 0.001467399}},
		{"small sinusoidal bank", 2, 0.34, 0.10, true, true, 1, 0, 0, 0,
			map[string]float64{wavlm: 0.36, es: 0.44, ChannelMouthStatic: 0.15, ChannelMouthDelta: 0.05}},
		{"WavLM and Mouth", 2, 0.47, 0, false, false, 0.726190476, 0, 0, 0,
			map[string]float64{wavlm: 0.876232728, ChannelMouthStatic: 0.092825454, ChannelMouthDelta: 0.030941818}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ids := []string{wavlm}
			if tc.all {
				ids = append(ids, es, wh)
			}
			idx := testIndex(tc.frames, ids...)
			idx.MouthStatic = make([]float32, tc.frames*worldcodec.MouthDims)
			idx.MouthDelta = make([]float32, tc.frames*worldcodec.MouthDims)
			if tc.sinusoidal {
				idx.C35 = 1.26
			}
			q := Query{Frames: 1, Channels: make(map[string][]float32),
				MouthStatic: make([]float32, worldcodec.MouthDims), MouthDelta: make([]float32, worldcodec.MouthDims)}
			for _, id := range ids {
				c := tc.wavCos
				if id == wh {
					c = tc.whCos
				}
				for f := range tc.frames {
					idx.Channels[id][f*2] = c
					idx.Channels[id][f*2+1] = float32(math.Sqrt(1 - float64(c)*float64(c)))
				}
				q.Channels[id] = []float32{2, 0}
			}
			p := ComputeAutoProfile(idx, q)
			checkNear(t, p.Cos1W6, float64(tc.wavCos), 1e-7)
			checkNear(t, p.Cos1WH, float64(tc.whCos), 1e-7)
			checkNear(t, p.Gap, tc.gap, 1e-6)
			checkNear(t, p.Xling, tc.xling, 2e-6)
			checkNear(t, p.ContextWeight, tc.context, 1e-6)
			checkNear(t, p.GlitchScale, tc.glitch, 1e-6)
			if p.IsSinusoidal != tc.sinusoidal || p.ContinuityBonus != 0.06 || len(p.ChannelWeights) != len(tc.want) {
				t.Fatalf("unexpected profile %+v", p)
			}
			for id, w := range tc.want {
				checkNear(t, p.ChannelWeights[id], w, 2e-6)
			}
		})
	}
}

func TestComputeCos1AndFallback(t *testing.T) {
	bank := []float32{1, 0, 0, 1}
	packed := tensor.PackB(bank, 2, 2, true)
	q := make([]float32, 65*2)
	for i := range 64 {
		q[i*2] = 5
	}
	q[128], q[129] = -3, -4
	checkNear(t, ComputeCos1(packed, 2, 2, q, 65), (64-0.6)/65, 1e-8)
	idx := testIndex(2, "z", "a")
	copy(idx.Channels["z"], bank)
	copy(idx.Channels["a"], []float32{0, 1, 0, -1})
	p := ComputeAutoProfile(idx, Query{Frames: 1, Channels: map[string][]float32{"z": {1, 0}, "a": {1, 0}}})
	if len(p.ChannelWeights) != 1 || p.ChannelWeights["a"] != 1 || p.Cos1W6 != 0 {
		t.Fatalf("unexpected fallback profile %+v", p)
	}
}

func TestWeightedSGEMMMatchesFused(t *testing.T) {
	ids := []string{"a", "b", "c", "d"}
	idx := testIndex(4, ids...)
	axes := [][]float32{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
	weights := make(map[string]float64)
	q := Query{Frames: 3, Channels: make(map[string][]float32)}
	fusedQ := make([]float32, q.Frames*8)
	fusedBank := make([]float32, idx.Frames*8)
	for c, id := range ids {
		weights[id] = 0.25
		for j := range idx.Frames {
			axis := axes[(j+c*c)%len(axes)]
			copy(idx.Channels[id][j*2:], axis)
			for d, v := range axis {
				fusedBank[j*8+c*2+d] = 0.5 * v
			}
		}
		for f := range q.Frames {
			axis := axes[(f+c)%len(axes)]
			q.Channels[id] = append(q.Channels[id], 2*axis[0], 2*axis[1])
			for d, v := range axis {
				fusedQ[f*8+c*2+d] = 0.5 * v
			}
		}
	}
	want := make([]float32, q.Frames*idx.Frames)
	tensor.SGEMMPacked(want, fusedQ, tensor.PackB(fusedBank, idx.Frames, 8, true), q.Frames, 1)
	got, err := Match(idx, q, Options{Profile: &AutoProfile{}, ChannelWeights: weights, TopK: 20})
	if err != nil {
		t.Fatal(err)
	}
	for f, chosen := range got.Nearest {
		if len(chosen) != DefaultTopK {
			t.Fatalf("got %d candidates", len(chosen))
		}
		for k, j := range chosen {
			if got.Scores[f][k] != want[f*idx.Frames+j] {
				t.Fatalf("frame %d candidate %d score %v, want %v", f, j, got.Scores[f][k], want[f*idx.Frames+j])
			}
		}
	}
}

func TestMatchingPenalties(t *testing.T) {
	for _, tc := range []struct {
		name  string
		apply func(*index.Index, *Query, *Options)
		score float64
	}{
		{"voicing", func(_ *index.Index, q *Query, _ *Options) { q.Voiced = []bool{true} }, 0.5},
		{"file", func(i *index.Index, _ *Query, _ *Options) { i.Files[0].Penalty = 0.2 }, 0.8},
		{"extra file", func(_ *index.Index, _ *Query, o *Options) { o.ExtraFilePenalties = []float32{0.3} }, 0.7},
		{"layer", func(_ *index.Index, _ *Query, o *Options) {
			o.LayerPenalties = map[speechpool.Layer]float32{speechpool.LayerRaw: 0.4}
		}, 0.6},
		{"glitch", func(i *index.Index, _ *Query, o *Options) { i.GlitchPenalty[0] = 0.6; o.Profile.GlitchScale = 0.5 }, 0.7},
		{"frame", func(_ *index.Index, _ *Query, o *Options) { o.FramePenalties = []float32{0.25} }, 0.75},
	} {
		t.Run(tc.name, func(t *testing.T) {
			idx := testIndex(2, wavlm)
			copy(idx.Channels[wavlm], []float32{1, 0, 1, 0})
			idx.FileID[1] = 1
			for sub := range index.SubFrames {
				idx.F0[index.SubFrames+sub] = 200
			}
			q := Query{Frames: 1, Channels: map[string][]float32{wavlm: {1, 0}}}
			opts := Options{Profile: &AutoProfile{ChannelWeights: map[string]float64{wavlm: 1}}}
			tc.apply(idx, &q, &opts)
			got, err := Match(idx, q, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(got.Nearest[0], []int{1, 0}) {
				t.Fatalf("unexpected candidates %v", got.Nearest[0])
			}
			checkNear(t, float64(got.Scores[0][1]), tc.score, 1e-7)
		})
	}
}

func TestContinuityContextAndStats(t *testing.T) {
	idx := testIndex(3, wavlm)
	copy(idx.Channels[wavlm], []float32{1, 0, 1, 0, 0, 1})
	idx.FileID = []int32{0, 1, 1}
	idx.Position = []int32{0, 0, 1}
	q := Query{Frames: 3, Channels: map[string][]float32{wavlm: {1, 0, 0, 1, 1, 0}}}
	opts := Options{Profile: &AutoProfile{ChannelWeights: map[string]float64{wavlm: 1}, ContextWeight: 0.05, ContinuityBonus: 0.06}, TopK: 1}
	got, err := Match(idx, q, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.Stats.SelectedFiles, []int32{0, 1, 0}) {
		t.Fatalf("unexpected reference tie behavior: %+v candidates %v", got.Stats, got.Nearest)
	}

	// A weaker isolated candidate loses to its file-contiguous competitor.
	copy(idx.Channels[wavlm], []float32{0.99, float32(math.Sqrt(1 - 0.99*0.99)), 1, 0, 0, 1})
	idx.ResetPacked()
	q.Channels[wavlm][4], q.Channels[wavlm][5] = 0.98, 0.20
	got, err = Match(idx, q, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Nearest[0][0] != 1 || got.Nearest[1][0] != 2 || got.Nearest[2][0] != 0 {
		t.Fatalf("unexpected path %v", got.Nearest)
	}
	checkNear(t, float64(got.Scores[1][0]), 1.06, 1e-6)
	s := got.Stats
	if s.Frames != 12 || s.Runs != 2 || s.Switches != 1 || s.UniqueUnits != 2 || s.MeanRunFrames != 6 ||
		!slices.Equal(s.SelectedFiles, []int32{1, 1, 0}) || s.LayerFrames[speechpool.LayerSpeech] != 2 || s.LayerFrames[speechpool.LayerRaw] != 1 {
		t.Fatalf("unexpected stats %+v", s)
	}

	// The previous candidate stays available after the first SGEMM block.
	idx = testIndex(matchBlockRows+2, wavlm)
	q = Query{Frames: idx.Frames, Channels: map[string][]float32{wavlm: make([]float32, idx.Frames*2)}}
	for j := range idx.Frames {
		idx.Channels[wavlm][j*2] = 1
		q.Channels[wavlm][j*2] = 1
	}
	got, err = Match(idx, q, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Nearest[matchBlockRows][0] != matchBlockRows || got.Stats.Runs != 1 {
		t.Fatalf("continuity lost at block boundary: %v", got.Nearest)
	}
}

func TestContextAveragingAndTailPenalty(t *testing.T) {
	idx := testIndex(3, wavlm)
	copy(idx.Channels[wavlm], []float32{1, 0, 1, 0, 0, -1})
	idx.FileID = []int32{0, 1, 1}
	idx.Position = []int32{0, 0, 1}
	q := Query{Frames: 2, Channels: map[string][]float32{wavlm: {1, 0, 0, 1}}}
	opts := Options{Profile: &AutoProfile{ChannelWeights: map[string]float64{wavlm: 1}, ContextWeight: 0.5}}
	got, err := Match(idx, q, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got.Nearest[0], []int{0, 1, 2}) {
		t.Fatalf("unexpected candidates %v", got.Nearest[0])
	}
	checkNear(t, float64(got.Scores[0][1]), 1.0/3, 1e-7)

	idx.FileID[0] = 1
	for sub := range index.SubFrames {
		idx.Spectrum[sub*index.SpectrumDims] = -8
	}
	opts.Profile.ContextWeight = 0
	opts.TailPenalty = 0.1
	q.Level5ms = []float64{0, 0, 0, 0, -30, -30, -30, -30}
	got, err = Match(idx, q, opts)
	if err != nil {
		t.Fatal(err)
	}
	checkNear(t, float64(got.Scores[0][1]), 0.7, 1e-7)
	checkNear(t, float64(got.Scores[1][1]), 0, 1e-7)
}

func TestBlendWeights(t *testing.T) {
	ceps := make([]float32, 2*candidateCepDims)
	ceps[candidateCepDims] = 2
	for _, tc := range []struct {
		opts  Options
		ratio float64
	}{
		{Options{RankDecay: 0.45}, 0.45},
		{Options{RankDecay: 0.45, CandCepSigma: 1}, 0.45 * math.Exp(-2)},
		{Options{RankDecay: 0.45, CandCepSigma: 1, SimTemp: 0.5}, math.Exp(-1)},
	} {
		w := blendWeights([]int{0, 1}, []float32{1, 0.5}, ceps, tc.opts)
		checkNear(t, w[1]/w[0], tc.ratio, 1e-12)
		checkNear(t, w[0]+w[1], 1, 1e-12)
	}
}
