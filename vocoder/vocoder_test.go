package vocoder

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/matcher"
)

type mockWorld struct {
	analyze func([]float64, int, float64, float64) ([]float64, []float64, []float64, int, error)
	synth   func([]float64, []float64, []float64, int, int, float64) ([]float64, error)
}

func (w mockWorld) Analyze(x []float64, rate int, hop, threshold float64) ([]float64, []float64, []float64, int, error) {
	return w.analyze(x, rate, hop, threshold)
}

func (w mockWorld) Synthesize(f0, sp, ap []float64, fftSize, rate int, hop float64) ([]float64, error) {
	return w.synth(f0, sp, ap, fftSize, rate, hop)
}

func near(t *testing.T, got, want float64) {
	t.Helper()
	if !finite(got) || math.Abs(got-want) > 1e-10*max(1, math.Abs(want)) {
		t.Fatalf("got %.12g, want %.12g", got, want)
	}
}

func TestAutoSynthConfig(t *testing.T) {
	for _, tc := range []struct {
		name string
		gap  float64
		sin  bool
		want SynthConfig
	}{
		{"low gap human", 0.1, false, SynthConfig{LevelTrackDB: 9.55, MaxSupportDB: 4.2, CrossfadeMS: 22,
			UnvoicedCFMS: 12, PostfilterBeta: 0.408, BetaMode: BetaLowOrder, HighOrderSmooth: 1, ModBoost: 0.35, PresenceDB: 0.62, QuietLevelDB: 40}},
		{"SafeRefAP threshold", 0.25, false, SynthConfig{UseSafeRefAP: true, LevelTrackDB: 9.5, MaxSupportDB: 5.75, CrossfadeMS: 22,
			UnvoicedCFMS: 12, PostfilterBeta: 0.42, BetaMode: BetaLowOrder, HighOrderSmooth: 1, ModBoost: 0.35, PresenceDB: 0.8, QuietLevelDB: 40}},
		{"high gap human", 1, false, SynthConfig{UseSafeRefAP: true, BodySupportDB: 5.5, LevelTrackDB: 8, MaxSupportDB: 11,
			CrossfadeMS: 22, UnvoicedCFMS: 12, PostfilterBeta: 0.48, BetaMode: BetaLowOrder, HighOrderSmooth: 1, ModBoost: 0.35,
			NoSmooth: true, PresenceDB: 1.7, QuietLevelDB: 40}},
		{"low gap sinusoidal", 0.2, true, SynthConfig{APCliffGuard: true, LevelTrackDB: 9.1, MaxSupportDB: 5.5,
			CrossfadeMS: 18, PostfilterBeta: 0.4, BetaMode: BetaMidOrder, HighOrderSmooth: 1, ModBoost: 0.25, SpectrumSmooth: 1, PresenceDB: 2.5, QuietLevelDB: 40}},
		{"high gap sinusoidal", 1, true, SynthConfig{UseSafeRefAP: true, APCliffGuard: true, BodySupportDB: 5.5, LevelTrackDB: 8,
			MaxSupportDB: 11, CrossfadeMS: 18, PostfilterBeta: 0.4, BetaMode: BetaMidOrder, HighOrderSmooth: 1, ModBoost: 0.25,
			SpectrumSmooth: 1, PresenceDB: 2.5, QuietLevelDB: 40}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := AutoSynthConfig(matcher.AutoProfile{Gap: tc.gap, IsSinusoidal: tc.sin})
			want := tc.want
			near(t, got.LevelTrackDB, want.LevelTrackDB)
			near(t, got.PostfilterBeta, want.PostfilterBeta)
			near(t, got.PresenceDB, want.PresenceDB)
			got.LevelTrackDB, got.PostfilterBeta, got.PresenceDB = want.LevelTrackDB, want.PostfilterBeta, want.PresenceDB
			if got != want {
				t.Fatalf("got %+v, want %+v", got, want)
			}
		})
	}
}

func TestAnalyzeReference(t *testing.T) {
	const fft, frames, rate = 512, 3, 22050
	const bins, rows = fft/2 + 1, frames * index.SubFrames
	samples := make([]float64, rate/10)
	for i := range samples {
		samples[i] = 0.25
	}
	w := mockWorld{analyze: func(x []float64, rate int, hop, threshold float64) ([]float64, []float64, []float64, int, error) {
		if rate != 44100 || hop != 5 || threshold != 0.85 || len(x) != 2*len(samples) {
			t.Fatalf("unexpected analysis arguments: %d samples at %d Hz, hop %v, threshold %v", len(x), rate, hop, threshold)
		}
		f0 := make([]float64, rows)
		sp, ap := make([]float64, rows*bins), make([]float64, rows*bins)
		for row := range rows {
			f0[row] = 220
			for b := range bins {
				hz := float64(b) * float64(rate) / fft
				power, noise := 0.01, 0.2
				if (row < 8 || row == 10) && hz >= 2800 && hz < 10000 {
					power = 1
				}
				if row >= 8 && row != 10 && hz >= 100 && hz < 900 {
					power = 1
				}
				if (row >= 4 && row < 8) || row == 10 {
					noise = 0.8
				}
				sp[row*bins+b], ap[row*bins+b] = power, noise
			}
		}
		return f0, sp, ap, fft, nil
	}}
	got, err := AnalyzeReference(w, samples, rate, frames, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	wantAP := []float32{0.10, 0.10, 0.10, 0.10, 0.90, 0.90, 0.90, 0.90, 0.10, 0.10, 0.45, 0.10}
	if !slices.Equal(got.SafeRefAP, wantAP) {
		t.Fatalf("got AP %v, want %v", got.SafeRefAP, wantAP)
	}
	if !slices.Equal(got.Voiced20ms(true), []bool{true, false, true}) ||
		!slices.Equal(got.Voiced20ms(false), []bool{true, true, true}) ||
		!slices.Equal(got.IsUV20, []bool{true, true, false}) {
		t.Fatalf("unexpected voicing: gated %v, raw %v, spectral %v", got.Voiced20ms(true), got.Voiced20ms(false), got.IsUV20)
	}
	if got.Frames != frames || got.FFTSize != fft || got.SampleRate != 44100 ||
		len(got.MouthStatic) != frames*12 || len(got.MouthDelta) != frames*12 || len(got.Levels) != rows {
		t.Fatal("unexpected analysis shapes")
	}
	near(t, got.Levels[0], 10*math.Log10(2.0/3.0))
	near(t, got.Levels[5], 0)
	for _, feature := range [][]float32{got.MouthStatic, got.MouthDelta} {
		var norm float64
		for _, v := range feature[12:24] {
			norm += float64(v) * float64(v)
		}
		if math.Abs(norm-1) > 1e-6 {
			t.Fatalf("Mouth row squared norm %.7f, want one", norm)
		}
	}
	got, err = AnalyzeReference(w, samples, rate, frames, []float32{330, 330}, 12.5)
	if err != nil {
		t.Fatal(err)
	}
	for _, hz := range got.F0 {
		near(t, hz, 330)
	}
}

func TestPitchHelpers(t *testing.T) {
	f0 := []float64{0, 100, 0, 0, 160, 0, 0, 0}
	FillVoicedByLevel(f0, []float64{0, 0, -20, -40, 0, -20, -40, -20}, -35)
	if !slices.Equal(f0, []float64{0, 100, 120, 0, 160, 160, 0, 0}) {
		t.Fatalf("filled F0 %v", f0)
	}
	out := OutputF0([]float64{100, 200, 300, 0}, 250, 0, 2)
	for i, want := range []float64{62.5, 250, 562.5, 0} {
		near(t, out[i], want)
	}
	interp := ReferenceF0([]float32{0, 200, 200, 0}, 10, 2)
	for row, want := range []float64{0, 0, 200, 200, 200, 0, 0, 0} {
		near(t, interp[row], want)
	}
}

func TestGuardVoicedAPCliff(t *testing.T) {
	ap := []float64{-0.2, -0.4, -0.2, -0.4, -0.2, -0.4, -1, -1}
	guardVoicedAPCliff(ap, 2, []float64{220, 220, 0, 220}, []float32{0.30, 0.31, 0.10, 0.10})
	for i, want := range []float64{-1.5, -1.7, -0.2, -0.4, -0.2, -0.4, -1, -1} {
		near(t, ap[i], want)
	}
}

func TestPostfilterModes(t *testing.T) {
	for _, mode := range []BetaMode{BetaFull, BetaLowOrder, BetaMidOrder} {
		c := make([]float64, 60)
		for k := range c {
			c[k] = 1
		}
		applyPostfilter(c, 0.4, mode)
		near(t, c[0], 1)
		near(t, c[1], 1)
		near(t, c[6], 1.4)
		switch mode {
		case BetaFull:
			near(t, c[24], 1.4)
		case BetaLowOrder:
			near(t, c[2], 1.4)
			near(t, c[18], 1.4)
			near(t, c[21], 1.2)
			near(t, c[24], 1)
		case BetaMidOrder:
			near(t, c[2], 1.02)
			near(t, c[5], 1.08)
			near(t, c[16], 1.4)
			near(t, c[20], 1.2)
			near(t, c[24], 1)
		}
	}
}

func TestHighOrderSmoothingAndModulation(t *testing.T) {
	const dims = 60
	spectrum := make([]float64, 4*dims)
	for r, v := range []float64{4, 8, 100, 16} {
		spectrum[r*dims+23], spectrum[r*dims+24] = v, v
	}
	smoothHighOrderVoiced(spectrum, dims, []float64{200, 200, 0, 200}, 24, 1)
	for r, want := range []float64{5, 20.0 / 3, 100, 16} {
		near(t, spectrum[r*dims+24], want)
	}
	near(t, spectrum[dims+23], 8)

	const rows = 17
	spectrum = make([]float64, rows*dims)
	f0, level := make([]float64, rows), make([]float64, rows)
	for r := range rows {
		f0[r] = 220
	}
	f0[2] = 0
	level[3] = -40
	spectrum[2*dims+7], spectrum[3*dims+7] = 100, -100
	spectrum[8*dims+7], spectrum[8*dims+18] = 4, 4
	enhanceModulation(spectrum, dims, f0, level, 10)
	near(t, spectrum[2*dims+7], 100)
	near(t, spectrum[3*dims+7], -100)
	near(t, spectrum[8*dims+18], 4)
	near(t, spectrum[8*dims+7], 4+0.40*math.Sqrt(16.0/15-math.Pow(4.0/15, 2)))
}

func TestBodySupport(t *testing.T) {
	const bins, rate, rows = 65, 16000, 16
	sp := make([]float64, bins*rows)
	f0, levels := make([]float64, rows), make([]float64, rows)
	for r := range rows {
		f0[r] = 200
		for b := range bins {
			sp[r*bins+b] = 1
			if r >= 4 && r <= 11 {
				sp[r*bins+b] = 0.01
			}
		}
	}
	applyVoicedBodySupport(sp, bins, rate, f0, levels, 4)
	low := 10 * math.Log10(sp[8*bins+2]/0.01)
	high := 10 * math.Log10(sp[8*bins+48]/0.01)
	if low < 3.8 || low > 4 || high > 0.02 {
		t.Fatalf("body gain %.6f dB at 250 Hz and %.6f dB at 6000 Hz", low, high)
	}
}

func vocoderIndex() *index.Index {
	const id = encoder.IDWavLMBasePlusL6
	return &index.Index{
		Header: index.Header{Version: index.Version, Frames: 2, SampleRate: 16000, FFTSize: 128, APBands: 1, F0Median: 250,
			Channels: []index.ChannelMeta{{ID: id, Dims: 2}}, Files: []index.FileEntry{{Path: "a"}, {Path: "b"}}},
		Channels:     map[string][]float32{id: {1, 0, 0, 1}},
		Spectrum:     make([]float32, 2*index.SubFrames*index.SpectrumDims),
		Aperiodicity: []float32{-0.2, -0.2, -0.2, -0.2, -5, -5, -5, -5},
		F0:           []float32{200, 200, 200, 200, 200, 200, 200, 200},
		FileID:       []int32{0, 1}, Position: []int32{0, 0}, GlitchPenalty: []float32{0, 0},
	}
}

func TestCrossfadeUsesWeightsAndContinuations(t *testing.T) {
	idx := vocoderIndex()
	match := &matcher.Result{Frames: 2, Nearest: [][]int{{0, 1}, {1}}, Weights: [][]float64{{0.75, 0.25}, {1}}}
	values := []float32{1, 2, 3, 4, 10, 20, 30, 40}
	f0 := []float64{200, 200, 200, 200, 200, 200, 200, 0}
	bankVoiced := []bool{true, true}
	cfg := SynthConfig{CrossfadeMS: 22, UnvoicedCFMS: 12}
	got := averageParameters(idx, match, values, 1, f0, bankVoiced, cfg, false)
	current, previous := 1-7.5/22.0, 1-12.5/22.0
	near(t, got[4], (current*10+previous*(0.75*4+0.25*40))/(current+previous))
	near(t, got[7], 40)
	idx.FileID[1], idx.Position[1] = 0, 1
	got = averageParameters(idx, match, values, 1, f0, bankVoiced, cfg, false)
	near(t, got[4], (current*10+previous*(0.75*10+0.25*40))/(current+previous))
	idx.Position[1] = 2
	got = averageParameters(idx, match, values, 1, f0, bankVoiced, cfg, true)
	near(t, got[4], (current*10+previous*4)/(current+previous))
}

func TestWorldVocoderSynthesize(t *testing.T) {
	idx := vocoderIndex()
	const id = encoder.IDWavLMBasePlusL6
	match, err := matcher.Match(idx, matcher.Query{Frames: 1, Channels: map[string][]float32{id: {1, 0}}},
		matcher.Options{TopK: 1})
	if err != nil {
		t.Fatal(err)
	}
	bins := idx.FFTSize/2 + 1
	cfg := SynthConfig{UseSafeRefAP: true, APCliffGuard: true, NoSmooth: true, PresenceDB: 3, PostfilterBeta: -1}
	req := SynthRequest{ReferenceF0: []float64{210, 0, 200, 260}, OutputF0: []float64{300, 310, 320, 330},
		ReferenceLevel: []float64{0, 0, 0, 0}, ReferenceAP: []float32{0.1, 0.1, 0.9, 0.45}, Config: &cfg}
	v := &WorldVocoder{World: mockWorld{synth: func(f0, sp, ap []float64, fft, rate int, hop float64) ([]float64, error) {
		if !slices.Equal(f0, []float64{300, 0, 0, 330}) || fft != idx.FFTSize || rate != idx.SampleRate || hop != 5 ||
			len(sp) != 4*bins || len(ap) != 4*bins {
			t.Fatalf("unexpected WORLD synthesis arguments: F0 %v, FFT %d, rate %d, hop %v", f0, fft, rate, hop)
		}
		near(t, sum(sp[:bins]), float64(bins))
		near(t, sp[20]/sp[0], math.Pow(10, 0.3))
		near(t, ap[24], math.Pow(10, -1.6/20))
		for _, a := range ap[bins : 3*bins] {
			near(t, a, 1)
		}
		near(t, ap[3*bins+24], 1-1e-12)
		return []float64{0.25, -0.25}, nil
	}}}
	got, err := v.Synthesize(idx, match, req)
	if err != nil || !slices.Equal(got, []float64{0.25, -0.25}) || !slices.Equal(req.OutputF0, []float64{300, 310, 320, 330}) {
		t.Fatalf("synthesis got %v, error %v, input F0 %v", got, err, req.OutputF0)
	}
	cfg.UseSafeRefAP = false
	params, err := v.Parameters(idx, match, req)
	if err != nil || !slices.Equal(params.F0, []float64{300, 0, 320, 330}) {
		t.Fatalf("config override got F0 %v, error %v", params.F0, err)
	}
	cfg.UseSafeRefAP = true
	req.OutputF0 = nil
	params, err = v.Parameters(idx, match, req)
	if err != nil {
		t.Fatal(err)
	}
	near(t, params.F0[0], 250*210.0/235)
	near(t, params.F0[3], 250*260.0/235)

	synthErr := errors.New("mock synthesis failure")
	v.World = mockWorld{synth: func([]float64, []float64, []float64, int, int, float64) ([]float64, error) {
		return nil, synthErr
	}}
	if _, err := v.Synthesize(idx, match, req); !errors.Is(err, synthErr) {
		t.Fatalf("got error %v, want %v", err, synthErr)
	}
}

func TestLevelTrackingAndQuietCut(t *testing.T) {
	const rows, bins = 20, 65
	c := worldcodec.New(16000, (bins-1)*2, worldcodec.SpectrumDims)
	spectrum := make([]float64, rows*worldcodec.SpectrumDims)
	levels := make([]float64, rows)
	for row := range rows {
		if row < 5 {
			levels[row] = -160
		}
		if row >= 12 {
			spectrum[row*worldcodec.SpectrumDims] = -3
		}
	}
	cut := quietCut(levels, rows, 40)
	trackLevelWithSupportAndTrack(cut, spectrum, levels, c, make([]float64, bins), 4, 10)
	near(t, cut[0], 80)
	if cut[rows-1] > -3.8 || cut[rows-1] < -4 {
		t.Fatalf("unexpected support gain %v dB", -cut[rows-1])
	}
}
