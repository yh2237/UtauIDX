package index

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
	"github.com/yh2237/audiodsp/wav"
	"github.com/yh2237/gograd/tensor"
)

type testWorld struct {
	calls atomic.Int32
}

func (w *testWorld) Analyze(samples []float64, rate int, frameMS, threshold float64) ([]float64, []float64, []float64, int, error) {
	w.calls.Add(1)
	if frameMS != WorldFrameMS || threshold != D4CThreshold {
		return nil, nil, nil, 0, errors.New("unexpected WORLD analysis parameters")
	}
	const fft = 128
	const bins = fft/2 + 1
	rows := len(samples) * 1000 / (rate * int(frameMS))
	f0 := make([]float64, rows)
	sp, ap := make([]float64, rows*bins), make([]float64, rows*bins)
	for row := range rows {
		f := row / SubFrames
		f0[row] = 200
		if f == 4 && row%SubFrames == 0 {
			f0[row] = 400
		}
		for bin := range bins {
			logPower := math.Sin(float64(bin)/5+float64(f)/2) * 3
			if f == 3 {
				logPower -= 25
			}
			sp[row*bins+bin] = math.Exp(logPower)
			ap[row*bins+bin] = 0.2
			if f == 5 {
				ap[row*bins+bin] = 0.99
			}
		}
	}
	return f0, sp, ap, fft, nil
}

func (*testWorld) Synthesize([]float64, []float64, []float64, int, int, float64) ([]float64, error) {
	return nil, errors.New("unexpected WORLD synthesis")
}

type testEncoder struct {
	id           string
	dims, frames int
	calls        atomic.Int32
}

func (enc *testEncoder) ID() string       { return enc.id }
func (enc *testEncoder) Dims() int        { return enc.dims }
func (enc *testEncoder) FrameMS() float64 { return FrameMS }

func (enc *testEncoder) Extract(samples []float32) ([]float32, int, error) {
	enc.calls.Add(1)
	if len(samples) == 0 {
		return nil, 0, errors.New("empty encoder input")
	}
	return testFeatures(enc.frames, enc.dims), enc.frames, nil
}

func testFeatures(frames, dims int) []float32 {
	values := make([]float32, frames*dims)
	for f := range frames {
		for d := range dims {
			values[f*dims+d] = float32((f+1)*(d+2)%13 - 6)
		}
	}
	return values
}

func testWAV(t *testing.T, dir, name string, rate, ms int, sample int16) string {
	t.Helper()
	pcm := &wav.PCM{SampleRate: rate, Channels: 1, Data: make([]int16, rate*ms/1000)}
	for i := range pcm.Data {
		pcm.Data[i] = sample
	}
	path := filepath.Join(dir, name)
	if err := wav.WriteFile(path, pcm); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestBuildSaveReadHeaderLoad(t *testing.T) {
	dir := t.TempDir()
	path := testWAV(t, dir, "recording.wav", 16000, 160, 7)
	entry := speechpool.Entry{
		Path: path, Layer: speechpool.LayerRaw, Penalty: 0.14,
		Regions: []speechpool.Region{{Offset: 20, Blank: -120}},
	}
	a, b := &testEncoder{id: "ChannelA", dims: 8, frames: 8}, &testEncoder{id: "ChannelB", dims: 4, frames: 7}
	var progress []int
	idx, err := Build([]speechpool.Entry{{Path: filepath.Join(dir, "missing.wav")}, entry},
		&testWorld{}, []encoder.Encoder{a, b}, BuildOptions{
			Source: "bank", CacheKey: "test-key",
			Progress: func(done, total int) {
				if total != 2 {
					t.Errorf("progress total = %d, want 2", total)
				}
				progress = append(progress, done)
			},
		})
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Validate(); err != nil {
		t.Fatal(err)
	}
	if idx.SampleRate != 16000 || idx.Frames != 5 || a.calls.Load() != 1 || b.calls.Load() != 1 ||
		!slices.Equal(progress, []int{1, 2}) || !slices.Equal(idx.Position, []int32{1, 2, 4, 5, 6}) {
		t.Fatalf("build header = %+v, positions = %v, progress = %v, calls = %d/%d", idx.Header, idx.Position, progress, a.calls.Load(), b.calls.Load())
	}
	if idx.Files[0] != (FileEntry{Path: path, Layer: speechpool.LayerRaw, Penalty: 0.14}) ||
		!slices.Equal(idx.FileID, []int32{0, 0, 0, 0, 0}) {
		t.Fatalf("files = %+v, IDs = %v", idx.Files, idx.FileID)
	}
	if idx.F0Median != 200 || idx.C35 <= 0 || idx.C35 != ComputeC35(idx.Spectrum, idx.Voiced) ||
		!slices.Equal(idx.GlitchPenalty, []float32{0, 0, 0.10, 0.12, 0}) {
		t.Fatalf("F0 median = %v, C35 = %v, glitch penalties = %v", idx.F0Median, idx.C35, idx.GlitchPenalty)
	}
	if len(idx.Voiced) != idx.Frames || len(idx.MouthStatic) != idx.Frames*worldcodec.MouthDims ||
		len(idx.MouthDelta) != idx.Frames*worldcodec.MouthDims || idx.Voiced[0] != true {
		t.Fatal("derived features were not populated")
	}
	for _, ch := range idx.Header.Channels {
		rows := idx.Channels[ch.ID]
		for f := range idx.Frames {
			var squared float64
			for _, x := range rows[f*ch.Dims : (f+1)*ch.Dims] {
				squared += float64(x) * float64(x)
			}
			if math.Abs(squared-1) > 1e-6 {
				t.Fatalf("%s row %d has squared norm %g", ch.ID, f, squared)
			}
		}
	}

	file := filepath.Join(dir, "bank.utsp")
	if err := idx.Save(file); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	if string(data[:4]) != "UTSP" {
		t.Fatalf("file magic = %q", data[:4])
	}
	size := int(binary.LittleEndian.Uint32(data[4:8]))
	payload := data[8+size:]
	wantSize := idx.Frames * (SubFrames*2*(SpectrumDims+idx.APBands+1) + 12 + a.Dims() + b.Dims() + 8)
	if len(payload) != wantSize || binary.LittleEndian.Uint16(payload) != worldcodec.Float32ToHalf(idx.Spectrum[0]) {
		t.Fatalf("payload length = %d, want %d; first coefficient = %04x", len(payload), wantSize, payload[:2])
	}
	header, err := ReadHeader(file)
	if err != nil || !reflect.DeepEqual(header, idx.Header) {
		t.Fatalf("ReadHeader = %+v, %v", header, err)
	}
	loaded, err := Load(file)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Header, idx.Header) ||
		!slices.Equal(loaded.Spectrum, idx.Spectrum) || !slices.Equal(loaded.Aperiodicity, idx.Aperiodicity) ||
		!slices.Equal(loaded.F0, idx.F0) || !slices.Equal(loaded.FileID, idx.FileID) ||
		!slices.Equal(loaded.Position, idx.Position) || !slices.Equal(loaded.GlitchPenalty, idx.GlitchPenalty) ||
		!slices.Equal(loaded.Voiced, idx.Voiced) || !slices.Equal(loaded.MouthStatic, idx.MouthStatic) ||
		!slices.Equal(loaded.MouthDelta, idx.MouthDelta) {
		t.Fatal("UTSP round trip changed the header, frame data, or derived features")
	}
	for _, meta := range idx.Header.Channels {
		want := idx.Channels[meta.ID]
		got := loaded.Channels[meta.ID]
		if len(got) != len(want) {
			t.Fatalf("%s length = %d, want %d", meta.ID, len(got), len(want))
		}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("%s value %d = %g, want %g", meta.ID, i, got[i], want[i])
			}
		}
	}
}

func TestLoadRejectsCorruption(t *testing.T) {
	base := Header{Version: Version, SampleRate: 16000, FFTSize: 128, APBands: 1,
		Frames: 1, Channels: []ChannelMeta{{ID: "channel", Dims: 2}}}
	makeHeader := func(h Header) []byte {
		t.Helper()
		data, err := json.Marshal(h)
		if err != nil {
			t.Fatal(err)
		}
		return append(binary.LittleEndian.AppendUint32([]byte(Magic), uint32(len(data))), data...)
	}
	version := base
	version.Version++
	huge := base
	huge.Frames = maxFrames
	badDims := base
	badDims.Channels = []ChannelMeta{{ID: "channel", Dims: maxDims + 1}}
	duplicate := base
	duplicate.Channels = append(slices.Clone(base.Channels), base.Channels[0])
	badFile := base
	badFile.Files = []FileEntry{{Path: "test.wav"}}
	size, _ := expectedPayloadSize(badFile)
	badFilePayload := make([]byte, size)
	fileIDOffset := SubFrames * 2 * (SpectrumDims + badFile.APBands + 1)
	binary.LittleEndian.PutUint32(badFilePayload[fileIDOffset:], 1)
	for _, tc := range []struct {
		name     string
		data     []byte
		headerOK bool
	}{
		{"magic", append([]byte("BAD!"), makeHeader(base)[4:]...), false},
		{"zero header length", binary.LittleEndian.AppendUint32([]byte(Magic), 0), false},
		{"oversized header", binary.LittleEndian.AppendUint32([]byte(Magic), maxHeaderBytes+1), false},
		{"truncated header", binary.LittleEndian.AppendUint32([]byte(Magic), maxHeaderBytes), false},
		{"wrong version", makeHeader(version), false},
		{"invalid dimensions", makeHeader(badDims), false},
		{"duplicate channel", makeHeader(duplicate), false},
		{"truncated payload", append(makeHeader(base), 0), true},
		{"huge declared payload", makeHeader(huge), true},
		{"invalid file ID", append(makeHeader(badFile), badFilePayload...), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "broken.utsp")
			if err := os.WriteFile(path, tc.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := ReadHeader(path); (err == nil) != tc.headerOK {
				t.Errorf("ReadHeader returned %v, want success = %t", err, tc.headerOK)
			}
			if _, err := Load(path); err == nil {
				t.Fatal("Load accepted a corrupt index")
			}
		})
	}
	broken := &Index{Header: huge}
	broken.EnsureDerived()
	if len(broken.MouthStatic) != 0 || len(broken.Voiced) != 0 {
		t.Fatal("derived features were allocated for an invalid index")
	}
}

func TestPackedMatrices(t *testing.T) {
	idx := mergeFixture(speechpool.LayerRaw, []ChannelMeta{{ID: "a", Dims: 3}}, 200, 5, 0)
	packs := make([]*tensor.PackedB, 8)
	var wg sync.WaitGroup
	for i := range packs {
		wg.Go(func() {
			packs[i], _ = idx.PackedChannel("a")
		})
	}
	wg.Wait()
	for _, p := range packs {
		if p == nil || p != packs[0] {
			t.Fatal("concurrent PackedChannel calls did not share the packed matrix")
		}
	}
	if _, ok := idx.PackedChannel("unknown"); ok {
		t.Fatal("PackedChannel accepted an unknown ID")
	}
	static, delta := idx.PackedMouthStatic(), idx.PackedMouthDelta()
	if static == nil || delta == nil || static != idx.PackedMouthStatic() || delta != idx.PackedMouthDelta() {
		t.Fatal("Mouth matrices were not cached")
	}
	idx.ResetPacked()
	p, _ := idx.PackedChannel("a")
	if p == nil || p == packs[0] || idx.PackedMouthStatic() == static || idx.PackedMouthDelta() == delta {
		t.Fatal("ResetPacked did not invalidate a matrix")
	}
}

func TestModalSampleRateAndRecordingSpan(t *testing.T) {
	dir := t.TempDir()
	files := []speechpool.Entry{
		{Path: testWAV(t, dir, "1.wav", 16000, 160, 1)},
		{Path: testWAV(t, dir, "2.wav", 24000, 160, 2)},
		{Path: testWAV(t, dir, "3.wav", 24000, 160, 3)},
		{Path: testWAV(t, dir, "short.wav", 48000, 10, 4)},
	}
	trunc := filepath.Join(dir, "truncated.wav")
	pcm, err := os.ReadFile(files[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(trunc, bytes.Clone(pcm[:len(pcm)-2]), 0o600); err != nil {
		t.Fatal(err)
	}
	files = append(files, speechpool.Entry{Path: trunc}, speechpool.Entry{Path: trunc})
	if got := chooseIndexRate(files); got != 24000 {
		t.Fatalf("chosen sample rate = %d, want 24000", got)
	}
	idx, err := Build(files, &testWorld{}, []encoder.Encoder{&testEncoder{id: "a", dims: 3, frames: 8}}, BuildOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if idx.SampleRate != 24000 || len(idx.Files) != 3 {
		t.Fatalf("Build target rate = %d, files = %v", idx.SampleRate, idx.Files)
	}
	start, end := RecordingSpan([]speechpool.Region{{Offset: 20, Blank: 10}, {Offset: -5, Blank: -20}}, 160)
	if start != 0 || end != 150 {
		t.Fatalf("span = [%v, %v), want [0, 150)", start, end)
	}
}

type sampleCaptureWorld struct {
	mu      sync.Mutex
	samples [][]float64
}

func (w *sampleCaptureWorld) Analyze(samples []float64, rate int, frameMS, threshold float64) ([]float64, []float64, []float64, int, error) {
	w.mu.Lock()
	w.samples = append(w.samples, slices.Clone(samples))
	w.mu.Unlock()
	const fft = 128
	const bins = fft/2 + 1
	rows := len(samples) * 1000 / (rate * int(frameMS))
	f0 := make([]float64, rows)
	sp, ap := make([]float64, rows*bins), make([]float64, rows*bins)
	rev := len(samples) > 0 && samples[0] < 0
	for row := range rows {
		f := row / SubFrames
		if rev {
			f0[row] = 520
		} else {
			f0[row] = 200
		}
		for bin := range bins {
			amp := 2.0
			if rev {
				amp = 9.0
			}
			sp[row*bins+bin] = math.Exp(math.Sin(float64(bin)/4+float64(f)/2) * amp)
			ap[row*bins+bin] = 0.1
		}
	}
	return f0, sp, ap, fft, nil
}

func (*sampleCaptureWorld) Synthesize([]float64, []float64, []float64, int, int, float64) ([]float64, error) {
	return nil, errors.New("unexpected WORLD synthesis")
}

type sampleCaptureEncoder struct {
	id      string
	dims    int
	frames  int
	mu      sync.Mutex
	samples [][]float32
}

func (e *sampleCaptureEncoder) ID() string       { return e.id }
func (e *sampleCaptureEncoder) Dims() int        { return e.dims }
func (e *sampleCaptureEncoder) FrameMS() float64 { return FrameMS }

func (e *sampleCaptureEncoder) Extract(samples []float32) ([]float32, int, error) {
	e.mu.Lock()
	e.samples = append(e.samples, slices.Clone(samples))
	e.mu.Unlock()
	return testFeatures(e.frames, e.dims), e.frames, nil
}

func TestBuildReversedSpanCacheAndAddChannel(t *testing.T) {
	dir := t.TempDir()
	const rate = 16000
	const nSamples = rate * 160 / 1000 // 2560 samples = 160 ms
	pcm := &wav.PCM{SampleRate: rate, Channels: 1, Data: make([]int16, nSamples)}
	for i := range nSamples {
		if i < nSamples/2 {
			pcm.Data[i] = int16(1000 + i)
		} else {
			pcm.Data[i] = int16(-2000 - (i - nSamples/2))
		}
	}
	path := filepath.Join(dir, "asym.wav")
	if err := wav.WriteFile(path, pcm); err != nil {
		t.Fatal(err)
	}
	entry := speechpool.Entry{
		Path: path, Layer: speechpool.LayerRaw, Penalty: 0.14,
		Regions: []speechpool.Region{{Offset: 20, Blank: -80}}, // span [20, 100) ms -> reversed [60, 140) ms
	}
	fwdOnly, err := Build([]speechpool.Entry{entry}, &sampleCaptureWorld{}, []encoder.Encoder{&sampleCaptureEncoder{id: "A", dims: 4, frames: 8}}, BuildOptions{})
	if err != nil {
		t.Fatal(err)
	}

	cache := OpenPartCache(filepath.Join(dir, "parts"))
	w := &sampleCaptureWorld{}
	encA := &sampleCaptureEncoder{id: "A", dims: 4, frames: 8}
	both, err := Build(speechpool.WithReversed([]speechpool.Entry{entry}), w, []encoder.Encoder{encA}, BuildOptions{Cache: cache})
	if err != nil {
		t.Fatal(err)
	}
	if len(both.Files) != 2 || both.Files[0].Reversed || !both.Files[1].Reversed {
		t.Fatalf("files = %+v", both.Files)
	}
	if len(w.samples) != 2 || len(encA.samples) != 2 {
		t.Fatalf("expected 2 analyze + 2 extract calls, got %d / %d", len(w.samples), len(encA.samples))
	}
	// Because processFiles runs in parallel, sort captured slices by sign of first sample.
	wFwd, wRev := w.samples[0], w.samples[1]
	if wFwd[0] < 0 {
		wFwd, wRev = wRev, wFwd
	}
	for i := range nSamples {
		if math.Abs(wFwd[i]-wRev[nSamples-1-i]) > 1e-12 {
			t.Fatalf("world sample %d mismatch: fwd=%g rev=%g", i, wFwd[i], wRev[nSamples-1-i])
		}
	}
	eFwd, eRev := encA.samples[0], encA.samples[1]
	if eFwd[0] < 0 {
		eFwd, eRev = eRev, eFwd
	}
	for i := range nSamples {
		if eFwd[i] != eRev[nSamples-1-i] {
			t.Fatalf("encoder sample %d mismatch: fwd=%g rev=%g", i, eFwd[i], eRev[nSamples-1-i])
		}
	}
	if !slices.Equal(both.Position, []int32{1, 2, 3, 4, 3, 4, 5, 6}) {
		t.Fatalf("positions = %v, want [1 2 3 4 3 4 5 6]", both.Position)
	}
	if both.F0Median != fwdOnly.F0Median || math.Abs(both.C35-fwdOnly.C35) > 1e-9 {
		t.Fatalf("reversed frames altered stats: got F0Med=%v C35=%v, want %v %v",
			both.F0Median, both.C35, fwdOnly.F0Median, fwdOnly.C35)
	}

	wavBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if worldKey(wavBytes, rate, 20, 100, false) == worldKey(wavBytes, rate, 20, 100, true) ||
		channelKey(wavBytes, "A", false) == channelKey(wavBytes, "A", true) {
		t.Fatal("part cache keys collided between forward and reversed")
	}

	// Second Build reuses PartCache with 0 new calls.
	w2 := &sampleCaptureWorld{}
	encA2 := &sampleCaptureEncoder{id: "A", dims: 4, frames: 8}
	if _, err := Build(speechpool.WithReversed([]speechpool.Entry{entry}), w2, []encoder.Encoder{encA2}, BuildOptions{Cache: cache}); err != nil {
		t.Fatal(err)
	}
	if len(w2.samples) != 0 || len(encA2.samples) != 0 {
		t.Fatalf("PartCache missed on rebuild: world=%d enc=%d", len(w2.samples), len(encA2.samples))
	}

	utsp := filepath.Join(dir, "rev.utsp")
	if err := both.Save(utsp); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(utsp)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Files[0].Reversed || !loaded.Files[1].Reversed || !reflect.DeepEqual(loaded.Header, both.Header) {
		t.Fatalf("Load header mismatch: %+v", loaded.Header)
	}

	encB := &sampleCaptureEncoder{id: "B", dims: 3, frames: 8}
	if err := AddChannel(loaded, encB, cache, nil); err != nil {
		t.Fatal(err)
	}
	if len(encB.samples) != 2 {
		t.Fatalf("AddChannel calls = %d, want 2", len(encB.samples))
	}
	bFwd, bRev := encB.samples[0], encB.samples[1]
	if bFwd[0] < 0 {
		bFwd, bRev = bRev, bFwd
	}
	if bFwd[0] != eFwd[0] || bRev[0] != eRev[0] {
		t.Fatalf("AddChannel did not preserve reversed sample orientation")
	}
}
