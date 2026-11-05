package refmatch

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/yh2237/UtauIDX/audio"
)

func TestIndexSaveLoad(t *testing.T) {
	index := &Index{
		Header: Header{Version: indexVersion, Model: "test", SampleRate: 44100, FFTSize: 2048, APBands: 2, Dims: 3,
			Frames: 2, F0Median: 220, Files: []string{"a.wav"}},
		Features:     []float32{0.6, 0.8, 0, -1, 0, 0},
		Spectrum:     make([]float32, 2*subFrames*spectrumDims),
		Aperiodicity: make([]float32, 2*subFrames*2),
		F0:           []float32{0, 100, 200, 300, 400, 500, 600, 700},
		File:         []int32{0, 0},
		Position:     []int32{3, 7},
	}
	for i := range index.Spectrum {
		index.Spectrum[i] = float32(i)
	}
	path := filepath.Join(t.TempDir(), "index.bin")
	if err := index.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadIndex(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded.Header, index.Header) || !reflect.DeepEqual(loaded.Spectrum, index.Spectrum) ||
		!reflect.DeepEqual(loaded.F0, index.F0) || !reflect.DeepEqual(loaded.File, index.File) || !reflect.DeepEqual(loaded.Position, index.Position) {
		t.Fatal("index changed after save and load")
	}
	for i, v := range loaded.Features {
		// 特徴量はint8で保存するため、フレームごとのスケール分の誤差が出る。
		if math.Abs(float64(v-index.Features[i])) > 1e-2 {
			t.Fatalf("feature %d = %g, want %g", i, v, index.Features[i])
		}
	}
}

func TestHalfFloat(t *testing.T) {
	for _, v := range []float32{0, 1, -1, 0.5, 65504, 1e-5, -3.14159, 1e-8} {
		got := halfToFloat32(float32ToHalf(v))
		if math.Abs(float64(got-v)) > math.Max(1e-3*math.Abs(float64(v)), 6e-8) {
			t.Errorf("half(%g) = %g", v, got)
		}
	}
}

type fakeWorld struct{}

func (fakeWorld) Analyze(samples []float64, rate int, frameMS, threshold float64) ([]float64, []float64, []float64, int, error) {
	const fft = 1024
	frames := int(1000*float64(len(samples))/float64(rate)/frameMS) + 1
	f0 := make([]float64, frames)
	spectrum := make([]float64, frames*(fft/2+1))
	aperiodicity := make([]float64, frames*(fft/2+1))
	for i := range spectrum {
		spectrum[i], aperiodicity[i] = 1+float64(i%7), 0.1
	}
	return f0, spectrum, aperiodicity, fft, nil
}

func (fakeWorld) Synthesize([]float64, []float64, []float64, int, int, float64) ([]float64, error) {
	return nil, nil
}

func TestBuildIndexSkipsUnreadableWAV(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.wav")
	pcm := &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 8000)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(8000 * math.Sin(float64(i)*0.1))
	}
	if err := audio.WriteWav(good, pcm); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(dir, "bad.wav")
	if err := os.WriteFile(bad, []byte("RIFF\x00\x00\x00\x00WAVEnot a pcm file"), 0o644); err != nil {
		t.Fatal(err)
	}
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		values := make([]float32, frames*3)
		for i := range values {
			values[i] = float32(i%3 + 1)
		}
		return values, frames, 3, nil
	}
	recordings := recordingsOf(bad, good)
	recordings[0].Phoneme = "skipped"
	recordings[1].Phoneme = "ka"
	index, err := BuildIndex(recordings, "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(index.Files, []string{good}) || index.Frames == 0 {
		t.Fatalf("files %v, frames %d", index.Files, index.Frames)
	}
	if !reflect.DeepEqual(index.Phonemes, []string{"ka"}) || !reflect.DeepEqual(index.FilePhonemes, []int32{0}) {
		t.Fatalf("phonemes %v, file phonemes %v", index.Phonemes, index.FilePhonemes)
	}
}

func TestBuildIndexReads24BitWAV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "voice24.wav")
	if err := write24BitWav(path, 16000, 1, 16000); err != nil {
		t.Fatal(err)
	}
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		return make([]float32, frames*3), frames, 3, nil
	}
	index, err := BuildIndex(recordingsOf(path), "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(index.Files, []string{path}) || index.SampleRate != 16000 || index.Frames == 0 {
		t.Fatalf("files %v, rate %d, frames %d", index.Files, index.SampleRate, index.Frames)
	}
}

// write24BitWavは、24bit PCMのWAVを書き出す。8/24/32bitの録音を読めるかの確認に使う。
func write24BitWav(path string, rate, channels, frames int) error {
	data := make([]byte, 0, frames*channels*3)
	for i := 0; i < frames*channels; i++ {
		value := int32(8000) << 8
		data = append(data, byte(value), byte(value>>8), byte(value>>16))
	}
	var buffer bytes.Buffer
	buffer.WriteString("RIFF")
	binary.Write(&buffer, binary.LittleEndian, uint32(0))
	buffer.WriteString("WAVE")
	buffer.WriteString("fmt ")
	binary.Write(&buffer, binary.LittleEndian, uint32(16))
	binary.Write(&buffer, binary.LittleEndian, uint16(1))
	binary.Write(&buffer, binary.LittleEndian, uint16(channels))
	binary.Write(&buffer, binary.LittleEndian, uint32(rate))
	binary.Write(&buffer, binary.LittleEndian, uint32(rate*channels*3))
	binary.Write(&buffer, binary.LittleEndian, uint16(channels*3))
	binary.Write(&buffer, binary.LittleEndian, uint16(24))
	buffer.WriteString("data")
	binary.Write(&buffer, binary.LittleEndian, uint32(len(data)))
	buffer.Write(data)
	out := buffer.Bytes()
	binary.LittleEndian.PutUint32(out[4:], uint32(len(out)-8))
	return os.WriteFile(path, out, 0o644)
}

func TestBuildIndexReusesUnchangedRecordings(t *testing.T) {
	dir := t.TempDir()
	makeWav := func(name string, frequency float64) string {
		path := filepath.Join(dir, name)
		pcm := &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 8000)}
		for i := range pcm.Data {
			pcm.Data[i] = int16(8000 * math.Sin(float64(i)*frequency))
		}
		if err := audio.WriteWav(path, pcm); err != nil {
			t.Fatal(err)
		}
		return path
	}
	files := []string{makeWav("a.wav", 0.1), makeWav("b.wav", 0.2)}
	cache := OpenPartCache(filepath.Join(t.TempDir(), "parts"))
	var calls atomic.Int32
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		calls.Add(1)
		frames := len(samples) * 50 / rate
		values := make([]float32, frames*3)
		for i := range values {
			values[i] = float32(i%3 + 1)
		}
		return values, frames, 3, nil
	}
	first, err := BuildIndex(recordingsOf(files...), "test", fakeWorld{}, features, nil, cache)
	if err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != int32(len(files)) {
		t.Fatalf("first build analyzed %d files", got)
	}
	calls.Store(0)
	second, err := BuildIndex(recordingsOf(files...), "test", fakeWorld{}, features, nil, cache)
	if err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("second build analyzed %d files, want 0", got)
	}
	if first.Frames != second.Frames || !reflect.DeepEqual(first.Features, second.Features) ||
		!reflect.DeepEqual(first.Spectrum, second.Spectrum) || !reflect.DeepEqual(first.Aperiodicity, second.Aperiodicity) ||
		!reflect.DeepEqual(first.F0, second.F0) || !reflect.DeepEqual(first.File, second.File) {
		t.Fatal("index built from cached parts differs")
	}
	if err := audio.WriteWav(files[0], &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 9000)}); err != nil {
		t.Fatal(err)
	}
	calls.Store(0)
	if _, err := BuildIndex(recordingsOf(files...), "test", fakeWorld{}, features, nil, cache); err != nil {
		t.Fatal(err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("build after one change analyzed %d files, want 1", got)
	}
}

func TestBuildIndexTakesRateFromUsableWAV(t *testing.T) {
	dir := t.TempDir()
	short := filepath.Join(dir, "short.wav")
	if err := audio.WriteWav(short, &audio.PCM{SampleRate: 44100, Channels: 1, Data: make([]int16, 441)}); err != nil {
		t.Fatal(err)
	}
	good := filepath.Join(dir, "good.wav")
	pcm := &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 8000)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(8000 * math.Sin(float64(i)*0.1))
	}
	if err := audio.WriteWav(good, pcm); err != nil {
		t.Fatal(err)
	}
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		return make([]float32, frames*3), frames, 3, nil
	}
	index, err := BuildIndex(recordingsOf(short, good), "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if index.SampleRate != 16000 || !reflect.DeepEqual(index.Files, []string{good}) {
		t.Fatalf("rate %d, files %v", index.SampleRate, index.Files)
	}
}

func recordingsOf(paths ...string) []Recording {
	recordings := make([]Recording, len(paths))
	for i, path := range paths {
		recordings[i].Path = path
	}
	return recordings
}

func TestRecordingSpanCoversFirstToLastAlias(t *testing.T) {
	recording := Recording{Regions: []Region{{Offset: 300, Blank: -200}, {Offset: 100, Blank: 400}, {Offset: 900, Blank: -50}}}
	if start, end := recording.span(1000); start != 100 || end != 950 {
		t.Fatalf("span %v-%v, want 100-950", start, end)
	}
	if start, end := (Recording{}).span(1000); start != 0 || end != 1000 {
		t.Fatalf("no aliases: span %v-%v, want the whole recording", start, end)
	}
	if start, end := (Recording{Regions: []Region{{Offset: 800, Blank: 300}}}).span(1000); start != 0 || end != 1000 {
		t.Fatalf("empty alias: span %v-%v, want the whole recording", start, end)
	}
}

func TestBuildIndexSkipsFramesOutsideAliases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "voice.wav")
	pcm := &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 16000)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(8000 * math.Sin(float64(i)*0.1))
	}
	if err := audio.WriteWav(path, pcm); err != nil {
		t.Fatal(err)
	}
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		values := make([]float32, frames*3)
		for i := range values {
			values[i] = float32(i%3 + 1)
		}
		return values, frames, 3, nil
	}
	whole, err := BuildIndex(recordingsOf(path), "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	part, err := BuildIndex([]Recording{{Path: path, Regions: []Region{{Offset: 200, Blank: -400}}}}, "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := 400 / int(FrameMS); part.Frames != want || whole.Frames <= part.Frames {
		t.Fatalf("frames: whole %d, aliased %d, want %d", whole.Frames, part.Frames, want)
	}
}

func writeIndexHeader(t *testing.T, header Header) string {
	t.Helper()
	data, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "index.idx")
	contents := append([]byte(indexMagic), make([]byte, 4)...)
	binary.LittleEndian.PutUint32(contents[4:], uint32(len(data)))
	contents = append(contents, data...)
	if err := os.WriteFile(path, contents, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadIndexRejectsHugeHeaderWithoutPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("LoadIndex panicked on a malformed header: %v", r)
		}
	}()
	for _, header := range []Header{
		{Version: indexVersion, SampleRate: 44100, FFTSize: 2048, APBands: 1, Dims: 1 << 30, Frames: 1 << 30},
		{Version: indexVersion, SampleRate: 44100, FFTSize: 2048, APBands: 1, Dims: 1, Frames: 1 << 30},
	} {
		if _, err := LoadIndex(writeIndexHeader(t, header)); err == nil {
			t.Fatalf("accepted header %+v", header)
		}
	}
}

func TestLoadIndexRejectsInvalidHeader(t *testing.T) {
	base := Header{Version: indexVersion, SampleRate: 44100, FFTSize: 2048, APBands: 5, Dims: 1, Frames: 1}
	for _, test := range []struct {
		name   string
		mutate func(*Header)
	}{
		{"version", func(h *Header) { h.Version = 5 }},
		{"sample rate", func(h *Header) { h.SampleRate = 0 }},
		{"fft size zero", func(h *Header) { h.FFTSize = 0 }},
		{"fft size odd", func(h *Header) { h.FFTSize = 3 }},
		{"ap bands", func(h *Header) { h.APBands = -1 }},
		{"frames", func(h *Header) { h.Frames = 0 }},
		{"dims", func(h *Header) { h.Dims = 0 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			header := base
			test.mutate(&header)
			if _, err := LoadIndex(writeIndexHeader(t, header)); err == nil {
				t.Fatalf("accepted header %+v", header)
			}
		})
	}
}

func TestLoadIndexRejectsTruncatedPayload(t *testing.T) {
	header := Header{Version: indexVersion, SampleRate: 16000, FFTSize: 512, APBands: 1, Dims: 1, Frames: 1}
	if _, err := LoadIndex(writeIndexHeader(t, header)); err == nil {
		t.Fatal("accepted an index without its payload")
	}
}

func validTestIndex() *Index {
	return &Index{
		Header:   Header{Version: indexVersion, SampleRate: 16000, FFTSize: 512, APBands: 1, Dims: 1, Frames: 1, F0Median: 200},
		Features: []float32{1}, Spectrum: make([]float32, subFrames*spectrumDims),
		Aperiodicity: make([]float32, subFrames), F0: []float32{200, 200, 200, 200},
		File: []int32{0}, Position: []int32{0},
	}
}

func TestSaveRejectsInconsistentIndex(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*Index)
	}{
		{"features", func(index *Index) { index.Features = nil }},
		{"spectrum", func(index *Index) { index.Spectrum = nil }},
		{"aperiodicity", func(index *Index) { index.Aperiodicity = nil }},
		{"f0", func(index *Index) { index.F0 = nil }},
		{"file", func(index *Index) { index.File = nil }},
		{"position", func(index *Index) { index.Position = []int32{0, 0} }},
		{"sample rate", func(index *Index) { index.SampleRate = 0 }},
		{"fft size", func(index *Index) { index.FFTSize = 0 }},
		{"version", func(index *Index) { index.Version = 0 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			index := validTestIndex()
			test.mutate(index)
			path := filepath.Join(t.TempDir(), "bad.idx")
			if err := index.Save(path); err == nil {
				t.Fatal("saved an inconsistent index")
			}
			if _, err := os.Stat(path); !os.IsNotExist(err) {
				t.Fatalf("failed save left a file: %v", err)
			}
		})
	}
}

func TestParametersRejectsInvalidIndex(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*Index)
	}{
		{"features", func(index *Index) { index.Features = nil }},
		{"file", func(index *Index) { index.File = nil }},
		{"position", func(index *Index) { index.Position = []int32{0, 0} }},
		{"sample rate", func(index *Index) { index.SampleRate = 0 }},
		{"fft size", func(index *Index) { index.FFTSize = 0 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			index := validTestIndex()
			test.mutate(index)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Parameters panicked: %v", r)
				}
			}()
			request := Request{Features: []float32{1}, Frames: 1, ReferenceF0: []float64{200, 200, 200, 200}}
			if _, err := index.Parameters(request); err == nil {
				t.Fatal("accepted an inconsistent index")
			}
		})
	}
}

func TestParametersWithoutAperiodicityBands(t *testing.T) {
	index := &Index{
		Header:   Header{SampleRate: 8000, FFTSize: 1024, APBands: 0, Dims: 1, Frames: 1, F0Median: 200},
		Features: []float32{1}, Spectrum: make([]float32, subFrames*spectrumDims),
		F0: []float32{200, 200, 200, 200}, File: []int32{0}, Position: []int32{0},
	}
	request := Request{Features: []float32{1}, Frames: 1, ReferenceF0: []float64{200, 200, 200, 200}}
	parameters, err := index.Parameters(request)
	if err != nil {
		t.Fatal(err)
	}
	bins := index.FFTSize/2 + 1
	if len(parameters.Aperiodicity) != subFrames*bins {
		t.Fatalf("aperiodicity length %d, want %d", len(parameters.Aperiodicity), subFrames*bins)
	}
	for i, value := range parameters.Aperiodicity {
		if math.IsNaN(value) || value < 0 || value > 1 {
			t.Fatalf("aperiodicity[%d] = %v", i, value)
		}
	}
}

func TestBuildIndexLowRateRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "low.wav")
	pcm := &audio.PCM{SampleRate: 8000, Channels: 1, Data: make([]int16, 8000)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(8000 * math.Sin(float64(i)*0.1))
	}
	if err := audio.WriteWav(path, pcm); err != nil {
		t.Fatal(err)
	}
	features := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		values := make([]float32, frames*3)
		for i := range values {
			values[i] = float32(i%3 + 1)
		}
		return values, frames, 3, nil
	}
	index, err := BuildIndex(recordingsOf(path), "test", fakeWorld{}, features, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if index.APBands != 0 {
		t.Fatalf("8 kHz index has %d aperiodicity bands, want 0", index.APBands)
	}
	indexPath := filepath.Join(t.TempDir(), "low.idx")
	if err := index.Save(indexPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadIndex(indexPath)
	if err != nil {
		t.Fatal(err)
	}
	loaded.F0Median = 200
	request := Request{Features: []float32{1, 1, 1}, Frames: 1, ReferenceF0: []float64{200, 200, 200, 200}}
	if _, err := loaded.Parameters(request); err != nil {
		t.Fatal(err)
	}
}

type invalidFFTWorld struct{ fakeWorld }

func (invalidFFTWorld) Analyze(samples []float64, rate int, frameMS, threshold float64) ([]float64, []float64, []float64, int, error) {
	f0, spectrum, aperiodicity, _, err := fakeWorld{}.Analyze(samples, rate, frameMS, threshold)
	return f0, spectrum, aperiodicity, 3, err
}

func TestBuildIndexRejectsInconsistentAnalyzer(t *testing.T) {
	path := filepath.Join(t.TempDir(), "voice.wav")
	pcm := &audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 16000)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(8000 * math.Sin(float64(i)*0.1))
	}
	if err := audio.WriteWav(path, pcm); err != nil {
		t.Fatal(err)
	}
	validFeatures := func(samples []float64, rate int) ([]float32, int, int, error) {
		frames := len(samples) * 50 / rate
		return make([]float32, frames*3), frames, 3, nil
	}
	if _, err := BuildIndex(recordingsOf(path), "test", invalidFFTWorld{}, validFeatures, nil, nil); err == nil {
		t.Fatal("accepted an odd FFT size")
	}
	badFeatures := func([]float64, int) ([]float32, int, int, error) {
		return []float32{1}, 2, 3, nil
	}
	if _, err := BuildIndex(recordingsOf(path), "test", fakeWorld{}, badFeatures, nil, nil); err == nil {
		t.Fatal("accepted inconsistent feature dimensions")
	}
}

type shortWorld struct {
	fft, f0Rows, spectrumRows, apRows int
}

func (w shortWorld) Analyze([]float64, int, float64, float64) ([]float64, []float64, []float64, int, error) {
	return make([]float64, w.f0Rows), make([]float64, w.spectrumRows), make([]float64, w.apRows), w.fft, nil
}

func (shortWorld) Synthesize([]float64, []float64, []float64, int, int, float64) ([]float64, error) {
	return nil, nil
}

// WORLD returns subFrames rows per feature frame. A spectrum or aperiodicity
// array with usable*bins values passes a feature-frame check but is four times
// too short for the row loop.
func TestAnalyzeBytesRejectsShortWorldRows(t *testing.T) {
	data := audio.PCMToWavBytes(&audio.PCM{SampleRate: 16000, Channels: 1, Data: make([]int16, 3200)})
	if data == nil {
		t.Fatal("encode WAV fixture")
	}
	features := func([]float64, int) ([]float32, int, int, error) {
		return []float32{1, 1}, 2, 1, nil
	}
	const fft = 256
	bins := fft/2 + 1
	for _, test := range []struct {
		name                 string
		spectrumRows, apRows int
	}{
		{"spectrum", 2 * bins, 8 * bins},
		{"aperiodicity", 8 * bins, 2 * bins},
	} {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("analyzeBytes panicked: %v", recovered)
				}
			}()
			world := shortWorld{fft: fft, f0Rows: 8, spectrumRows: test.spectrumRows, apRows: test.apRows}
			if _, err := analyzeBytes(data, world, features, 0, 0, 1000); err == nil {
				t.Fatal("accepted short WORLD rows")
			}
		})
	}
}
