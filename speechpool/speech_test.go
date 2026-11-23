package speechpool

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/yh2237/audiodsp/wav"
)

func TestPrepareSpeechDirSplitsAtSilence(t *testing.T) {
	src := t.TempDir()
	nested := filepath.Join(src, "nested")
	if err := os.Mkdir(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	const rate = 8000
	path := filepath.Join(nested, "Breath1.WAV")
	pcm := &wav.PCM{SampleRate: rate, Channels: 2, Data: make([]int16, 15*rate*2)}
	for i := 0; i < len(pcm.Data)/2; i++ {
		sec := float64(i) / rate
		if (sec >= 4.7 && sec < 5.3) || (sec >= 8.6 && sec < 9.2) {
			continue
		}
		v := math.Sin(2 * math.Pi * 170 * sec)
		pcm.Data[2*i], pcm.Data[2*i+1] = int16(v*1300), int16(v*3500)
	}
	if err := wav.WriteFile(path, pcm); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(src, "prepared")
	entries, err := PrepareSpeechDir(src, out, SpeechOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("got %d segments, want 2", len(entries))
	}

	var totalSamples int
	for i, entry := range entries {
		if entry.Layer != LayerSpeech || entry.Penalty != 0.15 || filepath.Dir(entry.Path) != out || len(entry.Regions) != 0 {
			t.Errorf("unexpected entry %+v", entry)
		}
		segment, err := wav.ReadFile(entry.Path)
		if err != nil {
			t.Fatal(err)
		}
		if segment.Channels != 1 || segment.SampleRate != rate {
			t.Errorf("segment format = %d channels at %d Hz", segment.Channels, segment.SampleRate)
		}
		totalSamples += len(segment.Data)
		sec := float64(len(segment.Data)) / rate
		if sec < 0.4 || sec > 10 || (i == 0 && (sec < 8.6 || sec > 9.2)) {
			t.Errorf("segment %d duration = %g s", i, sec)
		}
		var sumSq, peak float64
		for _, sample := range monoSamples(segment) {
			sumSq += sample * sample
			peak = max(peak, math.Abs(sample))
		}
		if rms := math.Sqrt(sumSq / float64(len(segment.Data))); math.Abs(rms-0.10) > 0.001 || peak > 0.95 {
			t.Errorf("segment %d RMS = %g, peak = %g", i, rms, peak)
		}
	}
	if totalSamples != len(pcm.Data)/2 {
		t.Errorf("segments contain %d samples, want %d", totalSamples, len(pcm.Data)/2)
	}
	again, err := PrepareSpeechDir(src, out, SpeechOptions{})
	if err != nil || !reflect.DeepEqual(entries, again) {
		t.Errorf("repeat scan = %+v, %v, want %+v", again, err, entries)
	}
}

func TestPrepareSpeechDirReadOnlyAndUniqueNames(t *testing.T) {
	src := t.TempDir()
	first := filepath.Join(src, "a", "きゃ.wav")
	second := filepath.Join(src, "b", "きゃ.wav")
	writeTestTone(t, second, 8000, 4000, 0.2)
	writeTestTone(t, first, 8000, 4000, 0.2)
	entries, err := PrepareSpeechDir(src, "", SpeechOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want := []Entry{{Path: first, Layer: LayerSpeech}, {Path: second, Layer: LayerSpeech}}
	if !reflect.DeepEqual(entries, want) {
		t.Fatalf("read-only entries = %+v, want %+v", entries, want)
	}
	out := t.TempDir()
	entries, err = PrepareSpeechDir(src, out, SpeechOptions{TargetRMS: -1})
	if err != nil {
		t.Fatal(err)
	}
	want[0].Path = filepath.Join(out, "seg_0001_000.wav")
	want[1].Path = filepath.Join(out, "seg_0002_000.wav")
	if !reflect.DeepEqual(entries, want) {
		t.Fatalf("written entries = %+v, want %+v", entries, want)
	}
	pcm, err := wav.ReadFile(entries[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	var peak float64
	for _, v := range monoSamples(pcm) {
		peak = max(peak, math.Abs(v))
	}
	if math.Abs(peak-0.2) > 0.001 {
		t.Errorf("peak with RMS normalization disabled = %g, want 0.2", peak)
	}
}

func TestSpeechFallbackCutRespectsMinAndMax(t *testing.T) {
	const rate = 8000
	const length = 6*rate + rate/5
	samples := make([]float64, length)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 127 * float64(i) / rate)
	}
	opts := (SpeechOptions{MaxSegmentSec: 3, MinSegmentSec: 1}).withDefaults()
	ends, err := speechSegmentEnds(samples, rate, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(ends) != 3 || ends[len(ends)-1] != length {
		t.Fatalf("segment ends = %v, want 3 segments covering %d samples", ends, length)
	}
	start := 0
	for i, end := range ends {
		if end-start < rate || end-start > 3*rate {
			t.Errorf("segment %d has %d samples", i, end-start)
		}
		if end < length && samples[end-1]*samples[end] > 0 {
			t.Errorf("segment %d cut at %d, want a zero crossing", i, end)
		}
		start = end
	}
}

func TestNormalizedPCM16PeakLimit(t *testing.T) {
	samples := make([]float64, 1000)
	samples[0] = 1
	for _, target := range []float64{0.10, -1} {
		got := monoSamples(normalizedPCM16(samples, 8000, target))[0]
		if math.Abs(got-0.95) > 1.0/32768 {
			t.Errorf("peak with target %g = %g, want 0.95", target, got)
		}
	}
}
