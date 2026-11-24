package speechpool

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/yh2237/audiodsp/wav"
)

func writeTestText(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTestTone(t *testing.T, path string, rate, frames int, amplitude float64) {
	t.Helper()
	pcm := &wav.PCM{SampleRate: rate, Channels: 1, Data: make([]int16, frames)}
	for i := range pcm.Data {
		pcm.Data[i] = int16(amplitude * 32767 * math.Sin(2*math.Pi*220*float64(i)/float64(rate)))
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := wav.WriteFile(path, pcm); err != nil {
		t.Fatal(err)
	}
}

func TestCleanFilePenalty(t *testing.T) {
	for _, test := range []struct {
		path  string
		small bool
		want  float32
	}{
		{" SBR1 .wav", false, 0.15},
		{"Breath4.wav", false, 0.15},
		{"吸気.wav", false, 0.15},
		{"きゃ.wav", false, 0},
		{"きゃ.wav", true, 0.14},
		{"リョ.wav", true, 0.14},
		{"しゃ.wav", true, 0},
		{"ジョ.wav", true, 0},
		{"か゜.wav", true, 0.18},
		{"ぱ゜.wav", true, 0},
		{"ポ゜.wav", true, 0},
		{"息ゃ.wav", true, 0.15},
		{"吸ゃ゜.wav", true, 0.18},
		{"_xか.wav", true, 0},
	} {
		if got := CleanFilePenalty(test.path, test.small); got != test.want {
			t.Errorf("CleanFilePenalty(%q, %v) = %v, want %v", test.path, test.small, got, test.want)
		}
	}
}

func TestScanRawVoicebankCV(t *testing.T) {
	bank := t.TempDir()
	const rate = 16000
	for _, name := range []string{"きゃ", "しゃ", "か゜", "ぱ゜", "息ゃ", "sbr1", "bad"} {
		writeTestTone(t, filepath.Join(bank, name+".wav"), rate, rate/2, 0.18)
	}
	writeTestTone(t, filepath.Join(bank, "sub", "other.wav"), 24000, 12000, 0.18)
	writeTestTone(t, filepath.Join(bank, "むおん.wav"), rate, rate/2, 0)
	writeTestTone(t, filepath.Join(bank, "みじかい.wav"), rate, rate/100, 0.18)
	writeTestText(t, filepath.Join(bank, "broken.wav"), "corrupt data")
	writeTestText(t, filepath.Join(bank, "oto.ini"), strings.Join([]string{
		"きゃ.wav=きゃ,25,30,-220,10,3",
		"きゃ.wav=* きゃ,26,30,-220,10,3",
		"きゃ.wav=きゃ2,100,30,-100,10,3",
		"しゃ.wav=しゃ,0,30,20,10,3",
		"か゜.wav=か゜,0,30,20,10,3",
		"ぱ゜.wav=ぱ゜,0,30,20,10,3",
		"息ゃ.wav=息ゃ,0,30,20,10,3",
		"sbr1.wav=sbr1,0,30,20,10,3",
		"むおん.wav=む,0,30,0,10,3",
		"みじかい.wav=た,0,0,0,0,0",
		"bad.wav=bad,600,0,0,0,0",
		"broken.wav=broken,0,0,0,0,0",
		"missing.wav=missing,0,0,0,0,0",
	}, "\n"))
	writeTestText(t, filepath.Join(bank, "sub", "OTO.INI"), "other.wav=other,0,30,20,10,3")
	for _, name := range []string{"sample", "サンプル", "Art", "立ち絵", "br", "息", "息 br"} {
		dir := filepath.Join(bank, "sub", name)
		writeTestTone(t, filepath.Join(dir, "skip.wav"), rate, rate/2, 0.18)
		writeTestText(t, filepath.Join(dir, "oto.ini"), "skip.wav=skip,0,30,20,10,3")
	}

	entries, profile, err := ScanRawVoicebank(bank)
	if err != nil {
		t.Fatal(err)
	}
	wantProfile := BankProfile{
		TotalOtoEntries: 14, UsableUnits: 5, SampleRate: rate, Topology: "cv-diverse", P90SpanMS: 500, IsSmallJPCV: true,
	}
	if profile != wantProfile {
		t.Fatalf("profile = %+v, want %+v", profile, wantProfile)
	}
	wantPenalties := map[string]float32{
		"きゃ.wav": 0.14, "しゃ.wav": 0, "か゜.wav": 0.18, "ぱ゜.wav": 0,
		"息ゃ.wav": 0.15, "sbr1.wav": 0.15, "other.wav": 0, "むおん.wav": 0, "みじかい.wav": 0,
	}
	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		paths = append(paths, entry.Path)
		name := filepath.Base(entry.Path)
		want, found := wantPenalties[name]
		if !found || entry.Penalty != want || entry.Layer != LayerRaw {
			t.Errorf("unexpected raw entry %+v", entry)
		}
		delete(wantPenalties, name)
		if name == "きゃ.wav" {
			regions := []Region{{Offset: 25, Blank: -220}, {Offset: 26, Blank: -220}, {Offset: 100, Blank: -100}}
			if !reflect.DeepEqual(entry.Regions, regions) {
				t.Errorf("regions = %+v, want %+v", entry.Regions, regions)
			}
		}
	}
	if len(wantPenalties) != 0 || !sort.StringsAreSorted(paths) {
		t.Errorf("remaining entries = %v, sorted = %v", wantPenalties, sort.StringsAreSorted(paths))
	}
}

func TestScanRawVoicebankVCV(t *testing.T) {
	bank := t.TempDir()
	const rate = 24000
	var rows strings.Builder
	tokens := []string{"a", "e", "i", "o", "u", "n", "-"}
	for _, take := range []struct{ file, prefix, suffix string }{
		{"main.wav", "↑", "_main"}, {"alt.wav", "↓", "_alt"},
	} {
		writeTestTone(t, filepath.Join(bank, take.file), rate, rate*7, 0.18)
		unit := 0
		for i, left := range tokens {
			for j, right := range tokens {
				if i == j {
					continue
				}
				fmt.Fprintf(&rows, "%s=%s%s %s%s,%d,70,-125,55,25\n", take.file, take.prefix, left, right, take.suffix, unit*150)
				unit++
			}
		}
	}
	writeTestText(t, filepath.Join(bank, "oto.ini"), rows.String())
	writeTestText(t, filepath.Join(bank, "character.yaml"), "subbanks:\n - color: ''\n   prefix: ↑\n   suffix: _main\n   tone_ranges:\n    - C4\n")
	writeTestText(t, filepath.Join(bank, "prefix.map"), "C4\t↓\t_alt\n")

	entries, profile, err := ScanRawVoicebank(bank)
	if err != nil {
		t.Fatal(err)
	}
	want := BankProfile{TotalOtoEntries: 84, UsableUnits: 42, SampleRate: rate, Topology: "diphone-graph", P90SpanMS: 6275}
	if profile != want || len(entries) != 2 || len(entries[0].Regions) != 42 || len(entries[1].Regions) != 42 {
		t.Fatalf("profile = %+v, entry count = %d, want %+v and two raw takes", profile, len(entries), want)
	}
	units, _, _, err := loadBankUnits(bank)
	if err != nil {
		t.Fatal(err)
	}
	for _, unit := range units {
		if filepath.Base(unit.Entry.Filename) != "main.wav" {
			t.Fatalf("C4 subbank from character.yaml was not selected: %s", unit.Entry.Filename)
		}
	}
}

func TestScanRawVoicebankDurationThreshold(t *testing.T) {
	bank := t.TempDir()
	const rate = 1000
	writeTestTone(t, filepath.Join(bank, "きゃ.wav"), rate, 120*rate, 0.05)
	writeTestText(t, filepath.Join(bank, "oto.ini"), "きゃ.wav=きゃ,0,0,0,0,0")
	entries, profile, err := ScanRawVoicebank(bank)
	if err != nil {
		t.Fatal(err)
	}
	if profile.IsSmallJPCV || len(entries) != 1 || entries[0].Penalty != 0 {
		t.Errorf("profile = %+v, entries = %+v", profile, entries)
	}
}

func TestMeasureEntries(t *testing.T) {
	dir := t.TempDir()
	const rate = 16000
	aWav := filepath.Join(dir, "a.wav")
	bWav := filepath.Join(dir, "b.wav")
	shortWav := filepath.Join(dir, "short.wav")
	writeTestTone(t, aWav, rate, rate*3, 0.18)      // 3000 ms
	writeTestTone(t, bWav, rate, rate/2, 0.18)      // 500 ms
	writeTestTone(t, shortWav, rate, rate/20, 0.18) // 50 ms (< 100 ms skipped)
	frames, p90 := MeasureEntries([]Entry{
		{Path: aWav, Regions: []Region{{Offset: 100, Blank: -2000}}}, // span 2000 ms -> 100 fr
		{Path: bWav, Regions: []Region{{Offset: 0, Blank: 100}}},     // span 400 ms -> 20 fr
		{Path: shortWav},
	})
	if frames != 120 || p90 != 2000 {
		t.Fatalf("MeasureEntries = (%d, %v), want (120, 2000)", frames, p90)
	}
}
