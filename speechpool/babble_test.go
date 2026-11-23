package speechpool

import (
	"context"
	"crypto/sha256"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/yh2237/audiodsp/wav"
	"github.com/yh2237/utauio/oto"
)

func TestUtilityAliasAndSkipDir(t *testing.T) {
	for _, bad := range []string{"", "br", "br2", "Breath", "R", "sample", "息1", "つ子音", "-", "*"} {
		if !isUtilityAlias(bad) {
			t.Errorf("expected %q to be utility alias", bad)
		}
	}
	for _, good := range []string{"あ", "* あ", "hh ah", "b ae1", "ka", "a B"} {
		if isUtilityAlias(good) {
			t.Errorf("expected %q not to be utility alias", good)
		}
	}
	for _, dir := range []string{"sample", "立ち絵", "息 br", "sub/Art"} {
		if !shouldSkipOtoDir(dir) {
			t.Errorf("expected %q directory to be skipped", dir)
		}
	}
	if shouldSkipOtoDir("CVC") || shouldSkipOtoDir("C3") {
		t.Errorf("valid speech directories should not be skipped")
	}
}

func TestCleanAliasWithAffixesAndTone(t *testing.T) {
	affixes := []bankAffix{{Prefix: "↑", Suffix: "_D4"}, {Prefix: "", Suffix: "_D4"}}
	got, matched := cleanAliasWithAffixes("↑か_D4", affixes)
	if got != "か" || matched == nil || matched.Prefix != "↑" {
		t.Fatalf("got clean=%q matched=%+v, want か with ↑ prefix", got, matched)
	}
	got2, _ := cleanAliasWithAffixes("b aaC#4", nil)
	if got2 != "b aa" {
		t.Fatalf("strip tone: got %q, want %q", got2, "b aa")
	}
}

func TestTopologyDiphoneGraphAndCV(t *testing.T) {
	var diphoneUnits []babbleUnit
	tokens := []string{"-", "hh", "aa", "l", "ow", "-"}
	for i := 0; i < 6; i++ {
		for j := 0; j < 8; j++ {
			l := tokens[i]
			r := tokens[(i+1+j)%len(tokens)]
			diphoneUnits = append(diphoneUnits, babbleUnit{
				Entry:      oto.Entry{Alias: l + " " + r, Preutterance: 80, Overlap: 40, Fixed: 90},
				CleanAlias: l + " " + r,
				Tokens:     []string{l, r},
				Left:       l,
				Right:      r,
				SegMS:      300,
			})
		}
	}
	mode, phrases := detectTopologyAndBuildPhrases(diphoneUnits, 6.0, 0)
	if mode != "diphone-graph" {
		t.Fatalf("got mode %q, want diphone-graph", mode)
	}
	if len(phrases) < 2 {
		t.Fatalf("expected >= 2 diphone phrases, got %d", len(phrases))
	}

	cvUnits := []babbleUnit{
		{Entry: oto.Entry{Alias: "あ"}, CleanAlias: "あ", Tokens: []string{"あ"}, Initial: 'あ', Final: 'あ', SegMS: 300},
		{Entry: oto.Entry{Alias: "か"}, CleanAlias: "か", Tokens: []string{"か"}, Initial: 'か', Final: 'か', SegMS: 300},
		{Entry: oto.Entry{Alias: "さ"}, CleanAlias: "さ", Tokens: []string{"さ"}, Initial: 'さ', Final: 'さ', SegMS: 300},
		{Entry: oto.Entry{Alias: "た"}, CleanAlias: "た", Tokens: []string{"た"}, Initial: 'た', Final: 'た', SegMS: 300},
	}
	cvMode, cvPhrases := detectTopologyAndBuildPhrases(cvUnits, 6.0, 0)
	if cvMode != "cv-diverse" || len(cvPhrases) == 0 {
		t.Fatalf("got cvMode=%q phrases=%d", cvMode, len(cvPhrases))
	}
}

func TestOpenUtauTimingEnvelopeAndF0(t *testing.T) {
	steps := []babbleStep{
		{Unit: &babbleUnit{Entry: oto.Entry{Preutterance: 0, Overlap: 0, Fixed: 50}}, DurationMS: 110},
		{Unit: &babbleUnit{Entry: oto.Entry{Preutterance: 180, Overlap: 90, Fixed: 120}}, DurationMS: 120},
		{Pause: true, DurationMS: 100},
		{Unit: &babbleUnit{Entry: oto.Entry{Preutterance: 40, Overlap: 20, Fixed: 60}}, DurationMS: 140},
	}
	timings, startMS := computeOpenUtauTimings(steps, true)
	if math.IsNaN(startMS) || math.IsInf(startMS, 0) {
		t.Fatalf("invalid phraseStartMS: %v", startMS)
	}
	for i, st := range steps {
		if st.Pause {
			continue
		}
		if timings[i].Preutter < 0 || math.IsNaN(timings[i].Preutter) {
			t.Fatalf("invalid preutter at %d: %+v", i, timings[i])
		}
		env := openUtauEnvelope(st.DurationMS, timings[i])
		if len(env) != 5 || env[4].XMS <= env[0].XMS {
			t.Fatalf("invalid envelope at %d: %+v", i, env)
		}
	}
	for p := 0; p < 4; p++ {
		f0 := buildConversationalF0Curve(220.0, 1500.0, p)
		for idx, v := range f0 {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 80 || v > 500 {
				t.Fatalf("pattern %d frame %d out of range: %v", p, idx, v)
			}
		}
	}
}

func TestBabbleSeedAndSparseDiphones(t *testing.T) {
	units := []babbleUnit{
		{Tokens: []string{"a", "i"}, Left: "a", Right: "i"},
		{Tokens: []string{"i", "e"}, Left: "i", Right: "e"},
		{Tokens: []string{"e", "o"}, Left: "e", Right: "o"},
		{Tokens: []string{"o", "a"}, Left: "o", Right: "a"},
	}
	mode, phrases := detectTopologyAndBuildPhrases(units, 4, 0)
	_, seed42 := detectTopologyAndBuildPhrases(units, 4, 42)
	_, seed43 := detectTopologyAndBuildPhrases(units, 4, 43)
	if mode != "cv-diverse" || len(phrases) == 0 {
		t.Fatalf("mode = %q, phrase count = %d", mode, len(phrases))
	}
	if !reflect.DeepEqual(phrases, seed42) || reflect.DeepEqual(phrases, seed43) {
		t.Fatal("phrase plans did not follow the configured seeds")
	}
}

func TestBabbleEntriesAndCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if summary, err := GenerateBabble(ctx, BabbleOptions{}); !errors.Is(err, context.Canceled) || summary.Phrases != 0 {
		t.Fatalf("GenerateBabble = %+v, %v", summary, err)
	}
	summary := BabbleSummary{Files: []string{"babble_000.wav", "babble_001.wav"}}
	want := []Entry{{Path: summary.Files[0], Layer: LayerBabble}, {Path: summary.Files[1], Layer: LayerBabble}}
	if entries := BabbleEntries(summary); !reflect.DeepEqual(entries, want) {
		t.Errorf("BabbleEntries = %+v, want %+v", entries, want)
	}
}

func TestBabbleNativeReference(t *testing.T) {
	world := os.Getenv("SPEECHPOOL_TEST_WORLD")
	bank := os.Getenv("SPEECHPOOL_TEST_REFERENCE_BANK")
	reference := os.Getenv("SPEECHPOOL_TEST_REFERENCE_WAV")
	if world == "" || bank == "" || reference == "" {
		t.Skip("set SPEECHPOOL_TEST_WORLD and SPEECHPOOL_TEST_REFERENCE_BANK/WAV/F0 to compare with a Step 23 phrase")
	}
	f0, err := strconv.ParseFloat(os.Getenv("SPEECHPOOL_TEST_REFERENCE_F0"), 64)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var progress, totalPhrases int
	summary, err := GenerateBabble(ctx, BabbleOptions{
		BankDir: bank, WorldPath: world, OutDir: t.TempDir(), TargetSec: 8, BaseF0Hz: f0,
		Progress: func(done, total int) {
			progress, totalPhrases = done, total
			cancel()
		},
	})
	if !errors.Is(err, context.Canceled) || progress != 1 || totalPhrases <= 1 || summary.Phrases != 1 || len(summary.Files) != 1 {
		t.Fatalf("GenerateBabble = %+v, %v, progress %d/%d", summary, err, progress, totalPhrases)
	}
	actual, err := os.ReadFile(summary.Files[0])
	if err != nil {
		t.Fatal(err)
	}
	expected, err := os.ReadFile(reference)
	if err != nil {
		t.Fatal(err)
	}
	got, want := sha256.Sum256(actual), sha256.Sum256(expected)
	if got != want {
		t.Fatalf("generated WAV SHA-256 = %x, reference = %x", got, want)
	}
	t.Logf("Step 23 WAV SHA-256 matched: %x", got)
}

func TestSelectBabbleCandidatesSourcePaths(t *testing.T) {
	bankDir := t.TempDir()
	charYAML := "subbanks:\n" +
		"  - color: \"\"\n    prefix: \"\"\n    suffix: _C4\n    tone_ranges:\n      - C4-C4\n" +
		"  - color: Whisper\n    prefix: \"\"\n    suffix: _W\n    tone_ranges:\n      - C4-C4\n"
	if err := os.WriteFile(filepath.Join(bankDir, "character.yaml"), []byte(charYAML), 0o644); err != nil {
		t.Fatal(err)
	}
	writeSineWAV := func(path string) {
		const sr, durMS = 16000, 300
		pcm := &wav.PCM{SampleRate: sr, Channels: 1, Data: make([]int16, sr*durMS/1000)}
		for i := range pcm.Data {
			pcm.Data[i] = int16(8000 * math.Sin(2*math.Pi*220*float64(i)/sr))
		}
		if err := wav.WriteFile(path, pcm); err != nil {
			t.Fatal(err)
		}
	}
	var lines []string
	for i := 0; i < 32; i++ {
		normalPath := filepath.Join(bankDir, "n"+strconv.Itoa(i)+".wav")
		writeSineWAV(normalPath)
		lines = append(lines, "n"+strconv.Itoa(i)+".wav=a"+strconv.Itoa(i)+"_C4,10,50,100,20,10")
	}
	whisperPath := filepath.Join(bankDir, "w.wav")
	writeSineWAV(whisperPath)
	lines = append(lines, "w.wav=ka_W,10,50,100,20,10", "w.wav=br_W,10,50,100,20,10")
	if err := os.WriteFile(filepath.Join(bankDir, "oto.ini"), []byte(strings.Join(lines, "\n")), 0o644); err != nil {
		t.Fatal(err)
	}

	raw, err := scanBankOtoEntries(bankDir)
	if err != nil {
		t.Fatal(err)
	}
	unfiltered := selectBabbleCandidatesForPaths(bankDir, raw, nil)
	direct := selectBabbleCandidates(bankDir, raw)
	if !reflect.DeepEqual(unfiltered, direct) {
		t.Fatalf("empty SourcePaths altered candidate selection")
	}
	for _, c := range unfiltered {
		if c.entry.Filename == whisperPath {
			t.Fatalf("default selection unexpectedly included whisper path %s", whisperPath)
		}
	}

	units, _, sr, err := loadBankUnitsForPaths(bankDir, []string{whisperPath})
	if err != nil {
		t.Fatal(err)
	}
	if sr != 16000 || len(units) != 1 || units[0].Entry.Filename != whisperPath || units[0].CleanAlias != "ka" {
		t.Fatalf("SourcePaths loadBankUnits = %+v (sr=%d)", units, sr)
	}
}
