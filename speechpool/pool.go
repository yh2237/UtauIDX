// Package speechpool prepares recordings for UtauIDX speech indices.
package speechpool

import (
	"fmt"
	"math"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/yh2237/UtauWLD/worldrender"
	"github.com/yh2237/audiodsp/wav"
)

// Layer identifies the origin of a pool entry.
type Layer string

const (
	LayerRaw    Layer = "raw"
	LayerBabble Layer = "babble"
	LayerSpeech Layer = "speech"
)

// PoolMode determines how raw and babble entries are selected.
type PoolMode string

const (
	PoolAuto       PoolMode = "auto"
	PoolBabbleOnly PoolMode = "babble_only"
	PoolMerge      PoolMode = "merge"
	PoolRawOnly    PoolMode = "raw_only"
)

const (
	topologyDiphone = "diphone-graph"
	topologyCV      = "cv-diverse"
)

// Region holds the oto.ini offset and blank in milliseconds.
// Negative blank gives the length from the offset.
type Region struct {
	Offset, Blank float64
}

// Entry is a recording and its optional oto.ini regions.
type Entry struct {
	Path    string
	Layer   Layer
	Regions []Region
	Penalty float32
}

// BankProfile describes the recordings and their suitability for babble generation.
type BankProfile struct {
	TotalOtoEntries, UsableUnits, SampleRate int
	Topology                                 string
	IsSmallJPCV                              bool
}

var (
	utilBreathRe     = regexp.MustCompile(`^(?i:s?br\d*|breath\d*)$`)
	kanaRe           = regexp.MustCompile(`[\x{3040}-\x{30ff}]`)
	sibilantPrefixes = []string{
		"しゃ", "しゅ", "しょ", "ちゃ", "ちゅ", "ちょ", "じゃ", "じゅ", "じょ",
		"シャ", "シュ", "ショ", "チャ", "チュ", "チョ", "ジャ", "ジュ", "ジョ",
	}
	handakuonPrefixes = []string{"ぱ", "ぴ", "ぷ", "ぺ", "ぽ", "パ", "ピ", "プ", "ペ", "ポ"}
)

// CleanFilePenalty returns the filename-based penalty for raw or speech audio.
func CleanFilePenalty(path string, isSmallJPCV bool) float32 {
	stem := strings.TrimSpace(fileStem(path))
	var penalty float32
	if isSmallJPCV {
		switch {
		case strings.Contains(stem, "゜") && !hasPrefix(stem, handakuonPrefixes):
			penalty = 0.18
		case strings.ContainsAny(stem, "ゃゅょャュョ") && !hasPrefix(stem, sibilantPrefixes):
			penalty = 0.14
		}
	}
	if utilBreathRe.MatchString(stem) || strings.ContainsAny(stem, "息吸") {
		penalty = max(penalty, 0.15)
	}
	return penalty
}

func hasPrefix(stem string, prefixes []string) bool {
	for _, prefix := range prefixes {
		if strings.HasPrefix(stem, prefix) {
			return true
		}
	}
	return false
}

func fileStem(path string) string {
	base := filepath.Base(path)
	return strings.TrimSuffix(base, filepath.Ext(base))
}

// ScanRawVoicebank groups valid oto.ini regions by readable WAV path.
// It keeps mixed sample rates and reports the most common rate among the returned files.
func ScanRawVoicebank(bankDir string) ([]Entry, BankProfile, error) {
	raw, err := scanBankOtoEntries(bankDir)
	if err != nil {
		return nil, BankProfile{}, err
	}

	type recording struct {
		entry      Entry
		start, end float64
	}
	cache := make(map[string]*wavCacheEntry)
	recordings := make(map[string]*recording)
	rateCounts := make(map[int]int)
	for _, unit := range raw {
		path := unit.Filename
		cacheBankWAV(cache, path)
		info := cache[path]
		if info == nil {
			continue
		}
		start := max(0, unit.Offset)
		end := min(info.DurMS, worldrender.SourceEndMS(info.DurMS, unit.Offset, unit.Blank))
		if end <= start {
			continue
		}
		rec := recordings[path]
		if rec == nil {
			rec = &recording{entry: Entry{Path: path, Layer: LayerRaw}, start: start, end: end}
			recordings[path] = rec
			rateCounts[info.SampleRate]++
		}
		rec.entry.Regions = append(rec.entry.Regions, Region{Offset: unit.Offset, Blank: unit.Blank})
		rec.start, rec.end = min(rec.start, start), max(rec.end, end)
	}

	profile := BankProfile{TotalOtoEntries: len(raw), SampleRate: modalSampleRate(rateCounts)}
	if len(recordings) == 0 {
		return nil, profile, fmt.Errorf("no readable WAV files with valid oto regions in %s", bankDir)
	}
	candidates := selectBabbleCandidates(bankDir, raw)
	units := collectBabbleUnits(candidates, cache, babbleSampleRate(candidates, cache))
	profile.UsableUnits = len(units)
	profile.Topology = babbleTransitions(units).topology

	paths := make([]string, 0, len(recordings))
	for path := range recordings {
		paths = append(paths, path)
	}
	sort.Strings(paths)

	var spanMS float64
	kanaCount := 0
	for _, path := range paths {
		rec := recordings[path]
		spanMS += rec.end - rec.start
		if kanaRe.MatchString(fileStem(path)) {
			kanaCount++
		}
	}
	profile.IsSmallJPCV = spanMS < 120_000 && kanaCount*5 >= len(paths)*3

	entries := make([]Entry, 0, len(paths))
	for _, path := range paths {
		entry := recordings[path].entry
		entry.Penalty = CleanFilePenalty(path, profile.IsSmallJPCV)
		entries = append(entries, entry)
	}
	return entries, profile, nil
}

func modalSampleRate(counts map[int]int) int {
	rate, best := 0, 0
	for sampleRate, count := range counts {
		if count > best || (count == best && sampleRate > rate) {
			rate, best = sampleRate, count
		}
	}
	return rate
}

// normalizedPCM16 applies a uniform gain to bring RMS to targetRMS, subject to
// a 0.95 peak ceiling. A negative target skips RMS gain.
func normalizedPCM16(samples []float64, sampleRate int, targetRMS float64) *wav.PCM {
	scale := 1.0
	if targetRMS > 0 && len(samples) > 0 {
		var sumSq float64
		for _, v := range samples {
			sumSq += v * v
		}
		rms := math.Sqrt(sumSq/float64(len(samples)) + 1e-12)
		if rms > 1e-5 {
			scale = targetRMS / rms
		}
	}
	var peak float64
	for _, v := range samples {
		peak = max(peak, math.Abs(v))
	}
	if peak > 0 {
		scale = min(scale, 0.95/peak)
	}

	pcm := &wav.PCM{SampleRate: sampleRate, Channels: 1, Data: make([]int16, len(samples))}
	for i, v := range samples {
		pcm.Data[i] = int16(math.RoundToEven(v * scale * 32767))
	}
	return pcm
}

func monoSamples(pcm *wav.PCM) []float64 {
	samples := make([]float64, len(pcm.Data)/pcm.Channels)
	for i := range samples {
		var sum float64
		for _, sample := range pcm.Data[i*pcm.Channels : (i+1)*pcm.Channels] {
			sum += float64(sample)
		}
		samples[i] = sum / float64(pcm.Channels) / 32768
	}
	return samples
}
