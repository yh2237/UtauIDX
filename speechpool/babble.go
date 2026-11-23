package speechpool

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/yh2237/UtauWLD/worldrender"
	"github.com/yh2237/audiodsp/wav"
	"github.com/yh2237/utauio/character"
	"github.com/yh2237/utauio/oto"
	"github.com/yh2237/utauio/prefixmap"
	"github.com/yh2237/utauio/textdecode"
)

// BabbleOptions configures voicebank phrase generation.
type BabbleOptions struct {
	BankDir   string
	WorldPath string
	OutDir    string
	TargetSec float64 // Defaults to 180 seconds.
	BaseF0Hz  float64 // A nonpositive value estimates pitch from the bank.
	Seed      int64   // Zero uses seed 42.
	// Optional WAV path allowlist (e.g., a selected voice color). When empty, auto-selects the bank's default pitch/color.
	SourcePaths []string
	Progress    func(done, total int)
}

// BabbleSummary describes the phrases written to disk.
type BabbleSummary struct {
	Bank            string   `json:"bank"`
	Topology        string   `json:"topology"`
	Mode            string   `json:"mode"`
	TotalOtoEntries int      `json:"total_oto_entries"`
	UsableUnits     int      `json:"usable_units"`
	Phrases         int      `json:"phrases"`
	TotalSec        float64  `json:"total_sec"`
	SampleRate      int      `json:"sample_rate"`
	BaseF0Hz        float64  `json:"base_f0_hz"`
	Files           []string `json:"files"`
}

type bankAffix struct {
	Prefix string
	Suffix string
}

type wavCacheEntry struct {
	SampleRate int
	Samples    []float64
	DurMS      float64
}

type babbleUnit struct {
	Entry      oto.Entry
	CleanAlias string
	Tokens     []string
	Left       string
	Right      string
	SegMS      float64
	DurMS      float64
	Initial    rune
	Final      rune
	RMS        float64
}

type babbleStep struct {
	Unit       *babbleUnit
	DurationMS float64
	Pause      bool
}

type openUtauTiming struct {
	Preutter    float64
	Overlap     float64
	TailIntrude float64
	TailOverlap float64
	Overlapped  bool
}

var utilityAliasRe = regexp.MustCompile(`^(?:[a-z]?br\d*|breath\d*|r|sample|end|noise|sil|pau|息.*|吸.*|-|\*)$`)

func shouldSkipOtoDir(relDir string) bool {
	if relDir == "." || relDir == "" {
		return false
	}
	for _, seg := range strings.Split(filepath.ToSlash(relDir), "/") {
		low := strings.ToLower(strings.TrimSpace(seg))
		if low == "" {
			continue
		}
		if strings.Contains(low, "sample") ||
			strings.Contains(low, "サンプル") ||
			strings.Contains(low, "art") ||
			strings.Contains(seg, "立ち絵") ||
			low == "br" ||
			low == "息" ||
			strings.Contains(low, "息 br") {
			return true
		}
	}
	return false
}

func isUtilityAlias(cleanAlias string) bool {
	if cleanAlias == "" {
		return true
	}
	low := strings.ToLower(strings.TrimSpace(cleanAlias))
	if utilityAliasRe.MatchString(low) {
		return true
	}
	if strings.Contains(cleanAlias, "子音") {
		return true
	}
	return false
}

func loadBankAffixes(bankDir string) (preferred bankAffix, hasPreferred bool, all []bankAffix) {
	seen := map[bankAffix]bool{}
	add := func(a bankAffix) {
		if a.Prefix == "" && a.Suffix == "" {
			return
		}
		if !seen[a] {
			seen[a] = true
			all = append(all, a)
		}
	}

	if data, err := os.ReadFile(filepath.Join(bankDir, "character.yaml")); err == nil {
		if text, _, err := textdecode.Decode(data); err == nil {
			cy, _ := character.ParseYAML(text)
			for _, sb := range cy.Subbanks {
				add(bankAffix{Prefix: sb.Prefix, Suffix: sb.Suffix})
			}
			for _, sb := range cy.Subbanks {
				if sb.Color != "" {
					continue
				}
				a := bankAffix{Prefix: sb.Prefix, Suffix: sb.Suffix}
				for _, tr := range sb.ToneRanges {
					if tr.Low <= 60 && 60 <= tr.High { // C4 = MIDI 60
						preferred, hasPreferred = a, true
						break
					}
				}
				if hasPreferred {
					break
				}
			}
		}
	}

	if data, err := os.ReadFile(filepath.Join(bankDir, "prefix.map")); err == nil {
		if text, _, err := textdecode.Decode(data); err == nil {
			var entries []prefixmap.Entry
			_, _ = prefixmap.Scan(text, func(e prefixmap.Entry) {
				entries = append(entries, e)
				add(bankAffix{Prefix: e.Prefix, Suffix: e.Suffix})
			})
			if !hasPreferred && len(entries) > 0 {
				bestDist := 1 << 30
				for _, e := range entries {
					n, ok := midiNoteNumber(e.Tone)
					if !ok {
						continue
					}
					d := absInt(n - 60)
					if d < bestDist {
						bestDist = d
						preferred = bankAffix{Prefix: e.Prefix, Suffix: e.Suffix}
						hasPreferred = true
					}
				}
			}
		}
	}
	return preferred, hasPreferred, all
}

func midiNoteNumber(tone string) (int, bool) {
	tone = strings.ToUpper(strings.TrimSpace(tone))
	if len(tone) < 2 {
		return 0, false
	}
	semis := map[byte]int{'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
	s, ok := semis[tone[0]]
	if !ok {
		return 0, false
	}
	i := 1
	if i < len(tone) && (tone[i] == '#' || tone[i] == 'B') {
		if tone[i] == '#' {
			s++
		} else {
			s--
		}
		i++
	}
	oct, err := strconv.Atoi(tone[i:])
	if err != nil {
		return 0, false
	}
	return (oct+1)*12 + s, true
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func stripTrailingTone(alias string) string {
	for size := 3; size >= 2; size-- {
		if len(alias) <= size {
			continue
		}
		suf := alias[len(alias)-size:]
		if _, ok := midiNoteNumber(suf); ok {
			return strings.TrimSpace(alias[:len(alias)-size])
		}
	}
	return alias
}

func cleanAliasWithAffixes(alias string, affixes []bankAffix) (string, *bankAffix) {
	alias = strings.TrimSpace(alias)
	var matched *bankAffix
	bestLen := -1
	for i := range affixes {
		a := affixes[i]
		if a.Prefix != "" && !strings.HasPrefix(alias, a.Prefix) {
			continue
		}
		if a.Suffix != "" && !strings.HasSuffix(alias, a.Suffix) {
			continue
		}
		if len(a.Prefix)+len(a.Suffix) > bestLen {
			bestLen = len(a.Prefix) + len(a.Suffix)
			aa := a
			matched = &aa
		}
	}
	if matched != nil {
		alias = strings.TrimSuffix(strings.TrimPrefix(alias, matched.Prefix), matched.Suffix)
	}
	alias = stripTrailingTone(strings.TrimSpace(alias))
	return alias, matched
}

func scanBankOtoEntries(bankDir string) ([]oto.Entry, error) {
	var otoFiles []string
	err := filepath.WalkDir(bankDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			rel, _ := filepath.Rel(bankDir, path)
			if shouldSkipOtoDir(rel) {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.EqualFold(d.Name(), "oto.ini") {
			otoFiles = append(otoFiles, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(otoFiles) == 0 {
		return nil, fmt.Errorf("no oto.ini found in %s", bankDir)
	}
	sort.Strings(otoFiles)

	var entries []oto.Entry
	for _, p := range otoFiles {
		raw, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		if len(raw) == 0 {
			continue
		}
		text, _, err := textdecode.Decode(raw)
		if err != nil {
			return nil, fmt.Errorf("decode %s: %w", p, err)
		}
		baseDir := filepath.Dir(p)
		_, err = oto.Scan(strings.NewReader(text), baseDir, func(e oto.Entry) {
			entries = append(entries, e)
		})
		if err != nil {
			return nil, fmt.Errorf("scan %s: %w", p, err)
		}
	}
	return entries, nil
}

func loadBankUnits(bankDir string) ([]babbleUnit, int, int, error) {
	return loadBankUnitsForPaths(bankDir, nil)
}

func loadBankUnitsForPaths(bankDir string, sourcePaths []string) ([]babbleUnit, int, int, error) {
	rawEntries, err := scanBankOtoEntries(bankDir)
	if err != nil {
		return nil, 0, 0, err
	}
	candidates := selectBabbleCandidatesForPaths(bankDir, rawEntries, sourcePaths)
	cache := make(map[string]*wavCacheEntry)
	for _, item := range candidates {
		cacheBankWAV(cache, item.entry.Filename)
	}
	sampleRate := babbleSampleRate(candidates, cache)
	if sampleRate == 0 {
		return nil, len(rawEntries), 0, fmt.Errorf("no readable speech WAV files referenced by %s", bankDir)
	}
	return collectBabbleUnits(candidates, cache, sampleRate), len(rawEntries), sampleRate, nil
}

type babbleCandidate struct {
	entry   oto.Entry
	clean   string
	matched *bankAffix
}

func normalizeSourcePath(path string) string {
	clean := filepath.Clean(path)
	if runtime.GOOS == "windows" {
		clean = strings.ToLower(clean)
	}
	return clean
}

func selectBabbleCandidatesForPaths(bankDir string, rawEntries []oto.Entry, sourcePaths []string) []babbleCandidate {
	if len(sourcePaths) == 0 {
		return selectBabbleCandidates(bankDir, rawEntries)
	}
	allowed := make(map[string]bool, len(sourcePaths))
	for _, p := range sourcePaths {
		if strings.TrimSpace(p) != "" {
			allowed[normalizeSourcePath(p)] = true
		}
	}
	_, _, affixes := loadBankAffixes(bankDir)
	filtered := make([]babbleCandidate, 0, len(rawEntries))
	for _, e := range rawEntries {
		if !allowed[normalizeSourcePath(e.Filename)] {
			continue
		}
		clean, matched := cleanAliasWithAffixes(e.Alias, affixes)
		if isUtilityAlias(clean) {
			continue
		}
		filtered = append(filtered, babbleCandidate{entry: e, clean: clean, matched: matched})
	}
	return filtered
}

func selectBabbleCandidates(bankDir string, rawEntries []oto.Entry) []babbleCandidate {
	preferred, hasPreferred, affixes := loadBankAffixes(bankDir)

	filtered := make([]babbleCandidate, 0, len(rawEntries))
	affixCounts := map[bankAffix]int{}
	for _, e := range rawEntries {
		clean, matched := cleanAliasWithAffixes(e.Alias, affixes)
		if isUtilityAlias(clean) {
			continue
		}
		if matched != nil {
			affixCounts[*matched]++
		}
		filtered = append(filtered, babbleCandidate{entry: e, clean: clean, matched: matched})
	}
	// Pick one dominant/preferred affix only when multiple affixes cover substantial subsets.
	if len(affixes) > 1 {
		targetAffix := preferred
		if !hasPreferred {
			bestCnt := 0
			for _, a := range affixes {
				if c := affixCounts[a]; c > bestCnt {
					bestCnt = c
					targetAffix = a
				}
			}
		}
		if affixCounts[targetAffix] > 30 {
			var narrowed []babbleCandidate
			for _, item := range filtered {
				if item.matched != nil && *item.matched == targetAffix {
					narrowed = append(narrowed, item)
				}
			}
			if len(narrowed) > 30 {
				filtered = narrowed
			}
		}
	}

	return filtered
}

func cacheBankWAV(cache map[string]*wavCacheEntry, path string) {
	if _, ok := cache[path]; ok {
		return
	}
	pcm, err := wav.ReadFile(path)
	if err != nil || len(pcm.Data) == 0 {
		cache[path] = nil
		return
	}
	samples := monoSamples(pcm)
	cache[path] = &wavCacheEntry{
		SampleRate: pcm.SampleRate,
		Samples:    samples,
		DurMS:      float64(len(samples)) * 1000 / float64(pcm.SampleRate),
	}
}

func babbleSampleRate(candidates []babbleCandidate, cache map[string]*wavCacheEntry) int {
	counts := make(map[int]int)
	seen := make(map[string]bool)
	for _, item := range candidates {
		path := item.entry.Filename
		if wc := cache[path]; wc != nil && !seen[path] {
			counts[wc.SampleRate]++
			seen[path] = true
		}
	}
	return modalSampleRate(counts)
}

func collectBabbleUnits(filtered []babbleCandidate, wavCache map[string]*wavCacheEntry, targetSR int) []babbleUnit {
	type dedupKey struct {
		path  string
		start int
		end   int
	}
	dedup := map[dedupKey]babbleUnit{}
	var orderedKeys []dedupKey

	for _, item := range filtered {
		e := item.entry
		p := filepath.Clean(e.Filename)
		wc := wavCache[p]
		if wc == nil || wc.SampleRate != targetSR {
			continue
		}
		if e.Offset < 0 || math.IsNaN(e.Offset) || math.IsNaN(e.Fixed) || math.IsNaN(e.Blank) {
			continue
		}
		if e.Offset == 0 && e.Fixed == 0 && e.Blank == 0 && e.Preutterance == 0 && e.Overlap == 0 && wc.DurMS > 2600 {
			continue
		}
		endMS := worldrender.SourceEndMS(wc.DurMS, e.Offset, e.Blank)
		segMS := endMS - e.Offset
		if segMS < 45.0 || segMS > 2500.0 || endMS > wc.DurMS+5.0 {
			continue
		}
		startFrame := max(0, int(e.Offset/worldrender.FramePeriodMS))
		endFrame := min(len(wc.Samples)*1000/(wc.SampleRate*int(worldrender.FramePeriodMS))+1, int(math.Ceil(endMS/worldrender.FramePeriodMS)))
		if endFrame-startFrame < 4 {
			continue
		}
		sIdx := max(0, int(math.Round(e.Offset*float64(wc.SampleRate)/1000.0)))
		eIdx := min(len(wc.Samples), int(math.Round(endMS*float64(wc.SampleRate)/1000.0)))
		if eIdx-sIdx < 200 {
			continue
		}
		var sumSq float64
		for _, v := range wc.Samples[sIdx:eIdx] {
			sumSq += v * v
		}
		rms := math.Sqrt(sumSq / float64(eIdx-sIdx))
		if rms < 0.008 {
			continue
		}

		// Clamp pathological oto preutter/overlap against segment length so envelopes stay valid.
		if e.Preutterance > segMS*0.85 {
			e.Preutterance = segMS * 0.85
		}
		if e.Preutterance < 0 {
			e.Preutterance = 0
		}
		if e.Overlap > e.Preutterance*0.95 {
			e.Overlap = e.Preutterance * 0.95
		}
		if e.Fixed < 0 {
			e.Fixed = 0
		}

		toks := strings.Fields(item.clean)
		if len(toks) == 0 {
			continue
		}
		left, right := "", ""
		if len(toks) == 2 {
			left, right = toks[0], toks[1]
		}
		initR, _ := utf8.DecodeRuneInString(toks[len(toks)-1])
		lastTok := toks[len(toks)-1]
		finR, _ := utf8.DecodeLastRuneInString(lastTok)

		bu := babbleUnit{
			Entry:      e,
			CleanAlias: item.clean,
			Tokens:     toks,
			Left:       left,
			Right:      right,
			SegMS:      segMS,
			DurMS:      wc.DurMS,
			Initial:    initR,
			Final:      finR,
			RMS:        rms,
		}
		dk := dedupKey{
			path:  p,
			start: int(math.Round(e.Offset / 5.0)),
			end:   int(math.Round(endMS / 5.0)),
		}
		prev, exists := dedup[dk]
		if !exists {
			orderedKeys = append(orderedKeys, dk)
			dedup[dk] = bu
		} else if preferBabbleAlias(bu.CleanAlias, prev.CleanAlias) {
			dedup[dk] = bu
		}
	}

	units := make([]babbleUnit, 0, len(orderedKeys))
	for _, dk := range orderedKeys {
		units = append(units, dedup[dk])
	}
	sort.SliceStable(units, func(i, j int) bool {
		if units[i].CleanAlias != units[j].CleanAlias {
			return units[i].CleanAlias < units[j].CleanAlias
		}
		return units[i].Entry.Filename < units[j].Entry.Filename
	})
	return units
}

func preferBabbleAlias(a, b string) bool {
	aStar := strings.HasPrefix(a, "*") || strings.HasPrefix(a, "-") || strings.Contains(a, "・")
	bStar := strings.HasPrefix(b, "*") || strings.HasPrefix(b, "-") || strings.Contains(b, "・")
	if aStar != bStar {
		return !aStar
	}
	if len(a) != len(b) {
		return len(a) < len(b)
	}
	return a < b
}

func isChainBoundary(tok string) bool {
	return tok == "-" || tok == "*" || tok == "・"
}

type babbleGraph struct {
	topology            string
	diphones, syllables []int
	exact, base         map[string][]int
}

func babbleTransitions(units []babbleUnit) babbleGraph {
	var twoTokIdx []int
	var oneTokIdx []int
	adjExact := map[string][]int{}
	adjBase := map[string][]int{}

	for i := range units {
		u := &units[i]
		if len(u.Tokens) == 2 && u.Left != "*" && u.Left != "・" {
			twoTokIdx = append(twoTokIdx, i)
			adjExact[u.Left] = append(adjExact[u.Left], i)
			baseLeft := strings.TrimRight(u.Left, "0123456789")
			adjBase[baseLeft] = append(adjBase[baseLeft], i)
		} else if len(u.Tokens) == 1 {
			oneTokIdx = append(oneTokIdx, i)
		}
	}

	chainable := 0
	for _, idx := range twoTokIdx {
		r := units[idx].Right
		if isChainBoundary(r) {
			continue
		}
		if len(adjExact[r]) > 0 || len(adjBase[strings.TrimRight(r, "0123456789")]) > 0 {
			chainable++
		}
	}

	useDiphoneGraph := len(twoTokIdx) >= 40 && chainable*2 >= len(units)
	var mode string
	if useDiphoneGraph {
		mode = topologyDiphone
	} else {
		mode = topologyCV
		if len(oneTokIdx) == 0 || (len(oneTokIdx) < 15 && len(twoTokIdx) >= 15) {
			for i := range units {
				oneTokIdx = append(oneTokIdx, i)
			}
		}
	}

	return babbleGraph{mode, twoTokIdx, oneTokIdx, adjExact, adjBase}
}

func detectTopologyAndBuildPhrases(units []babbleUnit, targetSec float64, seed int64) (mode string, phrases [][]babbleStep) {
	graph := babbleTransitions(units)
	mode = graph.topology
	if len(units) == 0 {
		return mode, nil
	}
	if seed == 0 {
		seed = 42
	}
	rng := rand.New(rand.NewSource(seed))
	useDiphoneGraph := mode == topologyDiphone
	twoTokIdx, oneTokIdx := graph.diphones, graph.syllables
	adjExact, adjBase := graph.exact, graph.base

	visits := make([]int, len(units))
	pickMinVisit := func(cands []int, avoidIdx int, avoidRune rune) int {
		if len(cands) == 0 {
			return -1
		}
		minScore := 1 << 30
		var best []int
		for _, ci := range cands {
			if len(cands) > 1 && ci == avoidIdx {
				continue
			}
			score := visits[ci] * 4
			if avoidRune != 0 && len(cands) > 6 && units[ci].Initial == avoidRune {
				score += 2
			}
			if score < minScore {
				minScore = score
				best = best[:0]
				best = append(best, ci)
			} else if score == minScore {
				best = append(best, ci)
			}
		}
		if len(best) == 0 {
			return cands[rng.Intn(len(cands))]
		}
		return best[rng.Intn(len(best))]
	}

	var starters []int
	if useDiphoneGraph {
		for _, idx := range twoTokIdx {
			if units[idx].Left == "-" && !isChainBoundary(units[idx].Right) {
				starters = append(starters, idx)
			}
		}
		if len(starters) == 0 {
			for _, idx := range twoTokIdx {
				if !isChainBoundary(units[idx].Right) {
					starters = append(starters, idx)
				}
			}
		}
		if len(starters) == 0 {
			starters = twoTokIdx
		}
	}

	cumSec := 0.0
	for pIdx := 0; cumSec < targetSec && pIdx < 160; pIdx++ {
		var steps []babbleStep
		if useDiphoneGraph {
			nUnits := 10 + rng.Intn(6) // 10..15 units
			midPause := -1
			if nUnits >= 12 {
				midPause = 4 + rng.Intn(nUnits-7)
			}
			curr := pickMinVisit(starters, -1, 0)
			for s := 0; s < nUnits; s++ {
				if s == midPause {
					steps = append(steps, babbleStep{DurationMS: 95 + rng.Float64()*35, Pause: true})
					curr = pickMinVisit(starters, curr, 0)
				}
				visits[curr]++
				dur := 102.0 * (0.82 + rng.Float64()*0.40)
				if s == nUnits-1 || s == midPause-1 {
					dur = 140.0 + rng.Float64()*40.0
				}
				steps = append(steps, babbleStep{Unit: &units[curr], DurationMS: dur})
				if s == nUnits-1 {
					break
				}
				right := units[curr].Right
				if isChainBoundary(right) {
					steps = append(steps, babbleStep{DurationMS: 90, Pause: true})
					curr = pickMinVisit(starters, curr, 0)
					continue
				}
				nextCands := adjExact[right]
				if len(nextCands) == 0 {
					nextCands = adjBase[strings.TrimRight(right, "0123456789")]
				}
				if s+2 == nUnits {
					var endings []int
					for _, nc := range nextCands {
						if units[nc].Right == "-" {
							endings = append(endings, nc)
						}
					}
					if len(endings) > 0 {
						nextCands = endings
					}
				}
				if len(nextCands) == 0 {
					steps = append(steps, babbleStep{DurationMS: 90, Pause: true})
					curr = pickMinVisit(starters, curr, 0)
				} else {
					curr = pickMinVisit(nextCands, curr, 0)
				}
			}
		} else {
			nUnits := 12 + rng.Intn(7) // 12..18 units
			midPause := -1
			if nUnits >= 14 {
				midPause = 5 + rng.Intn(nUnits-9)
			}
			prevIdx := -1
			var prevFinal, avoidInit rune
			for s := 0; s < nUnits; s++ {
				if s == midPause {
					steps = append(steps, babbleStep{DurationMS: 95 + rng.Float64()*40, Pause: true})
					prevIdx = -1
					prevFinal = 0
					avoidInit = 0
					continue
				}
				// Splice a matching 2-token transition (e.g. V-V or V-C in CVVC banks) 25% of the time if available.
				if prevFinal != 0 && rng.Float64() < 0.25 {
					if transCands := adjExact[string(prevFinal)]; len(transCands) > 0 {
						tc := pickMinVisit(transCands, prevIdx, 0)
						visits[tc]++
						steps = append(steps, babbleStep{
							Unit:       &units[tc],
							DurationMS: 90.0 + rng.Float64()*30.0,
						})
						prevIdx = tc
						avoidInit = 0
						prevFinal = 0
						continue
					}
				}
				chosen := pickMinVisit(oneTokIdx, prevIdx, avoidInit)
				visits[chosen]++
				dur := 118.0 * (0.80 + rng.Float64()*0.44)
				if s == nUnits-1 || s == midPause-1 {
					dur = 155.0 + rng.Float64()*45.0
				}
				steps = append(steps, babbleStep{Unit: &units[chosen], DurationMS: dur})
				prevIdx = chosen
				avoidInit = units[chosen].Initial
				prevFinal = units[chosen].Final
			}
		}

		pSec := 0.14
		for _, st := range steps {
			pSec += st.DurationMS / 1000.0
		}
		if len(steps) > 0 {
			phrases = append(phrases, steps)
			cumSec += pSec
		}
	}
	return mode, phrases
}

func computeOpenUtauTimings(steps []babbleStep, singleCV bool) ([]openUtauTiming, float64) {
	timings := make([]openUtauTiming, len(steps))
	previous := -1
	first := -1
	for i, st := range steps {
		if st.Pause || st.Unit == nil {
			continue
		}
		if first < 0 {
			first = i
		}
		u := st.Unit
		autoPre := u.Entry.Preutterance
		autoOv := u.Entry.Overlap
		if singleCV && autoOv <= 0 && autoPre <= 15 {
			if autoPre < 18.0 {
				autoPre = 18.0
			}
			autoOv = 18.0
		}

		adjacent := false
		if previous >= 0 {
			prevStep := steps[previous]
			prevStart := stepStartMS(steps, previous)
			curStart := stepStartMS(steps, i)
			gapMS := curStart - (prevStart + prevStep.DurationMS)
			prevDur := prevStep.DurationMS
			maxPre := autoPre
			if gapMS <= 0 {
				adjacent = true
				if autoOv > 0 && autoPre-autoOv > prevDur*0.5 {
					maxPre = prevDur * 0.5 / (autoPre - autoOv) * autoPre
				} else if autoOv <= 0 {
					maxPre = math.Min(maxPre, prevDur*0.9)
				}
				maxPre = math.Min(maxPre, prevDur)
				if timings[previous].Preutter < 5 {
					maxPre = math.Min(maxPre, prevDur+timings[previous].Preutter-5)
				}
			} else if gapMS < autoPre {
				maxPre = gapMS
			}
			if autoPre > maxPre && autoPre > 0 {
				ratio := maxPre / autoPre
				autoPre = maxPre
				autoOv *= ratio
			}
			if autoOv < 0 {
				autoOv = math.Max(autoOv, math.Min(0, 35-prevDur+autoPre))
			}
		}
		autoPre = math.Max(0, autoPre)
		timings[i].Preutter = autoPre
		timings[i].Overlap = autoOv
		timings[i].Overlapped = previous >= 0 && adjacent && autoOv > 0
		if previous >= 0 && adjacent {
			timings[previous].TailIntrude = math.Max(timings[previous].TailIntrude, math.Max(autoPre, autoPre-autoOv))
			timings[previous].TailOverlap = math.Max(timings[previous].TailOverlap, math.Max(autoOv, 0))
		}
		previous = i
	}
	phraseStart := 0.0
	if first >= 0 {
		phraseStart = stepStartMS(steps, first) - timings[first].Preutter
	}
	return timings, phraseStart
}

func stepStartMS(steps []babbleStep, idx int) float64 {
	s := 0.0
	for i := 0; i < idx && i < len(steps); i++ {
		s += steps[i].DurationMS
	}
	return s
}

func openUtauEnvelope(durationMS float64, t openUtauTiming) []worldrender.EnvelopePoint {
	fadeIn := 5.0
	if t.Overlapped {
		fadeIn = t.Overlap
	}
	fadeOut := 35.0
	if t.TailOverlap > 0 {
		fadeOut = t.TailOverlap
	}
	p0 := -t.Preutter
	p1 := math.Max(p0+5, p0+fadeIn)
	p2 := math.Max(0, p1)
	p4 := durationMS - t.TailIntrude + t.TailOverlap
	if p4 <= p0+10 {
		p4 = p0 + 10
	}
	p3 := math.Max(p2, p4-fadeOut)
	return []worldrender.EnvelopePoint{
		{XMS: p0, Y: 0},
		{XMS: p1, Y: 1},
		{XMS: p2, Y: 1},
		{XMS: p3, Y: 1},
		{XMS: p4, Y: 0},
	}
}

func buildConversationalF0Curve(baseHz float64, totalMS float64, patternIdx int) []float64 {
	frames := max(8, int(math.Ceil(totalMS/worldrender.FramePeriodMS))+4)
	curve := make([]float64, frames)
	pat := patternIdx % 4
	for f := 0; f < frames; f++ {
		p := float64(f) / float64(max(1, frames-1))
		var semi float64
		switch pat {
		case 0: // Declarative declination + gentle undulation
			semi = 2.0 - 4.4*p + 1.1*math.Sin(2*math.Pi*2.3*p)
		case 1: // Early-mid peak falling toward tail
			semi = 0.6 + 2.8*math.Sin(math.Pi*math.Pow(p, 0.75)) - 2.2*p + 0.8*math.Sin(2*math.Pi*3.1*p)
		case 2: // Continuation / rising tail
			semi = 1.1 - 2.3*math.Sin(math.Pi*p*0.85) + 3.1*math.Pow(p, 2.2) + 0.7*math.Cos(2*math.Pi*2.7*p)
		default: // Double-accent prosodic wave
			semi = 1.8 - 3.6*p + 1.6*math.Sin(2*math.Pi*2.0*p+0.4) + 0.8*math.Sin(2*math.Pi*4.0*p)
		}
		curve[f] = baseHz * math.Pow(2.0, semi/12.0)
	}
	return curve
}

func estimateBankBaseF0(ctx context.Context, units []babbleUnit, targetSR int, worldPath string) (float64, error) {
	w, err := worldrender.OpenWorld(worldPath)
	if err != nil {
		return 0, err
	}
	defer w.Close()

	step := max(1, len(units)/16)
	var f0Vals []float64
	seenFiles := map[string]bool{}
	for i := 0; i < len(units) && len(f0Vals) < 256; i += step {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		u := units[i]
		if seenFiles[u.Entry.Filename] && len(units) > 16 {
			continue
		}
		seenFiles[u.Entry.Filename] = true
		pcm, err := wav.ReadFile(u.Entry.Filename)
		if err != nil || pcm.SampleRate != targetSR {
			continue
		}
		samples := monoSamples(pcm)
		endMS := worldrender.SourceEndMS(float64(len(samples))*1000.0/float64(targetSR), u.Entry.Offset, u.Entry.Blank)
		sIdx := max(0, int(u.Entry.Offset*float64(targetSR)/1000.0))
		eIdx := min(len(samples), int(endMS*float64(targetSR)/1000.0))
		if eIdx-sIdx < targetSR/10 {
			continue
		}
		f0, _, _, _, err := w.Analyze(samples[sIdx:eIdx], targetSR, 10.0, 0.85)
		if err != nil {
			continue
		}
		for _, hz := range f0 {
			if hz >= 70 && hz <= 600 {
				f0Vals = append(f0Vals, hz)
			}
		}
	}
	if len(f0Vals) == 0 {
		return 220.0, nil
	}
	sort.Float64s(f0Vals)
	med := f0Vals[len(f0Vals)/2]
	return math.Max(80, math.Min(450, med)), nil
}

// GenerateBabble renders deterministic conversational phrases from a voicebank.
// On error, the summary includes any phrases already written.
func GenerateBabble(ctx context.Context, opts BabbleOptions) (BabbleSummary, error) {
	if err := ctx.Err(); err != nil {
		return BabbleSummary{}, err
	}
	if opts.WorldPath == "" {
		return BabbleSummary{}, errors.New("babble requires a WORLD library path")
	}
	if opts.OutDir == "" {
		return BabbleSummary{}, errors.New("babble requires an output directory")
	}
	bankDir, worldPath, outDir := opts.BankDir, opts.WorldPath, opts.OutDir
	targetSec := opts.TargetSec
	if targetSec <= 0 {
		targetSec = 180.0
	}
	if math.IsNaN(targetSec) || math.IsInf(targetSec, 0) || math.IsNaN(opts.BaseF0Hz) || math.IsInf(opts.BaseF0Hz, 0) {
		return BabbleSummary{}, errors.New("babble duration and base F0 must be finite")
	}

	units, totalEntries, sampleRate, err := loadBankUnitsForPaths(bankDir, opts.SourcePaths)
	if err != nil {
		return BabbleSummary{}, err
	}
	if len(units) < 4 {
		return BabbleSummary{}, fmt.Errorf("too few usable units in %s: %d", bankDir, len(units))
	}

	baseF0 := opts.BaseF0Hz
	if baseF0 <= 0 {
		baseF0, err = estimateBankBaseF0(ctx, units, sampleRate, worldPath)
		if err != nil {
			return BabbleSummary{}, err
		}
	}

	mode, phrases := detectTopologyAndBuildPhrases(units, targetSec, opts.Seed)
	singleCV := mode == topologyCV
	renderer := worldrender.NewRenderer()
	releaseMS := 80.0
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return BabbleSummary{}, err
	}

	summary := BabbleSummary{
		Bank:            bankDir,
		Mode:            mode,
		Topology:        mode,
		TotalOtoEntries: totalEntries,
		UsableUnits:     len(units),
		SampleRate:      sampleRate,
		BaseF0Hz:        math.Round(baseF0*10) / 10,
	}
	var renderedSec float64
	for pIdx, steps := range phrases {
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		timings, phraseStartMS := computeOpenUtauTimings(steps, singleCV)
		leadingMS := math.Max(0, -phraseStartMS)
		totalDurMS := stepStartMS(steps, len(steps))
		f0Curve := buildConversationalF0Curve(baseF0, totalDurMS+releaseMS+leadingMS, pIdx)

		var worldUnits []worldrender.Unit
		cursor := 0.0
		unitNum := 0
		for i, st := range steps {
			startMS := cursor
			cursor += st.DurationMS
			if st.Pause || st.Unit == nil {
				continue
			}
			u := st.Unit
			t := timings[i]
			origPre := u.Entry.Preutterance
			if singleCV && u.Entry.Overlap <= 0 && origPre <= 15 && origPre < 18.0 {
				origPre = 18.0
			}
			skipMS := math.Max(0, origPre-t.Preutter)
			env := openUtauEnvelope(st.DurationMS, t)
			posMS := startMS - t.Preutter + leadingMS
			lengthMS := env[4].XMS - env[0].XMS
			fadeInMS := env[1].XMS - env[0].XMS
			fadeOutMS := env[4].XMS - env[3].XMS
			if posMS < 0 {
				trim := -posMS
				skipMS += trim
				lengthMS -= trim
				posMS = 0
			}
			if lengthMS < 10 {
				lengthMS = 10
			}
			durCorr := t.Preutter - t.TailIntrude + t.TailOverlap
			reqLen := math.Max(math.Max(st.DurationMS+durCorr, env[4].XMS+t.Preutter)+skipMS, u.Entry.Fixed)
			reqLen = math.Ceil(reqLen/50.0+0.5) * 50.0

			// Keep energy steady inside conversational dynamic range (avoid near-silent drops).
			eWave := math.Sin(2 * math.Pi * float64(unitNum+1) / float64(max(4, len(steps))))
			energy := 0.94 + 0.12*eWave
			unitNum++

			cacheKey := fmt.Sprintf("%s|%.4f|%.4f|%.4f|fs=%d",
				u.Entry.Filename, u.Entry.Offset, u.Entry.Fixed, u.Entry.Blank, sampleRate)
			worldUnits = append(worldUnits, worldrender.Unit{
				CacheKey:          cacheKey,
				Source:            u.Entry.Filename,
				PositionMS:        posMS,
				SkipMS:            skipMS,
				LengthMS:          lengthMS,
				FadeInMS:          math.Max(2.0, fadeInMS),
				FadeOutMS:         math.Max(2.0, fadeOutMS),
				OffsetMS:          u.Entry.Offset,
				RequiredLengthMS:  reqLen,
				ConsonantMS:       u.Entry.Fixed,
				CutoffMS:          u.Entry.Blank,
				ConsonantVelocity: 100.0,
				Volume:            100.0,
				EnergyFactor:      energy,
				Envelope:          env,
			})
		}
		if len(worldUnits) == 0 {
			continue
		}
		phrase := worldrender.Phrase{
			SampleRate: sampleRate,
			F0Curve:    f0Curve,
			Units:      worldUnits,
		}
		pcm, _, err := renderer.Render(ctx, phrase, worldPath)
		if err != nil {
			return summary, fmt.Errorf("render babble phrase %d (%s): %w", pIdx, mode, err)
		}
		if err := ctx.Err(); err != nil {
			return summary, err
		}
		if len(pcm.Data) == 0 {
			return summary, errors.New("empty rendered PCM")
		}
		outPath := filepath.Join(outDir, fmt.Sprintf("babble_%03d.wav", summary.Phrases))
		if err := wav.WriteFile(outPath, normalizedPCM16(monoSamples(pcm), pcm.SampleRate, 0.10)); err != nil {
			return summary, err
		}
		summary.Files = append(summary.Files, outPath)
		summary.Phrases++
		renderedSec += float64(len(pcm.Data)/pcm.Channels) / float64(pcm.SampleRate)
		summary.TotalSec = math.Round(renderedSec*100) / 100
		if opts.Progress != nil {
			opts.Progress(summary.Phrases, len(phrases))
		}
	}

	return summary, ctx.Err()
}

// BabbleEntries returns one pool entry for each written phrase.
func BabbleEntries(summary BabbleSummary) []Entry {
	entries := make([]Entry, len(summary.Files))
	for i, path := range summary.Files {
		entries[i] = Entry{Path: path, Layer: LayerBabble}
	}
	return entries
}
