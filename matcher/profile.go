// Package matcher selects speech-index frames for 20 ms reference features.
package matcher

import (
	"math"
	"slices"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/gograd/tensor"
)

const (
	ChannelMouthStatic = "mouth-static"
	ChannelMouthDelta  = "mouth-delta"

	DefaultContinuityBonus = 0.06
	DefaultTopK            = 4
	DefaultRankDecay       = 0.45
	DefaultVoicingPenalty  = 0.5
	ContextFrames          = 2
)

// Query holds row-major features on the 20 ms grid and optional 5 ms levels.
type Query struct {
	Frames      int
	Channels    map[string][]float32
	MouthStatic []float32 // 12 values per frame
	MouthDelta  []float32 // 12 values per frame
	Voiced      []bool    // nil disables the voicing penalty
	Level5ms    []float64 // dB relative to p95
}

type AutoProfile struct {
	Cos1W6          float64
	Cos1WH          float64
	SimGap          float64
	SizeGap         float64
	Gap             float64
	Xling           float64
	IsSinusoidal    bool
	ChannelWeights  map[string]float64
	ContextWeight   float64
	GlitchScale     float64
	ContinuityBonus float64
}

// ComputeCos1 returns the mean of the highest bank cosine for each query row.
// packed must contain indexFrames unit-normalized rows of width dims.
func ComputeCos1(packed *tensor.PackedB, indexFrames, dims int, queryVals []float32, queryFrames int) float64 {
	if packed == nil || indexFrames <= 0 || queryFrames <= 0 || dims <= 0 || queryFrames > len(queryVals)/dims {
		return 0
	}
	const blockRows = 64
	norm := make([]float32, min(queryFrames, blockRows)*dims)
	sim := make([]float32, min(queryFrames, blockRows)*indexFrames)
	var total float64
	for first := 0; first < queryFrames; first += blockRows {
		rows := min(blockRows, queryFrames-first)
		normalizeBlock(norm, queryVals, first, rows, dims)
		tensor.SGEMMPacked(sim[:rows*indexFrames], norm[:rows*dims], packed, rows, 1)
		for row := range rows {
			total += float64(slices.Max(sim[row*indexFrames : (row+1)*indexFrames]))
		}
	}
	return total / float64(queryFrames)
}

func normalizeBlock(dst, src []float32, first, rows, dims int) {
	for row := range rows {
		start := (first + row) * dims
		worldcodec.NormalizeInto(dst[row*dims:(row+1)*dims], src[start:start+dims])
	}
}

// ComputeAutoProfile estimates the domain gap and assigns weights to shared features.
func ComputeAutoProfile(idx *index.Index, query Query) AutoProfile {
	p := AutoProfile{ContinuityBonus: DefaultContinuityBonus, ChannelWeights: make(map[string]float64)}
	if idx == nil || idx.Frames <= 0 || query.Frames <= 0 {
		return p
	}
	var shared []string
	for _, ch := range idx.Header.Channels {
		if ch.ID != ChannelMouthStatic && ch.ID != ChannelMouthDelta && hasNeuralChannel(idx, query, ch.ID) {
			shared = append(shared, ch.ID)
		}
	}
	slices.Sort(shared)
	primary := encoder.IDWavLMBasePlusL6
	if !slices.Contains(shared, primary) {
		primary = ""
		if len(shared) > 0 {
			primary = shared[0]
		}
	}
	cos1 := func(id string) float64 {
		packed, ok := idx.PackedChannel(id)
		if !ok {
			return 0
		}
		return ComputeCos1(packed, idx.Frames, idx.ChannelDims(id), query.Channels[id], query.Frames)
	}
	if primary != "" {
		p.Cos1W6 = cos1(primary)
	}
	p.SimGap = clip01((0.57 - p.Cos1W6) / 0.21)
	p.SizeGap = clip01((4.0 - math.Log10(float64(max(idx.Frames, 500)))) / 0.80)
	p.Gap = clip01(p.SimGap + 0.25*p.SizeGap)
	p.IsSinusoidal = idx.C35 > 1.25
	p.ContextWeight = 0.05 * max(0, 1-p.Gap/0.25)
	p.GlitchScale = max(0, 1-p.Gap/0.35)

	es := encoder.IDWav2Vec2EspeakL20
	wh := encoder.IDWhisperSmallL12PosInv
	wEs := 0.44 * p.Gap
	wMouth := 0.20 * math.Pow(p.Gap, 1.5)
	wPrimary := 1 - wEs - wMouth
	if slices.Contains(shared, wh) {
		p.Cos1WH = p.Cos1W6
		if primary != wh {
			p.Cos1WH = cos1(wh)
		}
		if p.Gap < 0.25 {
			p.Xling = clip01((p.Cos1W6 - p.Cos1WH - 0.005) / 0.030)
		}
	}
	wWh := 0.50 * p.Xling
	wPrimary *= 1 - wWh
	wEs *= 1 - wWh
	if slices.Contains(shared, es) {
		p.ChannelWeights[es] = wEs
	} else {
		wPrimary += wEs
	}
	if wWh > 0 {
		p.ChannelWeights[wh] = wWh
	}
	if primary != "" {
		p.ChannelWeights[primary] += wPrimary
	}
	if hasMouth(idx, query) {
		p.ChannelWeights[ChannelMouthStatic] = 0.75 * wMouth
		p.ChannelWeights[ChannelMouthDelta] = 0.25 * wMouth
	}
	for id, w := range p.ChannelWeights {
		if w <= 0 {
			delete(p.ChannelWeights, id)
		}
	}
	ids := sortedChannelIDs(p.ChannelWeights)
	normalizeWeights(p.ChannelWeights, ids)
	return p
}

func clip01(x float64) float64 { return min(max(0, x), 1) }

func hasNeuralChannel(idx *index.Index, query Query, id string) bool {
	dims := idx.ChannelDims(id)
	bank := idx.Channels[id]
	return dims > 0 && len(bank)%dims == 0 && idx.Frames == len(bank)/dims &&
		query.Frames <= len(query.Channels[id])/dims
}

func hasMouth(idx *index.Index, query Query) bool {
	derivedMu.Lock()
	defer derivedMu.Unlock()
	return len(idx.MouthStatic) == idx.Frames*worldcodec.MouthDims &&
		len(idx.MouthDelta) == idx.Frames*worldcodec.MouthDims &&
		query.Frames <= len(query.MouthStatic)/worldcodec.MouthDims &&
		query.Frames <= len(query.MouthDelta)/worldcodec.MouthDims
}

func sortedChannelIDs(weights map[string]float64) []string {
	ids := make([]string, 0, len(weights))
	for id := range weights {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	return ids
}

func normalizeWeights(weights map[string]float64, ids []string) {
	var largest, total float64
	for _, id := range ids {
		largest = max(largest, weights[id])
		total += weights[id]
	}
	if math.IsInf(total, 1) {
		total = 0
		for _, id := range ids {
			weights[id] /= largest
			total += weights[id]
		}
	}
	for _, id := range ids {
		weights[id] /= total
	}
}
