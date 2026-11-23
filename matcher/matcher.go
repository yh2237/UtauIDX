package matcher

import (
	"fmt"
	"math"
	"sync"

	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
	"github.com/yh2237/gograd/tensor"
)

const matchBlockRows = 32

// derivedMu protects lazy initialization for indices assembled in memory.
var derivedMu sync.Mutex

type Options struct {
	Profile            *AutoProfile
	ChannelWeights     map[string]float64 // replaces the profile weights when nonempty
	ContinuityBonus    *float64
	ContextWeight      *float64
	GlitchScale        *float64
	TailPenalty        float64
	TopK               int
	RankDecay          float64
	ExtraFilePenalties []float32
	LayerPenalties     map[speechpool.Layer]float32
	FramePenalties     []float32
	SimTemp            float64
	CandCepSigma       float64
}

// SelectStats describes the rank-one path. Frames and MeanRunFrames use 5 ms
// WORLD rows. SelectedFiles and LayerFrames use 20 ms feature frames.
type SelectStats struct {
	Frames, Runs, Switches, UniqueUnits int
	MeanRunFrames                       float64
	SelectedFiles                       []int32
	LayerFrames                         map[speechpool.Layer]int
}

// Result holds ranked candidates, scores and normalized blend weights per 20 ms frame.
type Result struct {
	Profile AutoProfile
	Frames  int
	Nearest [][]int
	Scores  [][]float32
	Weights [][]float64
	Stats   SelectStats
}

type matchChannel struct {
	packed *tensor.PackedB
	values []float32
	dims   int
	weight float32
}

// Match scores frames in blocks while preserving continuity across block boundaries.
func Match(idx *index.Index, query Query, opts Options) (*Result, error) {
	if err := idx.Validate(); err != nil {
		return nil, err
	}
	if query.Frames <= 0 || query.Frames > math.MaxInt/worldcodec.SubFrames {
		return nil, fmt.Errorf("matcher: invalid query frame count %d", query.Frames)
	}
	if query.Voiced != nil && len(query.Voiced) != query.Frames {
		return nil, fmt.Errorf("matcher: voicing length %d, want %d", len(query.Voiced), query.Frames)
	}
	derivedMu.Lock()
	if len(idx.Voiced) != idx.Frames || len(idx.MouthStatic) != idx.Frames*worldcodec.MouthDims ||
		len(idx.MouthDelta) != idx.Frames*worldcodec.MouthDims {
		idx.EnsureDerived()
	}
	derivedMu.Unlock()

	var profile AutoProfile
	if opts.Profile == nil {
		profile = ComputeAutoProfile(idx, query)
	} else {
		profile = *opts.Profile
	}
	weights := profile.ChannelWeights
	if len(opts.ChannelWeights) != 0 {
		weights = opts.ChannelWeights
	}
	effective := make(map[string]float64, len(weights))
	for id, w := range weights {
		if !finiteNonnegative(w) {
			return nil, fmt.Errorf("matcher: invalid weight for channel %q", id)
		}
		if w > 0 {
			effective[id] = w
		}
	}
	ids := sortedChannelIDs(effective)
	if len(ids) == 0 {
		return nil, fmt.Errorf("matcher: no active channels shared by index and query")
	}
	normalizeWeights(effective, ids)
	channels := make([]matchChannel, 0, len(ids))
	maxDims := 0
	for _, id := range ids {
		channel, err := resolveChannel(idx, query, id)
		if err != nil {
			return nil, err
		}
		channel.weight = float32(effective[id])
		maxDims = max(maxDims, channel.dims)
		channels = append(channels, channel)
	}
	profile.ChannelWeights = effective
	if opts.ContinuityBonus != nil {
		profile.ContinuityBonus = *opts.ContinuityBonus
	}
	if opts.ContextWeight != nil {
		profile.ContextWeight = *opts.ContextWeight
	}
	if opts.GlitchScale != nil {
		profile.GlitchScale = *opts.GlitchScale
	}
	for _, opt := range []struct {
		name  string
		value float64
	}{
		{"continuity bonus", profile.ContinuityBonus}, {"context weight", profile.ContextWeight},
		{"glitch scale", profile.GlitchScale}, {"tail penalty", opts.TailPenalty},
		{"rank decay", opts.RankDecay}, {"similarity temperature", opts.SimTemp},
		{"candidate cepstrum sigma", opts.CandCepSigma},
	} {
		if !finiteNonnegative(opt.value) {
			return nil, fmt.Errorf("matcher: invalid %s", opt.name)
		}
	}
	if opts.TopK < 0 || opts.RankDecay > 1 {
		return nil, fmt.Errorf("matcher: invalid top K or rank decay")
	}
	topK := min(opts.TopK, DefaultTopK)
	if topK == 0 {
		topK = DefaultTopK
	}
	if opts.RankDecay == 0 {
		opts.RankDecay = DefaultRankDecay
	}

	result := &Result{
		Profile: profile, Frames: query.Frames,
		Nearest: make([][]int, query.Frames),
		Scores:  make([][]float32, query.Frames),
		Weights: make([][]float64, query.Frames),
	}
	n := idx.Frames
	blockRows := min(query.Frames, matchBlockRows+2*ContextFrames)
	if n > math.MaxInt/4/blockRows || maxDims > math.MaxInt/4/blockRows {
		return nil, fmt.Errorf("matcher: similarity block is too large")
	}
	sim := make([]float32, blockRows*n)
	chanSim := make([]float32, len(sim))
	norm := make([]float32, blockRows*maxDims)
	penalty := framePenalties(idx, opts, float32(profile.GlitchScale))
	var decay []float32
	if opts.TailPenalty > 0 {
		decay = bankDecay(idx)
	}
	midCep := candidateCepstra(idx, opts)
	var previous []int
	for start := 0; start < query.Frames; start += matchBlockRows {
		end := min(start+matchBlockRows, query.Frames)
		first, last := max(0, start-ContextFrames), min(query.Frames, end+ContextFrames)
		rows := last - first
		block := sim[:rows*n]
		clear(block)
		for _, ch := range channels {
			normalizeBlock(norm, ch.values, first, rows, ch.dims)
			tensor.SGEMMPacked(chanSim[:len(block)], norm[:rows*ch.dims], ch.packed, rows, 1)
			for i := range block {
				block[i] += ch.weight * chanSim[i]
			}
		}
		for t := first; t < last; t++ {
			row := block[(t-first)*n : (t-first+1)*n]
			active := len(decay) != 0 && activeLevel(query.Level5ms, t)
			for j := range row {
				if query.Voiced != nil && query.Voiced[t] != idx.Voiced[j] {
					row[j] -= DefaultVoicingPenalty
				}
				row[j] -= penalty[j]
				if active {
					row[j] -= float32(opts.TailPenalty) * decay[j]
				}
			}
		}
		for t := start; t < end; t++ {
			chosen, scores := topCandidates(idx, block, first, last, t, topK, previous, profile)
			result.Nearest[t], result.Scores[t] = chosen, scores
			result.Weights[t] = blendWeights(chosen, scores, midCep, opts)
			previous = chosen
		}
	}
	result.Stats = selectionStats(idx, result.Nearest)
	return result, nil
}

func finiteNonnegative(x float64) bool {
	return x >= 0 && !math.IsInf(x, 0)
}

func resolveChannel(idx *index.Index, query Query, id string) (matchChannel, error) {
	var ch matchChannel
	switch id {
	case ChannelMouthStatic:
		ch = matchChannel{packed: idx.PackedMouthStatic(), values: query.MouthStatic, dims: worldcodec.MouthDims}
	case ChannelMouthDelta:
		ch = matchChannel{packed: idx.PackedMouthDelta(), values: query.MouthDelta, dims: worldcodec.MouthDims}
	default:
		ch.packed, _ = idx.PackedChannel(id)
		ch.values, ch.dims = query.Channels[id], idx.ChannelDims(id)
	}
	if ch.packed == nil || ch.dims <= 0 || query.Frames > len(ch.values)/ch.dims {
		return matchChannel{}, fmt.Errorf("matcher: missing or short active channel %q", id)
	}
	return ch, nil
}

func framePenalties(idx *index.Index, opts Options, glitchScale float32) []float32 {
	filePen := make([]float32, len(idx.Files))
	for i, entry := range idx.Files {
		filePen[i] = entry.Penalty + opts.LayerPenalties[entry.Layer]
		if i < len(opts.ExtraFilePenalties) {
			filePen[i] += opts.ExtraFilePenalties[i]
		}
	}
	penalty := make([]float32, idx.Frames)
	for j, id := range idx.FileID {
		if id >= 0 && int(id) < len(filePen) {
			penalty[j] = filePen[id]
		}
		penalty[j] += glitchScale * idx.GlitchPenalty[j]
		if j < len(opts.FramePenalties) {
			penalty[j] += opts.FramePenalties[j]
		}
	}
	return penalty
}

func activeLevel(levels []float64, frame int) bool {
	if frame >= len(levels)/worldcodec.SubFrames {
		return true
	}
	var level float64
	for _, v := range levels[frame*worldcodec.SubFrames : (frame+1)*worldcodec.SubFrames] {
		level += v
	}
	return level/float64(worldcodec.SubFrames) > -24
}

func bankDecay(idx *index.Index) []float32 {
	type fileVoicing struct {
		id     int32
		voiced bool
	}
	c0 := make([]float64, idx.Frames)
	groups := make(map[fileVoicing][]float64)
	for j, id := range idx.FileID {
		for sub := range worldcodec.SubFrames {
			c0[j] += float64(idx.Spectrum[(j*worldcodec.SubFrames+sub)*worldcodec.SpectrumDims])
		}
		c0[j] /= worldcodec.SubFrames
		key := fileVoicing{id, idx.Voiced[j]}
		groups[key] = append(groups[key], c0[j])
	}
	peaks := make(map[fileVoicing]float64, len(groups))
	for key, values := range groups {
		peaks[key] = worldcodec.Percentile(values, 90)
	}
	decay := make([]float32, idx.Frames)
	for j, id := range idx.FileID {
		decay[j] = float32(min(3.0, max(0, peaks[fileVoicing{id, idx.Voiced[j]}]-c0[j]-0.6)))
	}
	return decay
}

func topCandidates(idx *index.Index, block []float32, first, last, t, topK int, previous []int, profile AutoProfile) ([]int, []float32) {
	n := idx.Frames
	best := make([]int, 0, topK)
	scores := make([]float32, 0, topK)
	cw, bonus := float32(profile.ContextWeight), float32(profile.ContinuityBonus)
	for j, id := range idx.FileID {
		score := block[(t-first)*n+j]
		if cw > 0 {
			var total, count float32
			for delta := -ContextFrames; delta <= ContextFrames; delta++ {
				tt, jj := t+delta, j+delta
				if tt < first || tt >= last || jj < 0 || jj >= n || idx.FileID[jj] != id {
					continue
				}
				w := float32(1)
				if delta != 0 {
					w = cw / float32(max(delta, -delta))
				}
				total += w * block[(tt-first)*n+jj]
				count += w
			}
			score = total / count
		}
		if bonus != 0 {
			for _, p := range previous {
				if continues(idx, p, j) {
					score += bonus
					break
				}
			}
		}
		pos := len(best)
		if pos == topK {
			if score <= scores[pos-1] {
				continue
			}
			pos--
		} else {
			best, scores = append(best, 0), append(scores, 0)
		}
		for pos > 0 && scores[pos-1] < score {
			best[pos], scores[pos] = best[pos-1], scores[pos-1]
			pos--
		}
		best[pos], scores[pos] = j, score
	}
	return best, scores
}

func continues(idx *index.Index, previous, next int) bool {
	return idx.FileID[next] == idx.FileID[previous] &&
		int64(idx.FramePosition(next)) == int64(idx.FramePosition(previous))+1
}

const candidateCepDims = 15

func candidateCepstra(idx *index.Index, opts Options) []float32 {
	if opts.CandCepSigma <= 0 || opts.SimTemp > 0 {
		return nil
	}
	cep := make([]float32, idx.Frames*candidateCepDims)
	for frame := range idx.Frames {
		for sub := range worldcodec.SubFrames {
			off := (frame*worldcodec.SubFrames+sub)*worldcodec.SpectrumDims + 1
			for d, v := range idx.Spectrum[off : off+candidateCepDims] {
				cep[frame*candidateCepDims+d] += v / worldcodec.SubFrames
			}
		}
	}
	return cep
}

func blendWeights(chosen []int, scores []float32, cep []float32, opts Options) []float64 {
	weights := make([]float64, len(chosen))
	var total float64
	for k, j := range chosen {
		w := math.Pow(opts.RankDecay, float64(k))
		if opts.SimTemp > 0 {
			w = math.Exp(float64(scores[k]-scores[0]) / opts.SimTemp)
		} else if k > 0 && opts.CandCepSigma > 0 && len(cep) > 0 {
			var ss float64
			for d := range candidateCepDims {
				diff := float64(cep[j*candidateCepDims+d] - cep[chosen[0]*candidateCepDims+d])
				ss += diff * diff
			}
			if ss > 0 {
				w *= math.Exp(-0.5 * ss / (opts.CandCepSigma * opts.CandCepSigma))
			}
		}
		weights[k] = w
		total += w
	}
	for k := range weights {
		weights[k] /= total
	}
	return weights
}

func selectionStats(idx *index.Index, nearest [][]int) SelectStats {
	stats := SelectStats{SelectedFiles: make([]int32, 0, len(nearest)), LayerFrames: make(map[speechpool.Layer]int)}
	units := make(map[int32]struct{})
	previous := -1
	for _, chosen := range nearest {
		j := chosen[0]
		id := idx.FileID[j]
		stats.SelectedFiles = append(stats.SelectedFiles, id)
		units[id] = struct{}{}
		stats.Frames += worldcodec.SubFrames
		if id >= 0 && int(id) < len(idx.Files) {
			stats.LayerFrames[idx.Files[id].Layer]++
		}
		if previous < 0 {
			stats.Runs++
		} else if !continues(idx, previous, j) {
			stats.Runs++
			stats.Switches++
		}
		previous = j
	}
	stats.UniqueUnits = len(units)
	if stats.Runs > 0 {
		stats.MeanRunFrames = float64(stats.Frames) / float64(stats.Runs)
	}
	return stats
}
