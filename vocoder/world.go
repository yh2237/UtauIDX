package vocoder

import (
	"fmt"
	"math"
	"slices"

	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/matcher"
)

const (
	maxQuietCutDB = 80.0
	levelTrackDB  = 10.0
)

type WorldParameters struct {
	F0, Spectrum, Aperiodicity []float64
}

type WorldVocoder struct {
	World index.World
}

var _ Vocoder = (*WorldVocoder)(nil)

func (v *WorldVocoder) Synthesize(idx *index.Index, match *matcher.Result, req SynthRequest) ([]float64, error) {
	if v == nil || v.World == nil {
		return nil, fmt.Errorf("vocoder: missing WORLD engine")
	}
	params, err := v.Parameters(idx, match, req)
	if err != nil {
		return nil, err
	}
	return v.World.Synthesize(params.F0, params.Spectrum, params.Aperiodicity, idx.FFTSize, idx.SampleRate, worldcodec.WorldFrameMS)
}

// Parameters decodes the selected bank frames into WORLD synthesis parameters.
func (v *WorldVocoder) Parameters(idx *index.Index, match *matcher.Result, req SynthRequest) (WorldParameters, error) {
	if err := idx.Validate(); err != nil {
		return WorldParameters{}, err
	}
	bins := idx.FFTSize/2 + 1
	if match == nil || match.Frames <= 0 || match.Frames != len(match.Nearest) ||
		match.Frames > math.MaxInt/8/worldcodec.SubFrames/max(bins, worldcodec.SpectrumDims, idx.APBands) {
		return WorldParameters{}, fmt.Errorf("vocoder: invalid match frame count")
	}
	rows := match.Frames * worldcodec.SubFrames
	if err := validateRequest(idx, match, req, rows); err != nil {
		return WorldParameters{}, err
	}

	cfg := AutoSynthConfig(match.Profile)
	if req.Config != nil {
		cfg = *req.Config
	}
	if err := validateConfig(cfg); err != nil {
		return WorldParameters{}, err
	}
	beta := cfg.PostfilterBeta
	if req.Unblended && (req.Config == nil || beta <= 0) {
		beta = 0
	} else if beta == 0 {
		beta = 0.40
	}
	beta = max(0, beta)

	referenceF0 := slices.Clone(req.ReferenceF0[:rows])
	if cfg.UseSafeRefAP {
		for row, ap := range req.ReferenceAP {
			if ap >= 0.5 {
				referenceF0[row] = 0
			}
		}
	}
	var outF0 []float64
	if len(req.OutputF0) > 0 {
		outF0 = slices.Clone(req.OutputF0[:rows])
		for row, f0 := range referenceF0 {
			if f0 <= 0 {
				outF0[row] = 0
			}
		}
	} else {
		scale := req.PitchScale
		if scale == 0 {
			scale = 1
		}
		outF0 = OutputF0(referenceF0, idx.F0Median, req.CenterHz, scale)
		for row, hz := range outF0 {
			if !finite(hz) || hz < 0 {
				return WorldParameters{}, fmt.Errorf("vocoder: invalid derived F0 at row %d", row)
			}
		}
	}

	bankVoiced := worldcodec.FrameVoicing(idx.F0)
	spectrum := averageParameters(idx, match, idx.Spectrum, worldcodec.SpectrumDims, outF0, bankVoiced, cfg, req.Unblended)
	ap := averageParameters(idx, match, idx.Aperiodicity, idx.APBands, outF0, bankVoiced, cfg, req.Unblended)
	if !req.Unblended && !cfg.NoSmooth {
		smoothSpectrumEnhanced(spectrum, worldcodec.SpectrumDims, outF0, cfg.UnvoicedCFMS > 0)
		smoothVoicingAware(ap, idx.APBands, outF0, false)
	}
	if cfg.APCliffGuard {
		guardVoicedAPCliff(ap, idx.APBands, outF0, req.ReferenceAP)
	}
	smoothHighOrderVoiced(spectrum, worldcodec.SpectrumDims, outF0, 24, cfg.HighOrderSmooth)
	enhanceModulation(spectrum, worldcodec.SpectrumDims, outF0, req.ReferenceLevel, cfg.ModBoost)

	c := worldcodec.New(idx.SampleRate, idx.FFTSize, worldcodec.SpectrumDims)
	before := make([]float64, bins)
	quietDB := cfg.QuietLevelDB
	if quietDB <= 0 {
		quietDB = 40
	}
	cut := quietCut(req.ReferenceLevel, rows, quietDB)
	if !cfg.SoftKneeLevel {
		trackLevelWithSupportAndTrack(cut, spectrum, req.ReferenceLevel, c, before, cfg.MaxSupportDB, cfg.LevelTrackDB)
	}
	presence := presenceCurve(bins, idx.SampleRate, cfg.PresenceDB)

	params := WorldParameters{
		F0: outF0, Spectrum: make([]float64, rows*bins), Aperiodicity: make([]float64, rows*bins),
	}
	var sharpened [worldcodec.SpectrumDims]float64
	for row, hz := range outF0 {
		start := row * worldcodec.SpectrumDims
		coded := spectrum[start : start+worldcodec.SpectrumDims]
		c.DecodeSpectrum(coded, before)
		after := params.Spectrum[row*bins : (row+1)*bins]
		copy(sharpened[:], coded)
		applyPostfilter(sharpened[:], beta, cfg.BetaMode)
		c.DecodeSpectrum(sharpened[:], after)
		for b, gain := range presence {
			after[b] *= gain
		}
		ratio := sum(before) / sum(after) * math.Pow(10, -cut[row]/10)
		for b := range after {
			after[b] *= ratio
		}

		apRow := params.Aperiodicity[row*bins : (row+1)*bins]
		if hz == 0 {
			for b := range apRow {
				apRow[b] = 1
			}
		} else {
			c.DecodeAperiodicity(ap[row*idx.APBands:(row+1)*idx.APBands], apRow)
		}
	}
	if cfg.SoftKneeLevel {
		applySoftKneeLevel(params.Spectrum, bins, outF0, req.ReferenceLevel, cfg.MaxSupportDB)
	}
	applyVoicedBodySupport(params.Spectrum, bins, idx.SampleRate, outF0, req.ReferenceLevel, cfg.BodySupportDB)
	for range cfg.SpectrumSmooth {
		smoothVoicingAware(params.Spectrum, bins, outF0, false)
	}
	return params, nil
}

func validateRequest(idx *index.Index, match *matcher.Result, req SynthRequest, rows int) error {
	if len(req.ReferenceF0) < rows || (len(req.OutputF0) > 0 && len(req.OutputF0) < rows) {
		return fmt.Errorf("vocoder: F0 does not cover %d WORLD rows", rows)
	}
	if len(req.ReferenceAP) > 0 && len(req.ReferenceAP) != rows {
		return fmt.Errorf("vocoder: reference AP length %d, want %d", len(req.ReferenceAP), rows)
	}
	for row, ap := range req.ReferenceAP {
		if !finite(float64(ap)) || ap < 0 || ap > 1 {
			return fmt.Errorf("vocoder: invalid reference AP at row %d", row)
		}
	}
	for _, f0 := range [][]float64{req.ReferenceF0[:rows], req.OutputF0[:min(rows, len(req.OutputF0))]} {
		for row, hz := range f0 {
			if !finite(hz) {
				return fmt.Errorf("vocoder: invalid F0 at row %d", row)
			}
		}
	}
	for row, hz := range req.OutputF0[:min(rows, len(req.OutputF0))] {
		if hz < 0 {
			return fmt.Errorf("vocoder: negative output F0 at row %d", row)
		}
	}
	if !finite(req.CenterHz) || !finite(req.PitchScale) {
		return fmt.Errorf("vocoder: invalid center frequency or pitch scale")
	}
	if len(req.ReferenceLevel) > 0 && len(req.ReferenceLevel) < rows {
		return fmt.Errorf("vocoder: reference levels do not cover %d WORLD rows", rows)
	}
	for row, db := range req.ReferenceLevel[:min(rows, len(req.ReferenceLevel))] {
		if !finite(db) {
			return fmt.Errorf("vocoder: invalid reference level at row %d", row)
		}
	}
	if !req.Unblended && len(match.Weights) != match.Frames {
		return fmt.Errorf("vocoder: match weight frame count does not match candidates")
	}
	for frame, chosen := range match.Nearest {
		if len(chosen) == 0 {
			return fmt.Errorf("vocoder: no candidate at frame %d", frame)
		}
		for _, j := range chosen {
			if j < 0 || j >= idx.Frames {
				return fmt.Errorf("vocoder: invalid candidate %d at frame %d", j, frame)
			}
		}
		if req.Unblended {
			continue
		}
		weights := match.Weights[frame]
		if len(weights) != len(chosen) {
			return fmt.Errorf("vocoder: candidate weights do not match at frame %d", frame)
		}
		for _, w := range weights {
			if !finite(w) || w < 0 {
				return fmt.Errorf("vocoder: invalid candidate weight at frame %d", frame)
			}
		}
		if math.Abs(sum(weights)-1) > 1e-6 {
			return fmt.Errorf("vocoder: candidate weights must sum to one at frame %d", frame)
		}
	}
	return nil
}

func validateConfig(cfg SynthConfig) error {
	switch cfg.BetaMode {
	case "", BetaFull, BetaLowOrder, BetaMidOrder:
	default:
		return fmt.Errorf("vocoder: unknown beta mode %q", cfg.BetaMode)
	}
	for _, value := range []float64{
		cfg.BodySupportDB, cfg.LevelTrackDB, cfg.MaxSupportDB, cfg.CrossfadeMS,
		cfg.UnvoicedCFMS, cfg.PostfilterBeta, cfg.ModBoost, cfg.PresenceDB, cfg.QuietLevelDB,
	} {
		if !finite(value) {
			return fmt.Errorf("vocoder: non-finite synthesis configuration")
		}
	}
	if cfg.MaxSupportDB < 0 || cfg.HighOrderSmooth < 0 || cfg.SpectrumSmooth < 0 {
		return fmt.Errorf("vocoder: negative support limit or smoothing pass count")
	}
	return nil
}

func averageParameters(idx *index.Index, match *matcher.Result, values []float32, dims int, outF0 []float64, bankVoiced []bool, cfg SynthConfig, unblended bool) []float64 {
	if dims == 0 {
		return nil
	}
	const subFrames, hop, frameMS = worldcodec.SubFrames, worldcodec.WorldFrameMS, worldcodec.FrameMS
	cfMS := cfg.CrossfadeMS
	if cfMS == 0 {
		cfMS = 30
	} else if cfMS < 0 {
		cfMS = hop
	}
	reach := int(min(math.Ceil(cfMS/frameMS)+1, float64(match.Frames-1)))
	out := make([]float64, len(outF0)*dims)
	for row, hz := range outF0 {
		t, sub := row/subFrames, row%subFrames
		localCF := cfMS
		if hz == 0 && cfg.UnvoicedCFMS > 0 {
			localCF = min(localCF, cfg.UnvoicedCFMS)
		}
		at := (float64(sub)+0.5)*hop - frameMS/2
		target := out[row*dims : (row+1)*dims]
		var total float64
		current := match.Nearest[t]
		if unblended {
			current = current[:1]
		}
		for f := max(0, t-reach); f <= min(match.Frames-1, t+reach); f++ {
			weight := 1 - math.Abs(float64(t-f)*frameMS+at)/localCF
			if weight <= 0 {
				continue
			}
			chosen := match.Nearest[f]
			if unblended {
				chosen = chosen[:1]
			}
			for k, j := range chosen {
				source, s := j, sub
				if delta := t - f; delta != 0 {
					next := j + delta
					if next >= 0 && next < idx.Frames && (delta > 0 || slices.Contains(current, next)) &&
						idx.FileID[next] == idx.FileID[j] &&
						int64(idx.FramePosition(next)) == int64(idx.FramePosition(j))+int64(delta) &&
						bankVoiced[next] == bankVoiced[j] {
						source = next
					} else if delta > 0 {
						s = subFrames - 1
					} else {
						s = 0
					}
				}
				w := weight
				if !unblended {
					w *= match.Weights[f][k]
				}
				start := (source*subFrames + s) * dims
				for d, val := range values[start : start+dims] {
					target[d] += float64(val) * w
				}
				total += w
			}
		}
		if total > 0 {
			for d := range target {
				target[d] /= total
			}
		} else {
			// A narrow crossfade can leave edge rows outside every triangle.
			for k, j := range current {
				w := 1.0
				if !unblended {
					w = match.Weights[t][k]
				}
				start := (j*subFrames + sub) * dims
				for d, val := range values[start : start+dims] {
					target[d] += float64(val) * w
				}
			}
		}
	}
	return out
}

func applyPostfilter(coded []float64, beta float64, mode BetaMode) {
	if beta == 0 {
		return
	}
	end := len(coded)
	if mode == BetaLowOrder || mode == BetaMidOrder {
		end = min(end, 24)
	}
	for k := 2; k < end; k++ {
		weight := 1.0
		switch mode {
		case BetaMidOrder:
			if k < 6 {
				weight = 0.25 * float64(k-1) / 5
			} else if k > 16 {
				weight = 0.5 * (1 + math.Cos(math.Pi*float64(k-16)/8))
			}
		case BetaLowOrder:
			if k > 18 {
				weight = 0.5 * (1 + math.Cos(math.Pi*float64(k-18)/6))
			}
		}
		coded[k] *= 1 + beta*weight
	}
}

func presenceCurve(bins, sampleRate int, presenceDB float64) []float64 {
	if presenceDB == 0 {
		return nil
	}
	curve := make([]float64, bins)
	for b := range bins {
		hz := float64(b) * float64(sampleRate) / 2 / float64(bins-1)
		var bell float64
		if hz > 800 && hz < 7000 {
			l := math.Log2(hz/2500) / 0.85
			bell = math.Exp(-0.5 * l * l)
		}
		curve[b] = math.Pow(10, presenceDB*bell/10)
	}
	return curve
}

func quietCut(levels []float64, rows int, quietDB float64) []float64 {
	cut := make([]float64, rows)
	if len(levels) < rows {
		return cut
	}
	for row := range rows {
		if levels[row] < -quietDB {
			cut[row] = math.Min(-quietDB-levels[row], maxQuietCutDB)
		}
	}
	return worldcodec.GaussianFilter(cut, 2)
}

func trackLevelWithSupportAndTrack(cut, spectrum, levels []float64, c *worldcodec.Codec, scratch []float64, maxSupportDB, customTrackDB float64) {
	rows := len(cut)
	if len(levels) < rows {
		return
	}
	outLevel := make([]float64, rows)
	for row := range rows {
		c.DecodeSpectrum(spectrum[row*worldcodec.SpectrumDims:(row+1)*worldcodec.SpectrumDims], scratch)
		outLevel[row] = 10 * math.Log10(sum(scratch)+1e-30)
	}
	peak := worldcodec.Percentile(outLevel, 95)
	ltDB := levelTrackDB
	if customTrackDB > 0 {
		ltDB = customTrackDB
	}
	extra := make([]float64, rows)
	for row := range rows {
		if levels[row] < -ltDB {
			extra[row] = math.Max(0, outLevel[row]-peak-levels[row])
		}
	}
	extra = worldcodec.GaussianFilter(extra, 2)
	var support []float64
	if maxSupportDB > 0 {
		support = make([]float64, rows)
		for row := range rows {
			relOut := outLevel[row] - peak
			if levels[row] > -32 && relOut > -42 {
				deficit := levels[row] - relOut - 2
				if deficit > 0 {
					support[row] = math.Min(deficit*0.80, maxSupportDB)
				}
			}
		}
		support = worldcodec.GaussianFilter(support, 3)
	}
	for row := range rows {
		s := 0.0
		if support != nil {
			s = support[row]
		}
		cut[row] = math.Max(-maxSupportDB, math.Min(cut[row]+extra[row]-s, maxQuietCutDB))
	}
}

// applySoftKneeLevel gently compresses voiced energy sag upward while keeping quiet gating intact.
func applySoftKneeLevel(decodedSpectrum []float64, bins int, outF0, levels []float64, maxSupportDB float64) {
	rows := len(outF0)
	if rows == 0 || bins <= 0 || len(levels) < rows || len(decodedSpectrum) < rows*bins {
		return
	}
	outDB := make([]float64, rows)
	var voicedDB []float64
	for r := range rows {
		outDB[r] = 10 * math.Log10(sum(decodedSpectrum[r*bins:(r+1)*bins])+1e-30)
		if outF0[r] > 0 && levels[r] > -25 {
			voicedDB = append(voicedDB, outDB[r])
		}
	}
	if len(voicedDB) < 3 {
		for r := range rows {
			if levels[r] > -25 {
				voicedDB = append(voicedDB, outDB[r])
			}
		}
	}
	if len(voicedDB) == 0 {
		return
	}
	outPeak := worldcodec.Percentile(voicedDB, 95)
	unvoicedCap := min(maxSupportDB, 3.5)
	gDB := make([]float64, rows)
	for r := range rows {
		relOut := outDB[r] - outPeak
		tgt := levels[r]
		if outF0[r] > 0 && levels[r] > -30 {
			softTgt := math.Max(-11.5, math.Min(0.0, -3.0+0.55*(levels[r]+3.0)))
			if levels[r] < -18 {
				ramp := (levels[r] + 30.0) / 12.0
				tgt = levels[r] + ramp*(softTgt-levels[r])
			} else {
				tgt = softTgt
			}
		}
		if levels[r] > -30 && relOut > -36 && relOut < tgt {
			capDB := unvoicedCap
			if outF0[r] > 0 {
				capDB = maxSupportDB
			}
			gDB[r] = math.Min(0.85*(tgt-relOut), capDB)
		} else if levels[r] < -12 && relOut > tgt {
			gDB[r] = math.Max(-(relOut-tgt)*0.85, -18.0)
		}
	}
	gDB = worldcodec.GaussianFilter(gDB, 3.0)
	for r := range rows {
		scale := math.Pow(10, gDB[r]/10.0)
		spRow := decodedSpectrum[r*bins : (r+1)*bins]
		for b := range bins {
			spRow[b] *= scale
		}
	}
}

func sum(values []float64) float64 {
	var total float64
	for _, v := range values {
		total += v
	}
	return total
}

func smoothSpectrumEnhanced(values []float64, dims int, outF0 []float64, crispUnvoiced bool) {
	if dims <= 0 {
		return
	}
	rows := len(values) / dims
	if rows < 2 {
		return
	}
	original := append([]float64(nil), values...)
	for row := range rows {
		hasOutF0 := len(outF0) > row
		rowVoiced := hasOutF0 && outF0[row] > 0
		idxs := [3]int{max(row-1, 0), row, min(row+1, rows-1)}
		baseW := [3]float64{1, 1, 1}
		if crispUnvoiced && hasOutF0 && !rowVoiced {
			baseW = [3]float64{0.15, 0.70, 0.15}
		}
		target := values[row*dims : (row+1)*dims]
		clear(target)
		var wSum float64
		for i, nbr := range idxs {
			w := baseW[i]
			for d := range dims {
				target[d] += w * original[nbr*dims+d]
			}
			wSum += w
		}
		if wSum > 0 {
			for d := range dims {
				target[d] /= wSum
			}
		}
	}
}

func guardVoicedAPCliff(aperiodicity []float64, bands int, outF0 []float64, refAP []float32) {
	if bands <= 0 {
		return
	}
	rows := len(aperiodicity) / bands
	for row := range rows {
		if row >= len(outF0) || outF0[row] <= 0 {
			continue
		}
		if len(refAP) > row && refAP[row] > 0.30 {
			continue
		}
		rowSlice := aperiodicity[row*bands : (row+1)*bands]
		var mean float64
		for b := range bands {
			mean += rowSlice[b]
		}
		mean /= float64(bands)
		if mean > -0.6 {
			shift := mean - (-1.6)
			for b := range bands {
				rowSlice[b] -= shift
			}
		}
	}
}

func smoothHighOrderVoiced(spectrum []float64, dims int, outF0 []float64, startDim, passes int) {
	if dims <= startDim || passes <= 0 {
		return
	}
	rows := len(spectrum) / dims
	if rows < 2 || len(outF0) < rows {
		return
	}
	scratch := make([]float64, len(spectrum))
	wts := [3]float64{0.25, 0.50, 0.25}
	for range passes {
		copy(scratch, spectrum)
		for row := range rows {
			if outF0[row] <= 0 {
				continue
			}
			idxs := [3]int{max(row-1, 0), row, min(row+1, rows-1)}
			var wSum float64
			for i, nbr := range idxs {
				if outF0[nbr] > 0 {
					wSum += wts[i]
				}
			}
			if wSum <= 0 {
				continue
			}
			target := spectrum[row*dims : (row+1)*dims]
			for k := startDim; k < dims; k++ {
				var acc float64
				for i, nbr := range idxs {
					if outF0[nbr] > 0 {
						acc += wts[i] * scratch[nbr*dims+k]
					}
				}
				target[k] = acc / wSum
			}
		}
	}
}

func smoothVoicingAware(values []float64, dims int, outF0 []float64, isolateVoicing bool) {
	if !isolateVoicing || len(outF0) == 0 {
		smooth(values, dims)
		return
	}
	if dims <= 0 {
		return
	}
	rows := len(values) / dims
	if rows < 2 {
		return
	}
	original := append([]float64(nil), values...)
	for row := range rows {
		rowVoiced := len(outF0) > row && outF0[row] > 0
		idxs := [3]int{max(row-1, 0), row, min(row+1, rows-1)}
		target := values[row*dims : (row+1)*dims]
		clear(target)
		var count float64
		for _, nbr := range idxs {
			if len(outF0) > nbr && (outF0[nbr] > 0) != rowVoiced {
				continue
			}
			for d := range dims {
				target[d] += original[nbr*dims+d]
			}
			count++
		}
		if count > 0 {
			for d := range dims {
				target[d] /= count
			}
		}
	}
}

func maskedGaussianFilter(src []float64, active []bool, sigma float64) []float64 {
	n := len(src)
	dst := make([]float64, n)
	if n == 0 || sigma <= 0 {
		copy(dst, src)
		return dst
	}
	radius := int(math.Ceil(3 * sigma))
	kernel := make([]float64, radius+1)
	for d := range kernel {
		kernel[d] = math.Exp(-0.5 * float64(d*d) / (sigma * sigma))
	}
	for r := range n {
		if r >= len(active) || !active[r] {
			dst[r] = src[r]
			continue
		}
		var num, den float64
		for d := -radius; d <= radius; d++ {
			j := r + d
			if j < 0 || j >= n || j >= len(active) || !active[j] {
				continue
			}
			ad := d
			if ad < 0 {
				ad = -ad
			}
			w := kernel[ad]
			num += w * src[j]
			den += w
		}
		if den > 0 {
			dst[r] = num / den
		} else {
			dst[r] = src[r]
		}
	}
	return dst
}

func enhanceModulation(spectrum []float64, dims int, outF0, refLevel []float64, modBoost float64) {
	if dims <= 2 || modBoost <= 0 {
		return
	}
	rows := len(spectrum) / dims
	if rows < 3 || len(outF0) < rows {
		return
	}
	active := make([]bool, rows)
	for r := range rows {
		active[r] = outF0[r] > 0 && (len(refLevel) <= r || refLevel[r] > -35.0)
	}
	col := make([]float64, rows)
	for k := 2; k <= 17 && k < dims; k++ {
		var wk float64
		if k <= 5 {
			wk = 0.75
		} else if k <= 14 {
			wk = 1.0
		} else {
			wk = 0.5 * (1.0 + math.Cos(math.Pi*float64(k-14)/4.0))
		}
		var sumV, sumV2 float64
		var cntV int
		for r := range rows {
			v := spectrum[r*dims+k]
			col[r] = v
			if active[r] {
				sumV += v
				sumV2 += v * v
				cntV++
			}
		}
		if cntV < 2 {
			continue
		}
		meanV := sumV / float64(cntV)
		stdK := math.Sqrt(math.Max(0, sumV2/float64(cntV)-meanV*meanV))
		if stdK < 1e-6 {
			continue
		}
		cMed := maskedGaussianFilter(col, active, 1.0)
		cSlow := maskedGaussianFilter(col, active, 5.0)
		limit := 0.40 * stdK
		for r := range rows {
			if !active[r] {
				continue
			}
			delta := modBoost * wk * (cMed[r] - cSlow[r])
			if delta > limit {
				delta = limit
			} else if delta < -limit {
				delta = -limit
			}
			spectrum[r*dims+k] += delta
		}
	}
}

func applyVoicedBodySupport(decodedSpectrum []float64, bins, sampleRate int, outF0, levels []float64, bodySupportDB float64) {
	if bins <= 1 || sampleRate <= 0 || bodySupportDB <= 0 {
		return
	}
	nyq := float64(sampleRate) / 2.0
	bLo, bHi := bins, 0
	for i := range bins {
		hz := float64(i) * nyq / float64(bins-1)
		if hz >= 100.0 && hz < 1500.0 {
			if i < bLo {
				bLo = i
			}
			bHi = i + 1
		}
	}
	if bLo >= bHi {
		return
	}
	rows := len(decodedSpectrum) / bins
	if rows == 0 || len(levels) < rows {
		return
	}
	bodyDB := make([]float64, rows)
	var voicedBody []float64
	for row := range rows {
		rowBody := decodedSpectrum[row*bins+bLo : row*bins+bHi]
		bodyDB[row] = 10 * math.Log10(sum(rowBody)+1e-30)
		if (len(outF0) <= row || outF0[row] > 0) && levels[row] > -30.0 {
			voicedBody = append(voicedBody, bodyDB[row])
		}
	}
	if len(voicedBody) < 3 {
		return
	}
	bodyPeak := worldcodec.Percentile(voicedBody, 95)
	gain := make([]float64, rows)
	for row := range rows {
		if (len(outF0) <= row || outF0[row] > 0) && levels[row] > -24.0 {
			def := levels[row] - (bodyDB[row] - bodyPeak) - 2.0
			if def > 0 {
				gain[row] = math.Min(def*0.75, bodySupportDB)
			}
		}
	}
	gain = worldcodec.GaussianFilter(gain, 1.5)
	shelf := make([]float64, bins)
	for i := range bins {
		hz := float64(i) * nyq / float64(bins-1)
		r := hz / 1400.0
		r2 := r * r
		shelf[i] = 1.0 / (1.0 + r2*r2)
	}
	for row := range rows {
		g := gain[row]
		if g <= 1e-4 {
			continue
		}
		spRow := decodedSpectrum[row*bins : (row+1)*bins]
		for i := range bins {
			spRow[i] *= math.Pow(10, (g*shelf[i])/10.0)
		}
	}
}

func smooth(values []float64, dims int) {
	if dims <= 0 {
		return
	}
	rows := len(values) / dims
	if rows < 2 {
		return
	}
	original := append([]float64(nil), values...)
	for row := range rows {
		previous, next := max(row-1, 0), min(row+1, rows-1)
		for d := range dims {
			values[row*dims+d] = (original[previous*dims+d] + original[row*dims+d] + original[next*dims+d]) / 3
		}
	}
}
