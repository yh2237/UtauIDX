package refmatch

import (
	"errors"
	"math"
	"runtime"
	"sync"

	"github.com/yh2237/gograd/tensor"
)

const (
	contextFrames   = 2
	contextWeight   = 0.5
	neighbors       = 4
	voicingPenalty  = 0.5
	smoothingFrames = 3
	postfilterBeta  = 0.4
	// 参照の音量がピークからquietLevelDBより下がった分だけ、出力も最大maxQuietCutDBまで下げる。
	quietLevelDB = 40.0
	// maxQuietCutDBは、参照が無音に近いところで出力を下げる上限。
	// 索引に静かなフレームが無いので、下げが足りないと無音に声が残る。
	maxQuietCutDB = 80.0
	// phonemeFallbackRatioは、音素で絞った最良の類似度が全体の最良に対して
	// この割合を下回ったら音素の制約を外す目安。
	phonemeFallbackRatio = 0.8
	matchBlockRows       = 32
)

// Requestは参照音声の特徴量とF0。
type Request struct {
	// FeatureModeが空またはwavlmならFeatures、cepstrumならCepstrumで照合する。
	FeatureMode FeatureMode
	Cepstrum    []float32
	// Featuresは参照音声の20msごとの特徴量（Frames×Dims）。
	Features []float32
	Frames   int
	// ReferenceF0は5msごとのF0（Hz、0は無声）。長さはFrames×4。
	ReferenceF0 []float64
	// ReferenceAPは5msごとの非周期性（0〜1、低域の平均）。nilなら単位のAPと従来の有声判定を使う。
	ReferenceAP []float32
	// CenterHzは出力のF0の中心。0なら音源のF0の中央値。
	CenterHz float64
	// PitchScaleは参照のF0の動きを何倍にするか。
	PitchScale float64
	// OutputF0を渡すと、CenterHzとPitchScaleの代わりにこのF0（5msごと、Hz、0は無声）で合成する。
	OutputF0 []float64
	// ReferenceLevelは参照の5msごとの音量（dB、ReferenceLevelsの値）。無音の区間で出力を下げるのに使う。
	ReferenceLevel []float64
	// SelectStatsがnilでなければ、最上位候補の選択統計を書き込む。呼び出しごとに上書きする。
	SelectStats *SelectStats
	// ContinuityBonusは直前の上位候補と同じ録音の次の20ms位置に加える類似度。0なら従来どおり。
	ContinuityBonus float64
	// TargetPhonemesは20msフレームごとの期待音素ID（Index.Phonemesの添字）。
	// nilまたは-1は指定なし。一致する単位がない場合も絞り込まない。
	TargetPhonemes []int32
	// PhonemeFallbackを真にすると、指定した音素に近い単位が無いフレームで
	// 音素の制約を外して選び直す（息や物音の単位が混ざるのを防ぐ）。
	PhonemeFallback bool
}

// SelectStatsは最上位候補を5ms単位で追った統計。合成自体は上位4候補の平均を使う。
// 隣の出力フレームで、同じ録音の次の5ms位置を選ばなければSwitchesを1増やす。
// 同じ位置の反復、逆行、飛び越し、録音の変更はすべて切り替わりに含む。
// Runsは連続区間の数、UniqueUnitsは選ばれた録音数、MeanRunFramesはFrames/Runs。
// 選択は20msごとで、その中の4つの5ms位置は必ず連続する。
type SelectStats struct {
	Frames, Runs, Switches, UniqueUnits int
	MeanRunFrames                       float64
}

// framePositionは手作りの索引でPositionが省略された場合だけ、索引の位置を使う。
func (index *Index) framePosition(j int) int32 {
	if len(index.Position) == len(index.File) {
		return index.Position[j]
	}
	return int32(j)
}

func (index *Index) selectStats(nearest [][]int) SelectStats {
	var stats SelectStats
	units := make(map[int32]struct{})
	previous := -1
	for _, chosen := range nearest {
		if len(chosen) == 0 {
			previous = -1
			continue
		}
		j := chosen[0]
		units[index.File[j]] = struct{}{}
		stats.Frames += subFrames
		if previous < 0 {
			stats.Runs++
		} else if index.File[j] != index.File[previous] || index.framePosition(j) != index.framePosition(previous)+1 {
			stats.Switches++
			stats.Runs++
		}
		previous = j
	}
	stats.UniqueUnits = len(units)
	if stats.Runs > 0 {
		stats.MeanRunFrames = float64(stats.Frames) / float64(stats.Runs)
	}
	return stats
}

// Synthesizeは、参照の各フレームに前後2フレームまで含めて近い音源のフレームを4つ選び、その平均をWORLDで合成する。
func (index *Index) Synthesize(world World, request Request) ([]float64, error) {
	parameters, err := index.Parameters(request)
	if err != nil {
		return nil, err
	}
	return world.Synthesize(parameters.F0, parameters.Spectrum, parameters.Aperiodicity, index.FFTSize, index.SampleRate, worldFrameMS)
}

// SynthesizeWithStatsは合成が成功したらreportへ統計を渡す。
// コールバックの引数には既存の型だけを使い、旧版にも対応する呼び出し側がinterfaceで検出できるようにする。
func (index *Index) SynthesizeWithStats(world World, request Request, report func(frames, runs, switches, uniqueUnits int, meanRunFrames float64)) ([]float64, error) {
	var stats SelectStats
	output := request.SelectStats
	request.SelectStats = &stats
	audio, err := index.Synthesize(world, request)
	if output != nil {
		*output = stats
	}
	if err == nil && report != nil {
		report(stats.Frames, stats.Runs, stats.Switches, stats.UniqueUnits, stats.MeanRunFrames)
	}
	return audio, err
}

// ParametersはSynthesizeでWORLDへ渡すパラメータ。5msごと。
type Parameters struct {
	F0 []float64
	// SpectrumとAperiodicityはフレームごとにFFTSize/2+1個並ぶ。
	Spectrum     []float64
	Aperiodicity []float64
}

// ParametersはSynthesizeと同じようにフレームを選び、WORLDで合成する前のパラメータを返す。
func (index *Index) Parameters(request Request) (Parameters, error) {
	if request.SelectStats != nil {
		*request.SelectStats = SelectStats{}
	}
	if err := request.FeatureMode.Validate(); err != nil {
		return Parameters{}, err
	}
	if request.FeatureMode == FeatureCepstrum {
		matched, err := index.cepstrumIndex()
		if err != nil {
			return Parameters{}, err
		}
		request.Features = request.Cepstrum
		request.FeatureMode = FeatureWavLM
		return matched.Parameters(request)
	}
	if request.Frames <= 0 || len(request.Features) != request.Frames*index.Dims || len(request.ReferenceF0) < request.Frames*subFrames {
		return Parameters{}, errors.New("reference features and F0 do not match")
	}
	if math.IsNaN(request.ContinuityBonus) || math.IsInf(request.ContinuityBonus, 0) || request.ContinuityBonus < 0 {
		return Parameters{}, errors.New("continuity bonus must be finite and non-negative")
	}
	if request.TargetPhonemes != nil && len(request.TargetPhonemes) != request.Frames {
		return Parameters{}, errors.New("target phonemes and frames do not match")
	}
	for _, id := range request.TargetPhonemes {
		if id < -1 {
			return Parameters{}, errors.New("target phoneme ID must be -1 or non-negative")
		}
	}
	rows := request.Frames * subFrames
	if request.ReferenceAP != nil {
		if len(request.ReferenceAP) != rows {
			return Parameters{}, errors.New("reference AP and frames do not match")
		}
		for _, ap := range request.ReferenceAP {
			if math.IsNaN(float64(ap)) || ap < 0 || ap > 1 {
				return Parameters{}, errors.New("reference AP must be finite and between zero and one")
			}
		}
	}
	f0 := request.ReferenceF0[:rows]
	if request.ReferenceAP != nil {
		f0 = append([]float64(nil), f0...)
		for row, ap := range request.ReferenceAP {
			if ap >= 0.5 {
				f0[row] = 0
			}
		}
	}
	nearest := index.nearestWithPhonemes(request.Features, request.Frames, frameVoicing(f0), request.ContinuityBonus, request.TargetPhonemes, request.PhonemeFallback)
	if request.SelectStats != nil {
		*request.SelectStats = index.selectStats(nearest)
	}
	spectrum := index.average(index.Spectrum, spectrumDims, nearest, rows)
	aperiodicity := index.average(index.Aperiodicity, index.APBands, nearest, rows)
	smooth(spectrum, spectrumDims)
	smooth(aperiodicity, index.APBands)

	c := newCodec(index.SampleRate, index.FFTSize, spectrumDims)
	bins := index.FFTSize/2 + 1
	outF0 := request.OutputF0
	if len(outF0) < rows {
		outF0 = index.OutputF0(f0, request.CenterHz, request.PitchScale)
	}
	outF0 = outF0[:rows]
	if request.ReferenceAP != nil {
		outF0 = append([]float64(nil), outF0...)
		for row, ap := range request.ReferenceAP {
			// 音量が大きい摩擦音もあるため、音量による補間やOutputF0よりAPを優先する。
			// 有声は参照F0 > 0かつAP < 0.5の行だけとする。
			if f0[row] <= 0 || ap >= 0.5 {
				outF0[row] = 0
			}
		}
	}
	decodedSpectrum := make([]float64, rows*bins)
	decodedAP := make([]float64, rows*bins)
	sharpened := make([]float64, spectrumDims)
	before := make([]float64, bins)
	cut := quietCut(request.ReferenceLevel, rows)
	for row := range rows {
		coded := spectrum[row*spectrumDims : (row+1)*spectrumDims]
		c.decodeSpectrum(coded, before)
		copy(sharpened, coded)
		for k := 2; k < spectrumDims; k++ {
			sharpened[k] *= 1 + postfilterBeta
		}
		after := decodedSpectrum[row*bins : (row+1)*bins]
		c.decodeSpectrum(sharpened, after)
		ratio := sum(before) / sum(after) * math.Pow(10, -cut[row]/10)
		for i := range after {
			after[i] *= ratio
		}
		ap := decodedAP[row*bins : (row+1)*bins]
		if outF0[row] == 0 {
			for i := range ap {
				ap[i] = 1
			}
		} else {
			c.decodeAperiodicity(aperiodicity[row*index.APBands:(row+1)*index.APBands], ap)
		}
		if request.ReferenceAP != nil {
			reference := float64(request.ReferenceAP[row])
			// 線形APで(1-w)*単位AP+w*参照APを混ぜる。
			// AP<=0.2ではw=0.05。有声域(AP<0.5)は単位の比率を過半に保ち、APが高いほど参照へ寄せる（上限1）。
			weight := 0.05 + 0.95*math.Max(0, math.Min(1, (reference-0.2)/0.8))
			for i := range ap {
				ap[i] = (1-weight)*ap[i] + weight*reference
			}
		}
	}
	return Parameters{F0: outF0, Spectrum: decodedSpectrum, Aperiodicity: decodedAP}, nil
}

// ReferenceLevelsは参照音声の5msごとの音量を、95パーセンタイルを0dBとして返す。
func ReferenceLevels(samples []float64, sampleRate, rows int) []float64 {
	hop := sampleRate * int(worldFrameMS) / 1000
	levels := make([]float64, rows)
	for row := range rows {
		var energy float64
		for i := (row - 1) * hop; i < (row+2)*hop; i++ {
			if i >= 0 && i < len(samples) {
				energy += samples[i] * samples[i]
			}
		}
		levels[row] = 10 * math.Log10(energy/float64(3*hop)+1e-12)
	}
	peak := percentile(levels, 95)
	for i := range levels {
		levels[i] -= peak
	}
	return levels
}

// quietCutは行ごとに出力を下げるdBを返す。
func quietCut(levels []float64, rows int) []float64 {
	cut := make([]float64, rows)
	if len(levels) < rows {
		return cut
	}
	for row := range rows {
		if levels[row] < -quietLevelDB {
			cut[row] = math.Min(-quietLevelDB-levels[row], maxQuietCutDB)
		}
	}
	return gaussianFilter(cut, 2)
}

func frameVoicing(f0 []float64) []bool {
	voiced := make([]bool, len(f0)/subFrames)
	for frame := range voiced {
		count := 0
		for _, v := range f0[frame*subFrames : (frame+1)*subFrames] {
			if v > 0 {
				count++
			}
		}
		voiced[frame] = count*2 >= subFrames
	}
	return voiced
}

// nearestは参照のフレームごとに、文脈を含めた類似度が高い音源のフレームをneighbors個返す。
func (index *Index) nearest(features []float32, frames int, queryVoiced []bool) [][]int {
	return index.nearestWithContinuity(features, frames, queryVoiced, 0)
}

func (index *Index) nearestWithContinuity(features []float32, frames int, queryVoiced []bool, bonus float64) [][]int {
	return index.nearestWithPhonemes(features, frames, queryVoiced, bonus, nil, false)
}

func (index *Index) nearestWithPhonemes(features []float32, frames int, queryVoiced []bool, bonus float64, targets []int32, fallback bool) [][]int {
	dims, n := index.Dims, index.Frames
	available := make(map[int32]bool)
	if targets != nil {
		for _, file := range index.File {
			if file >= 0 && int(file) < len(index.FilePhonemes) {
				id := index.FilePhonemes[file]
				if id >= 0 && int(id) < len(index.Phonemes) {
					available[id] = true
				}
			}
		}
	}
	query := make([]float32, len(features))
	for t := range frames {
		copy(query[t*dims:], normalized(features[t*dims:(t+1)*dims]))
	}
	bankF0 := make([]float64, len(index.F0))
	for i, v := range index.F0 {
		bankF0[i] = float64(v)
	}
	bankVoiced := frameVoicing(bankF0)
	packed := index.packedFeatures()
	result := make([][]int, frames)
	blocks := make(chan int)
	var wg sync.WaitGroup
	workers := max(1, min(runtime.GOMAXPROCS(0), (frames+matchBlockRows-1)/matchBlockRows))
	// ボーナスありでは選択を時系列順に進め、計算ブロックの境界でも前の選択を保持する。
	if bonus != 0 {
		workers = 1
	}
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var previous []int
			for start := range blocks {
				end := min(start+matchBlockRows, frames)
				first, last := max(0, start-contextFrames), min(frames, end+contextFrames)
				sim := make([]float32, (last-first)*n)
				tensor.SGEMMPacked(sim, query[first*dims:last*dims], packed, last-first, 1)
				for t := first; t < last; t++ {
					row := sim[(t-first)*n : (t-first+1)*n]
					for j := range row {
						if queryVoiced[t] != bankVoiced[j] {
							row[j] -= voicingPenalty
						}
					}
				}
				for t := start; t < end; t++ {
					target := int32(-1)
					if targets != nil && available[targets[t]] {
						target = targets[t]
					}
					result[t] = index.topContextWithPhoneme(sim, first, last, t, previous, bonus, target, fallback)
					if bonus != 0 {
						previous = result[t]
					}
				}
			}
		}()
	}
	for start := 0; start < frames; start += matchBlockRows {
		blocks <- start
	}
	close(blocks)
	wg.Wait()
	return result
}

func (index *Index) topContext(sim []float32, first, last, t int) []int {
	return index.topContextWithContinuity(sim, first, last, t, nil, 0)
}

func (index *Index) topContextWithContinuity(sim []float32, first, last, t int, previous []int, bonus float64) []int {
	return index.topContextWithPhoneme(sim, first, last, t, previous, bonus, -1, false)
}

func (index *Index) topContextWithPhoneme(sim []float32, first, last, t int, previous []int, bonus float64, target int32, fallback bool) []int {
	n := index.Frames
	bestScore := make([]float32, 0, neighbors)
	best := make([]int, 0, neighbors)
	bestAll := float32(-1e30)
	for j := range n {
		var total, count float32
		for delta := -contextFrames; delta <= contextFrames; delta++ {
			tt, jj := t+delta, j+delta
			if tt < first || tt >= last || jj < 0 || jj >= n || index.File[jj] != index.File[j] {
				continue
			}
			weight := float32(1)
			if delta != 0 {
				weight = contextWeight
			}
			total += weight * sim[(tt-first)*n+jj]
			count += weight
		}
		score := total / max(count, 1e-6)
		if bonus != 0 {
			for _, p := range previous {
				if index.File[j] == index.File[p] && index.framePosition(j) == index.framePosition(p)+1 {
					// 類似度の加算はコストの減算と同じ。複数候補に続いても一度だけ加える。
					score += float32(bonus)
					break
				}
			}
		}
		if score > bestAll {
			bestAll = score
		}
		if target >= 0 {
			file := index.File[j]
			if file < 0 || int(file) >= len(index.FilePhonemes) || index.FilePhonemes[file] != target {
				continue
			}
		}
		if len(best) == neighbors && score <= bestScore[neighbors-1] {
			continue
		}
		position := len(best)
		if position == neighbors {
			position--
		} else {
			best = append(best, 0)
			bestScore = append(bestScore, 0)
		}
		for position > 0 && bestScore[position-1] < score {
			best[position], bestScore[position] = best[position-1], bestScore[position-1]
			position--
		}
		best[position], bestScore[position] = j, score
	}
	if fallback && target >= 0 && (len(best) == 0 || bestScore[0] < phonemeFallbackRatio*bestAll) {
		// 指定した音素に近い単位が無いときは、音素の制約を外して選び直す。
		// 遠い候補を無理に当てると、息や物音の単位が混ざる。
		return index.topContextWithPhoneme(sim, first, last, t, previous, bonus, -1, false)
	}
	return best
}

func (index *Index) average(values []float32, dims int, nearest [][]int, rows int) []float64 {
	out := make([]float64, rows*dims)
	for t, chosen := range nearest {
		for sub := range subFrames {
			target := out[(t*subFrames+sub)*dims : (t*subFrames+sub+1)*dims]
			for _, j := range chosen {
				source := values[(j*subFrames+sub)*dims : (j*subFrames+sub+1)*dims]
				for d, v := range source {
					target[d] += float64(v) / float64(len(chosen))
				}
			}
		}
	}
	return out
}

// smoothはscipy.ndimage.uniform_filter1d（size 3、reflect）と同じく時間方向に3行の平均をとる。
func smooth(values []float64, dims int) {
	rows := len(values) / dims
	if rows < 2 {
		return
	}
	original := append([]float64(nil), values...)
	for row := range rows {
		previous, next := max(row-1, 0), min(row+1, rows-1)
		for d := range dims {
			values[row*dims+d] = (original[previous*dims+d] + original[row*dims+d] + original[next*dims+d]) / smoothingFrames
		}
	}
}

// OutputF0は参照のF0を、中央値がcenterHz（0なら音源のF0の中央値）になるよう移し、動きをscale倍にする。
func (index *Index) OutputF0(reference []float64, centerHz, scale float64) []float64 {
	if centerHz <= 0 {
		centerHz = index.F0Median
	}
	var voiced []float64
	for _, v := range reference {
		if v > 0 {
			voiced = append(voiced, v)
		}
	}
	out := make([]float64, len(reference))
	if len(voiced) == 0 {
		return out
	}
	median := percentile(voiced, 50)
	for i, v := range reference {
		if v > 0 {
			out[i] = centerHz * math.Exp(scale*math.Log(math.Max(v, 1)/median))
		}
	}
	return out
}

func sum(values []float64) float64 {
	var total float64
	for _, v := range values {
		total += v
	}
	return total
}

// ReferenceF0は参照TTSのF0（f0FrameMSごと、60Hz以下は無声）を、有声区間の対数F0を約20msでなめらかにして5msごとに直す。
func ReferenceF0(f0 []float32, f0FrameMS float64, frames int) []float64 {
	var times, logF0 []float64
	voicedness := make([]float64, len(f0))
	axis := make([]float64, len(f0))
	for i, v := range f0 {
		axis[i] = float64(i) * f0FrameMS
		if v > 60 {
			voicedness[i] = 1
			times = append(times, axis[i])
			logF0 = append(logF0, math.Log(float64(v)))
		}
	}
	out := make([]float64, frames*subFrames)
	if len(times) == 0 {
		return out
	}
	filled := make([]float64, len(f0))
	for i, t := range axis {
		filled[i] = interpClamped(times, logF0, t)
	}
	smoothed := gaussianFilter(filled, 20/f0FrameMS)
	for i := range out {
		t := float64(i) * worldFrameMS
		if interpClamped(axis, voicedness, t) > 0.5 {
			out[i] = math.Exp(interpClamped(axis, smoothed, t))
		}
	}
	return out
}

// FillVoicedByLevelは、参照の音量が十分あるのに無声になった区間へ、前後のF0を埋める。
// 参照TTSのF0が語尾で途切れ、鼻音などが無声として合成されるのを防ぐ。
func FillVoicedByLevel(f0, levels []float64, thresholdDB float64) {
	n := len(f0)
	if n == 0 || len(levels) < n {
		return
	}
	previous := -1
	for i := range n {
		if f0[i] <= 0 {
			continue
		}
		if previous >= 0 {
			for k := previous + 1; k < i; k++ {
				if levels[k] <= thresholdDB {
					continue
				}
				f0[k] = f0[previous] + (f0[i]-f0[previous])*float64(k-previous)/float64(i-previous)
			}
		}
		previous = i
	}
	// 末尾側も、音量が残っていれば最後のF0を続ける。
	if previous >= 0 {
		for k := previous + 1; k < n; k++ {
			if levels[k] <= thresholdDB {
				break
			}
			f0[k] = f0[previous]
		}
	}
}

// interpClampedはnumpy.interpと同じく範囲外を端の値にする。
func interpClamped(x, y []float64, at float64) float64 {
	if at <= x[0] {
		return y[0]
	}
	if at >= x[len(x)-1] {
		return y[len(y)-1]
	}
	k := 1
	for x[k] < at {
		k++
	}
	return y[k-1] + (at-x[k-1])/(x[k]-x[k-1])*(y[k]-y[k-1])
}

// gaussianFilterはscipy.ndimage.gaussian_filter1d（truncate 4、reflect）と同じ。
func gaussianFilter(values []float64, sigma float64) []float64 {
	radius := int(4*sigma + 0.5)
	weights := make([]float64, 2*radius+1)
	var total float64
	for i := range weights {
		x := float64(i - radius)
		weights[i] = math.Exp(-0.5 * x * x / (sigma * sigma))
		total += weights[i]
	}
	n := len(values)
	reflect := func(i int) int {
		for i < 0 || i >= n {
			if i < 0 {
				i = -i - 1
			}
			if i >= n {
				i = 2*n - i - 1
			}
		}
		return i
	}
	out := make([]float64, n)
	for i := range values {
		var acc float64
		for k, w := range weights {
			acc += w * values[reflect(i+k-radius)]
		}
		out[i] = acc / total
	}
	return out
}

// packedFeaturesは行列積のために並べ替えた特徴量を、索引ごとに1度だけ作る。
func (index *Index) packedFeatures() *tensor.PackedB {
	index.packOnce.Do(func() {
		index.packed = tensor.PackB(index.Features, index.Frames, index.Dims, true)
	})
	return index.packed
}
