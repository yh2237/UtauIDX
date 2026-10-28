// Package renderは、参照の音声から索引（音源の録音）のフレームで合成し直す。
package render

import (
	"fmt"
	"math"
	"slices"

	"github.com/yh2237/UtauIDX/refmatch"
	"github.com/yh2237/gograd/dsp"
)

// Optionsは参照音声からの合成に使う設定。
type Options struct {
	// FeatureModeはwavlm（空なら既定）またはcepstrum。
	FeatureMode refmatch.FeatureMode
	// ExpectModelは索引の特徴量モデル。空でなければ一致を確かめる。
	ExpectModel string
	// UseReferenceAPは参照APによる有声判定と混合を有効にする。falseなら従来どおり。
	UseReferenceAP bool
	// ContinuityBonusは同じ録音の次の位置を優先する強さ。0なら従来どおり。
	ContinuityBonus float64
	// PhonemeFallbackは指定した音素に近い単位が無いとき制約を外す。
	PhonemeFallback bool
	// PitchSemitonesは参照から移す半音数。nilなら音源の音域に自動で合わせる。
	PitchSemitones *float64
	// TargetPhonemesは20msごとの索引音素ID。nilなら音素で絞らない。
	TargetPhonemes []int32
	Logf           func(string)
}

// Statsは5msごとの参照F0の有声フレーム数。
type Stats struct {
	Frames, Before, After int
}

// Resultは合成した音声と、そのとき使ったF0。
type Result struct {
	Audio      []float64
	SampleRate int
	// F0は出力のF0（5msごと、Hz、0は無声）。
	F0      []float64
	Voicing Stats
}

// FromAudioは参照の音声を分析し、索引のフレームで合成し直す。
// featuresはWavLMの特徴量を計算する関数（cepstrumのときは使わない）。
func FromAudio(index *refmatch.Index, world refmatch.World, features refmatch.FeatureFunc, audio []float64, sampleRate int, options Options) (Result, error) {
	if err := options.FeatureMode.Validate(); err != nil {
		return Result{}, err
	}
	if options.ExpectModel != "" && options.FeatureMode != refmatch.FeatureCepstrum && index.Model != options.ExpectModel {
		return Result{}, fmt.Errorf("voicebank index uses %q, want %q", index.Model, options.ExpectModel)
	}
	var query []float32
	var frames int
	var referenceF0, levels []float64
	var referenceAP []float32

	if options.FeatureMode == refmatch.FeatureCepstrum {
		var err error
		query, frames, referenceF0, referenceAP, err = analyzeCepstrum(index, world, audio, sampleRate, options.UseReferenceAP)
		if err != nil {
			return Result{}, err
		}
		levels = refmatch.ReferenceLevels(audio, sampleRate, frames*refmatch.SubFrames())
		if !options.UseReferenceAP {
			referenceAP = nil
		}
	} else {
		if features == nil {
			return Result{}, fmt.Errorf("feature function is required for %q", options.FeatureMode)
		}
		raw, _, ap, fftSize, err := world.Analyze(audio, sampleRate, refmatch.FrameMS, 0.85)
		if err != nil {
			return Result{}, fmt.Errorf("analyze reference: %w", err)
		}
		values, count, dims, err := features(audio, sampleRate)
		if err != nil {
			return Result{}, err
		}
		if dims != index.Dims {
			return Result{}, fmt.Errorf("feature dimensions are %d, want %d", dims, index.Dims)
		}
		query, frames = values, count
		raw32 := make([]float32, len(raw))
		for i, v := range raw {
			raw32[i] = float32(v)
		}
		referenceF0 = refmatch.ReferenceF0(raw32, refmatch.FrameMS, frames)
		levels = refmatch.ReferenceLevels(audio, sampleRate, frames*refmatch.SubFrames())
		if options.UseReferenceAP {
			referenceAP, err = referenceAPRows(ap, sampleRate, fftSize, refmatch.FrameMS, frames*refmatch.SubFrames())
			if err != nil {
				return Result{}, err
			}
		}
	}
	return synthesize(index, world, query, frames, referenceF0, levels, referenceAP, options)
}

// analyzeCepstrumは参照の包絡から検索用のケプストラムとF0を作る。
func analyzeCepstrum(index *refmatch.Index, world refmatch.World, samples []float64, rate int, useAP bool) ([]float32, int, []float64, []float32, error) {
	if rate <= 0 || index.SampleRate <= 0 || len(samples) == 0 {
		return nil, 0, nil, nil, fmt.Errorf("invalid reference audio or index sample rate")
	}
	if rate != index.SampleRate {
		wave := make([]float32, len(samples))
		for i, v := range samples {
			wave[i] = float32(v)
		}
		resampled := dsp.Resample(wave, rate, index.SampleRate)
		samples = make([]float64, len(resampled))
		for i, v := range resampled {
			samples[i] = float64(v)
		}
		rate = index.SampleRate
	}
	f0, spectrum, ap, fftSize, err := world.Analyze(samples, rate, refmatch.WorldFrameMS(), 0.85)
	if err != nil {
		return nil, 0, nil, nil, fmt.Errorf("analyze reference: %w", err)
	}
	cepstrum, frames, err := refmatch.ReferenceCepstrum(spectrum, rate, fftSize, refmatch.WorldFrameMS())
	if err != nil {
		return nil, 0, nil, nil, err
	}
	f0Values := make([]float32, len(f0))
	for i, v := range f0 {
		f0Values[i] = float32(v)
	}
	referenceF0 := refmatch.ReferenceF0(f0Values, refmatch.WorldFrameMS(), frames)
	var referenceAP []float32
	if useAP {
		referenceAP, err = referenceAPRows(ap, rate, fftSize, refmatch.WorldFrameMS(), frames*refmatch.SubFrames())
		if err != nil {
			return nil, 0, nil, nil, err
		}
	}
	return cepstrum, frames, referenceF0, referenceAP, nil
}

func synthesize(index *refmatch.Index, world refmatch.World, query []float32, frames int, referenceF0, levels []float64, referenceAP []float32, options Options) (Result, error) {
	ratio, referenceMedian, indexMedian, err := pitchRatio(referenceF0, index.F0, options.PitchSemitones)
	if err != nil {
		return Result{}, err
	}
	if options.Logf != nil {
		options.Logf(fmt.Sprintf("ピッチ補正: 参照の中央値 %.6f Hz、索引の中央値 %.6f Hz、倍率 %.9f", referenceMedian, indexMedian, ratio))
	}
	matched := index
	if options.FeatureMode != refmatch.FeatureCepstrum {
		matched, query, err = standardizeFeatures(index, query, frames)
		if err != nil {
			return Result{}, err
		}
	}
	stats := Stats{Frames: len(referenceF0), Before: voicedFrames(referenceF0)}
	if referenceAP == nil {
		refmatch.FillVoicedByLevel(referenceF0, levels, -35)
	} else {
		if len(referenceAP) != len(referenceF0) {
			return Result{}, fmt.Errorf("reference AP and F0 do not match")
		}
		// 大きな音の子音を音量だけで有声に戻さない。APが高ければF0があっても無声にする。
		for i, ap := range referenceAP {
			if ap >= 0.5 {
				referenceF0[i] = 0
			}
		}
	}
	stats.After = voicedFrames(referenceF0)
	automaticF0 := slices.Clone(referenceF0)
	for i, hz := range automaticF0 {
		if hz > 0 {
			automaticF0[i] = hz * ratio
		}
	}
	request := refmatch.Request{
		FeatureMode: options.FeatureMode, TargetPhonemes: options.TargetPhonemes,
		PhonemeFallback: options.PhonemeFallback,
		Features:        query, Frames: frames, ReferenceF0: referenceF0, OutputF0: automaticF0,
		ReferenceLevel: levels, ReferenceAP: referenceAP, ContinuityBonus: options.ContinuityBonus,
	}
	if options.FeatureMode == refmatch.FeatureCepstrum {
		request.Cepstrum, request.Features = query, nil
	}
	rendered, err := matched.Synthesize(world, request)
	if err != nil {
		return Result{}, err
	}
	return Result{Audio: rendered, SampleRate: matched.SampleRate, F0: automaticF0, Voicing: stats}, nil
}

// referenceAPRowsはWORLDの線形APを合成の5ms行に合わせる。
// 高域は母音でも非周期的なので、1500Hz以下の平均を有声判定と混合に使う。
func referenceAPRows(ap []float64, rate, fftSize int, frameMS float64, rows int) ([]float32, error) {
	bins := fftSize/2 + 1
	if rate <= 0 || fftSize <= 0 || frameMS <= 0 || len(ap) == 0 || len(ap)%bins != 0 || rows <= 0 {
		return nil, fmt.Errorf("invalid reference AP dimensions")
	}
	sourceRows := len(ap) / bins
	lowBins := min(bins, int(1500*float64(fftSize)/float64(rate))+1)
	means := make([]float64, sourceRows)
	for row := range sourceRows {
		for _, value := range ap[row*bins : row*bins+lowBins] {
			if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
				return nil, fmt.Errorf("invalid reference AP value")
			}
			means[row] += value / float64(lowBins)
		}
	}
	out := make([]float32, rows)
	for row := range rows {
		position := math.Min(float64(sourceRows-1), float64(row)*refmatch.WorldFrameMS()/frameMS)
		left := int(position)
		fraction := position - float64(left)
		out[row] = float32(means[left]*(1-fraction) + means[min(left+1, sourceRows-1)]*fraction)
	}
	return out, nil
}

func pitchRatio(reference []float64, bank []float32, semitones *float64) (ratio, referenceMedian, indexMedian float64, err error) {
	referenceMedian = voicedMedian(reference)
	indexMedian = voicedMedian(bank)
	ratio = 1
	if semitones != nil {
		if math.IsNaN(*semitones) || math.IsInf(*semitones, 0) {
			return 0, referenceMedian, indexMedian, fmt.Errorf("pitch semitones must be finite")
		}
		ratio = math.Exp2(math.Max(-12, math.Min(12, *semitones)) / 12)
	} else if referenceMedian > 0 && indexMedian > 0 {
		ratio = math.Max(0.5, math.Min(2, indexMedian/referenceMedian))
	}
	return ratio, referenceMedian, indexMedian, nil
}

func voicedMedian[T ~float32 | ~float64](f0 []T) float64 {
	voiced := make([]float64, 0, len(f0))
	for _, hz := range f0 {
		v := float64(hz)
		if v > 0 && !math.IsInf(v, 0) {
			voiced = append(voiced, v)
		}
	}
	if len(voiced) == 0 {
		return 0
	}
	slices.Sort(voiced)
	middle := len(voiced) / 2
	if len(voiced)%2 == 0 {
		return voiced[middle-1]/2 + voiced[middle]/2
	}
	return voiced[middle]
}

func voicedFrames(f0 []float64) int {
	count := 0
	for _, hz := range f0 {
		if hz > 0 {
			count++
		}
	}
	return count
}

// standardizeFeaturesは索引の統計で両側を標準化する。元の索引とその検索キャッシュは変更しない。
func standardizeFeatures(index *refmatch.Index, features []float32, frames int) (*refmatch.Index, []float32, error) {
	dims := index.Dims
	if dims <= 0 || index.Frames <= 0 || len(index.Features) != index.Frames*dims || frames <= 0 || len(features) != frames*dims {
		return nil, nil, fmt.Errorf("audio features and index dimensions do not match")
	}
	// 保存済みの索引はL2正規化済みなので、参照も同じ空間へ移してから統計を適用する。
	// 索引も再正規化し、保存時の量子化によるノルムのずれを除く。
	bank := unitFeatureRows(index.Features, dims)
	query := unitFeatureRows(features, dims)
	mean, variance := make([]float64, dims), make([]float64, dims)
	for row := range index.Frames {
		for d := range dims {
			v := float64(bank[row*dims+d])
			delta := v - mean[d]
			mean[d] += delta / float64(row+1)
			variance[d] += delta * (v - mean[d])
		}
	}
	inverseStd := make([]float64, dims)
	for d := range dims {
		inverseStd[d] = 1 / math.Max(math.Sqrt(math.Max(variance[d], 0)/float64(index.Frames)), 1e-6)
	}
	transform := func(values []float32) {
		for i, v := range values {
			d := i % dims
			values[i] = float32((float64(v) - mean[d]) * inverseStd[d])
		}
		normalizeFeatureRows(values, dims)
	}
	// nearestは参照をL2正規化して内積で比較するため、変換後の索引も単位長にする。
	transform(bank)
	transform(query)
	return index.WithFeatures(bank), query, nil
}

func unitFeatureRows(values []float32, dims int) []float32 {
	out := append([]float32(nil), values...)
	normalizeFeatureRows(out, dims)
	return out
}

func normalizeFeatureRows(values []float32, dims int) {
	for start := 0; start < len(values); start += dims {
		row := values[start : start+dims]
		var energy float64
		for _, v := range row {
			energy += float64(v) * float64(v)
		}
		scale := 1 / math.Max(math.Sqrt(energy), 1e-12)
		for d, v := range row {
			row[d] = float32(float64(v) * scale)
		}
	}
}
