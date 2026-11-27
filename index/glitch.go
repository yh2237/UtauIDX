package index

import (
	"math"

	"github.com/yh2237/UtauIDX/internal/worldcodec"
)

func ComputeC35(spectrum []float32, voiced []bool) float64 {
	return computeC35Masked(spectrum, voiced, nil)
}

func computeC35Masked(spectrum []float32, voiced, skip []bool) float64 {
	frames := len(spectrum) / (SubFrames * SpectrumDims)
	voiced = voiced[:min(frames, len(voiced))]
	voicedOnly := false
	for f, isVoiced := range voiced {
		if isVoiced && (skip == nil || f >= len(skip) || !skip[f]) {
			voicedOnly = true
			break
		}
	}
	var sum float64
	used := 0
	for f := range frames {
		if skip != nil && f < len(skip) && skip[f] {
			continue
		}
		if voicedOnly && (f >= len(voiced) || !voiced[f]) {
			continue
		}
		for sub := range SubFrames {
			start := (f*SubFrames+sub)*SpectrumDims + 3
			for _, coeff := range spectrum[start : start+3] {
				sum += float64(coeff) * float64(coeff)
			}
		}
		used++
	}
	if used == 0 {
		return 0
	}
	return math.Sqrt(sum / (float64(used) * SubFrames * 3))
}

func ComputeGlitchPenalties(f0, ap []float32, apBands int, voiced []bool, fileID []int32, globalF0Med float64) []float32 {
	frames := len(f0) / SubFrames
	penalties := make([]float32, frames)
	if len(voiced) != frames || len(fileID) != frames {
		return penalties
	}
	if globalF0Med <= 0 {
		globalF0Med = 200
	}
	apLength, ok := arrayLength(frames, SubFrames, 4)
	if ok {
		apLength, ok = arrayLength(apLength, apBands, 4)
	}
	hasAP := ok && apBands > 0 && len(ap) >= apLength
	rowSum := func(row int) float64 {
		var sum float64
		for _, v := range ap[row*apBands : (row+1)*apBands] {
			sum += float64(v)
		}
		return sum
	}

	fileF0 := make(map[int32][]float64)
	var cleanAP0 []float64
	for row, f := range f0[:frames*SubFrames] {
		if f <= 0 {
			continue
		}
		id := fileID[row/SubFrames]
		fileF0[id] = append(fileF0[id], float64(f))
		if hasAP && rowSum(row)/float64(apBands) <= -1 {
			cleanAP0 = append(cleanAP0, float64(ap[row*apBands]))
		}
	}
	fileMedian := make(map[int32]float64, len(fileF0))
	for id, positive := range fileF0 {
		fileMedian[id] = globalF0Med
		if len(positive) >= 4 {
			fileMedian[id] = worldcodec.Percentile(positive, 50)
		}
	}
	cleanAP0Med := -12.0
	if len(cleanAP0) > 0 {
		cleanAP0Med = worldcodec.Percentile(cleanAP0, 50)
	}
	ap0Cut := max(-5.0, cleanAP0Med+10)

	for frame, isVoiced := range voiced {
		if !isVoiced {
			continue
		}
		var pen float32
		var apSum, ap0Sum float64
		count := 0
		for row := frame * SubFrames; row < (frame+1)*SubFrames; row++ {
			if f0[row] <= 0 {
				continue
			}
			count++
			if math.Abs(1200*math.Log2(float64(f0[row])/fileMedian[fileID[frame]])) > 300 {
				pen = max(pen, 0.10)
			}
			if hasAP {
				apSum += rowSum(row)
				ap0Sum += float64(ap[row*apBands])
			}
		}
		if hasAP && count > 0 {
			if apSum/(float64(count)*float64(apBands)) > -0.8 {
				pen = max(pen, 0.12)
			}
			if cleanAP0Med < -8 && ap0Sum/float64(count) > ap0Cut {
				pen = max(pen, 0.06)
			}
		}
		penalties[frame] = pen
	}
	return penalties
}
