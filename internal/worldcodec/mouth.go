package worldcodec

import "math"

// MouthDims is the number of vocal-tract mel-cepstral dimensions (C2..C13) used by
// both static and delta Mouth channels.
const MouthDims = 12

// MouthWeights weights C2..C13 to emphasize F1/F2 vocal-tract articulation.
var MouthWeights = [MouthDims]float64{
	1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
}

// ComputeQueryMouth computes L2-normalized static (12d) and temporal-delta (12d)
// Mouth cepstral features from 20ms averaged C2..C13 coefficients.
func ComputeQueryMouth(c2To13 []float64, frames int, voiced []bool) (static, delta []float32) {
	if frames <= 0 || len(c2To13) < frames*MouthDims {
		return nil, nil
	}
	cnw := normalizeMouthWeighted(c2To13, frames, voiced)
	static = make([]float32, frames*MouthDims)
	delta = make([]float32, frames*MouthDims)
	var raw [MouthDims]float32
	for t := range frames {
		row := cnw[t*MouthDims : (t+1)*MouthDims]
		for d := range MouthDims {
			raw[d] = float32(row[d])
		}
		NormalizeInto(static[t*MouthDims:(t+1)*MouthDims], raw[:])

		prev := max(t-1, 0)
		next := min(t+1, frames-1)
		pRow := cnw[prev*MouthDims : (prev+1)*MouthDims]
		nRow := cnw[next*MouthDims : (next+1)*MouthDims]
		for d := range MouthDims {
			raw[d] = float32(0.5 * (nRow[d] - pRow[d]))
		}
		NormalizeInto(delta[t*MouthDims:(t+1)*MouthDims], raw[:])
	}
	return static, delta
}

// ComputeBankMouth reconstructs L2-normalized static (12d) and file/position-aware
// delta (12d) Mouth features on-the-fly from an index's 60d coded Spectrum array.
func ComputeBankMouth(spectrum []float32, frames int, voiced []bool, fileID, pos []int32) (static, delta []float32) {
	if frames <= 0 || len(spectrum) < frames*SubFrames*SpectrumDims {
		return nil, nil
	}
	c2To13 := make([]float64, frames*MouthDims)
	invSub := 1.0 / float64(SubFrames)
	for j := range frames {
		dst := c2To13[j*MouthDims : (j+1)*MouthDims]
		for sub := range SubFrames {
			src := spectrum[(j*SubFrames+sub)*SpectrumDims+2 : (j*SubFrames+sub)*SpectrumDims+2+MouthDims]
			for d := range MouthDims {
				dst[d] += float64(src[d]) * invSub
			}
		}
	}
	cnw := normalizeMouthWeighted(c2To13, frames, voiced)
	static = make([]float32, frames*MouthDims)
	delta = make([]float32, frames*MouthDims)
	hasMeta := len(fileID) == frames && len(pos) == frames
	var raw [MouthDims]float32
	for j := range frames {
		row := cnw[j*MouthDims : (j+1)*MouthDims]
		for d := range MouthDims {
			raw[d] = float32(row[d])
		}
		NormalizeInto(static[j*MouthDims:(j+1)*MouthDims], raw[:])

		pPrev, pNext := j, j
		if hasMeta {
			if j > 0 && fileID[j-1] == fileID[j] && pos[j-1] == pos[j]-1 {
				pPrev = j - 1
			}
			if j+1 < frames && fileID[j+1] == fileID[j] && pos[j+1] == pos[j]+1 {
				pNext = j + 1
			}
		}
		prevRow := cnw[pPrev*MouthDims : (pPrev+1)*MouthDims]
		nextRow := cnw[pNext*MouthDims : (pNext+1)*MouthDims]
		for d := range MouthDims {
			raw[d] = float32(0.5 * (nextRow[d] - prevRow[d]))
		}
		NormalizeInto(delta[j*MouthDims:(j+1)*MouthDims], raw[:])
	}
	return static, delta
}

func normalizeMouthWeighted(c2To13 []float64, frames int, voiced []bool) []float64 {
	var useAll bool
	if len(voiced) < frames {
		useAll = true
	} else {
		cnt := 0
		for _, v := range voiced[:frames] {
			if v {
				cnt++
			}
		}
		if cnt < 5 {
			useAll = true
		}
	}
	var mean, std [MouthDims]float64
	var active int
	for j := range frames {
		if !useAll && !voiced[j] {
			continue
		}
		active++
		row := c2To13[j*MouthDims : (j+1)*MouthDims]
		for d := range MouthDims {
			mean[d] += row[d]
		}
	}
	inv := 1.0 / float64(max(active, 1))
	for d := range MouthDims {
		mean[d] *= inv
	}
	for j := range frames {
		if !useAll && !voiced[j] {
			continue
		}
		row := c2To13[j*MouthDims : (j+1)*MouthDims]
		for d := range MouthDims {
			diff := row[d] - mean[d]
			std[d] += diff * diff
		}
	}
	for d := range MouthDims {
		std[d] = math.Sqrt(std[d]*inv) + 1e-5
	}
	cnw := make([]float64, frames*MouthDims)
	for j := range frames {
		src := c2To13[j*MouthDims : (j+1)*MouthDims]
		dst := cnw[j*MouthDims : (j+1)*MouthDims]
		for d := range MouthDims {
			dst[d] = ((src[d] - mean[d]) / std[d]) * MouthWeights[d]
		}
	}
	return cnw
}
