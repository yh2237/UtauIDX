package worldcodec

import (
	"math"
	"sort"
)

// Percentile returns the linearly interpolated q-th percentile (0..100), matching numpy.percentile.
func Percentile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	position := q / 100 * float64(len(sorted)-1)
	lower := int(math.Floor(position))
	upper := min(lower+1, len(sorted)-1)
	return sorted[lower] + (sorted[upper]-sorted[lower])*(position-float64(lower))
}

// GaussianFilter applies a 1D Gaussian filter with truncate=4 and reflect boundary padding,
// matching scipy.ndimage.gaussian_filter1d.
func GaussianFilter(values []float64, sigma float64) []float64 {
	n := len(values)
	if n == 0 || sigma <= 0 {
		return append([]float64(nil), values...)
	}
	radius := int(4*sigma + 0.5)
	weights := make([]float64, 2*radius+1)
	var total float64
	for i := range weights {
		x := float64(i - radius)
		weights[i] = math.Exp(-0.5 * x * x / (sigma * sigma))
		total += weights[i]
	}
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

// InterpClamped linearly interpolates y(x) at point at, clamping out-of-range queries to endpoints.
func InterpClamped(x, y []float64, at float64) float64 {
	if len(x) == 0 || len(y) == 0 {
		return 0
	}
	if at <= x[0] || len(x) == 1 {
		return y[0]
	}
	if at >= x[len(x)-1] {
		return y[len(y)-1]
	}
	k := sort.Search(len(x), func(i int) bool { return x[i] >= at })
	if k <= 0 {
		return y[0]
	}
	if k >= len(x) {
		return y[len(y)-1]
	}
	if x[k] == at {
		return y[k]
	}
	return y[k-1] + (at-x[k-1])/(x[k]-x[k-1])*(y[k]-y[k-1])
}

// NormalizeInto L2-normalizes src into dst. dst and src may alias.
func NormalizeInto(dst, src []float32) {
	var sum float64
	for _, v := range src {
		sum += float64(v) * float64(v)
	}
	scale := float32(1 / math.Max(math.Sqrt(sum), 1e-12))
	for i, v := range src {
		dst[i] = v * scale
	}
}

// NormalizeRows returns a copy of values with each dims-wide row L2-normalized.
func NormalizeRows(values []float32, frames, dims int) []float32 {
	if frames <= 0 || dims <= 0 || len(values) < frames*dims {
		return nil
	}
	out := make([]float32, frames*dims)
	for f := range frames {
		NormalizeInto(out[f*dims:(f+1)*dims], values[f*dims:(f+1)*dims])
	}
	return out
}

// FrameVoicing converts 5ms WORLD F0 into 20ms per-frame voicing decisions
// (at least half of the SubFrames sub-frames must be voiced).
func FrameVoicing[T ~float32 | ~float64](f0 []T) []bool {
	voiced := make([]bool, len(f0)/SubFrames)
	for frame := range voiced {
		count := 0
		for _, v := range f0[frame*SubFrames : (frame+1)*SubFrames] {
			if v > 0 {
				count++
			}
		}
		voiced[frame] = count*2 >= SubFrames
	}
	return voiced
}
