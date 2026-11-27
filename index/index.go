package index

import (
	"fmt"
	"math"
	"sync"

	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
	"github.com/yh2237/gograd/tensor"
)

const (
	Version        = 1
	Magic          = "UTSP"
	FrameMS        = worldcodec.FrameMS
	SubFrames      = worldcodec.SubFrames
	WorldFrameMS   = worldcodec.WorldFrameMS
	SpectrumDims   = worldcodec.SpectrumDims
	D4CThreshold   = 0.85
	QuietMarginC0  = 9.0
	MinimumFileSec = 0.1

	maxFrames  = 1 << 30
	maxDims    = 1 << 20
	maxAPBands = 1 << 10
	maxFFTSize = 1 << 20
	maxRate    = 1 << 20
)

type World interface {
	Analyze(samples []float64, sampleRate int, frameMS, d4cThreshold float64) (f0, spectrum, aperiodicity []float64, fftSize int, err error)
	Synthesize(f0, spectrum, aperiodicity []float64, fftSize, sampleRate int, frameMS float64) ([]float64, error)
}

type FileEntry struct {
	Path     string           `json:"path"`
	Layer    speechpool.Layer `json:"layer"`
	Penalty  float32          `json:"penalty,omitempty"`
	Reversed bool             `json:"reversed,omitempty"`
}

type ChannelMeta struct {
	ID   string `json:"id"`
	Dims int    `json:"dims"`
}

type Header struct {
	Version    int           `json:"version"`
	SampleRate int           `json:"sample_rate"`
	FFTSize    int           `json:"fft_size"`
	APBands    int           `json:"ap_bands"`
	Frames     int           `json:"frames"`
	F0Median   float64       `json:"f0_median"`
	C35        float64       `json:"c35"`
	Source     string        `json:"source,omitempty"`
	CacheKey   string        `json:"cache_key,omitempty"`
	Files      []FileEntry   `json:"files"`
	Channels   []ChannelMeta `json:"channels"`
}

type Index struct {
	Header
	Channels      map[string][]float32
	Spectrum      []float32
	Aperiodicity  []float32
	F0            []float32
	FileID        []int32
	Position      []int32
	GlitchPenalty []float32

	Voiced      []bool
	MouthStatic []float32
	MouthDelta  []float32

	packedMu      sync.Mutex
	packedChannel map[string]*tensor.PackedB
	packedStatic  *tensor.PackedB
	packedDelta   *tensor.PackedB
}

func (h Header) validateShape() error {
	if h.Frames <= 0 || h.Frames > maxFrames {
		return fmt.Errorf("index: invalid frame count %d", h.Frames)
	}
	if !validWorldShape(h.SampleRate, h.FFTSize, h.APBands) {
		return fmt.Errorf("index: invalid WORLD shape (rate %d, FFT size %d, AP bands %d)", h.SampleRate, h.FFTSize, h.APBands)
	}
	if len(h.Files) > math.MaxInt32 {
		return fmt.Errorf("index: too many files (%d)", len(h.Files))
	}
	for _, width := range []int{SubFrames * SpectrumDims, SubFrames * h.APBands, SubFrames, worldcodec.MouthDims} {
		if _, ok := arrayLength(h.Frames, width, 4); !ok {
			return fmt.Errorf("index: WORLD or Mouth dimensions overflow")
		}
	}
	if len(h.Channels) == 0 {
		return fmt.Errorf("index: no channels")
	}
	seen := make(map[string]bool, len(h.Channels))
	for _, ch := range h.Channels {
		if ch.ID == "" || ch.Dims <= 0 || ch.Dims > maxDims {
			return fmt.Errorf("index: invalid channel %q with %d dimensions", ch.ID, ch.Dims)
		}
		if seen[ch.ID] {
			return fmt.Errorf("index: duplicate channel %q", ch.ID)
		}
		seen[ch.ID] = true
		if _, ok := arrayLength(h.Frames, ch.Dims, 4); !ok {
			return fmt.Errorf("index: channel %q dimensions overflow", ch.ID)
		}
	}
	return nil
}

func validWorldShape(rate, fftSize, apBands int) bool {
	return rate > 0 && rate <= maxRate && fftSize >= 2 && fftSize <= maxFFTSize &&
		fftSize%2 == 0 && apBands >= 0 && apBands <= maxAPBands
}

func checkedMul(a, b int64) (int64, bool) {
	if a < 0 || b < 0 || (a != 0 && b > math.MaxInt64/a) {
		return 0, false
	}
	return a * b, true
}

func arrayLength(frames, width, elementBytes int) (int, bool) {
	n, ok := checkedMul(int64(frames), int64(width))
	return int(n), ok && n <= int64(math.MaxInt/elementBytes)
}

func (idx *Index) Validate() error {
	if idx == nil {
		return fmt.Errorf("index: nil index")
	}
	if err := idx.Header.validateShape(); err != nil {
		return err
	}
	for _, check := range []struct {
		name      string
		got, want int
	}{
		{"spectrum", len(idx.Spectrum), idx.Frames * (SubFrames * SpectrumDims)},
		{"aperiodicity", len(idx.Aperiodicity), idx.Frames * (SubFrames * idx.APBands)},
		{"F0", len(idx.F0), idx.Frames * SubFrames},
		{"file ID", len(idx.FileID), idx.Frames},
		{"position", len(idx.Position), idx.Frames},
		{"glitch penalty", len(idx.GlitchPenalty), idx.Frames},
	} {
		if check.got != check.want {
			return fmt.Errorf("index: %s length %d, want %d", check.name, check.got, check.want)
		}
	}
	for _, ch := range idx.Header.Channels {
		values, ok := idx.Channels[ch.ID]
		if !ok || len(values) != idx.Frames*ch.Dims {
			return fmt.Errorf("index: channel %q has %d values, want %d", ch.ID, len(values), idx.Frames*ch.Dims)
		}
	}
	if len(idx.Files) > 0 {
		for f, id := range idx.FileID {
			if id < 0 || int(id) >= len(idx.Files) {
				return fmt.Errorf("index: frame %d has invalid file ID %d", f, id)
			}
		}
	}
	return nil
}

// EnsureDerived refreshes the WORLD-derived features after changes to the frame data.
func (idx *Index) EnsureDerived() {
	if idx == nil {
		return
	}
	idx.packedMu.Lock()
	defer idx.packedMu.Unlock()
	if idx.Validate() == nil {
		idx.computeDerived()
	}
}

func (idx *Index) computeDerived() {
	idx.Voiced = worldcodec.FrameVoicing(idx.F0)
	idx.MouthStatic, idx.MouthDelta = worldcodec.ComputeBankMouth(idx.Spectrum, idx.Frames, idx.Voiced, idx.FileID, idx.Position)
	idx.packedStatic, idx.packedDelta = nil, nil
}

func (idx *Index) ChannelDims(id string) int {
	for _, meta := range idx.Header.Channels {
		if meta.ID == id {
			return meta.Dims
		}
	}
	return 0
}

func (idx *Index) HasChannel(id string) bool { return idx.ChannelDims(id) != 0 }

func (idx *Index) FramePosition(j int) int32 {
	if len(idx.Position) == len(idx.FileID) {
		return idx.Position[j]
	}
	return int32(j)
}

func (idx *Index) PackedChannel(id string) (*tensor.PackedB, bool) {
	idx.packedMu.Lock()
	defer idx.packedMu.Unlock()
	if p := idx.packedChannel[id]; p != nil {
		return p, true
	}
	dims := idx.ChannelDims(id)
	values := idx.Channels[id]
	n, ok := arrayLength(idx.Frames, dims, 4)
	if !ok || dims <= 0 || dims > maxDims || idx.Frames <= 0 || idx.Frames > maxFrames || len(values) != n {
		return nil, false
	}
	if idx.packedChannel == nil {
		idx.packedChannel = make(map[string]*tensor.PackedB)
	}
	p := tensor.PackB(values, idx.Frames, dims, true)
	idx.packedChannel[id] = p
	return p, true
}

func (idx *Index) PackedMouthStatic() *tensor.PackedB { return idx.packedMouth(false) }
func (idx *Index) PackedMouthDelta() *tensor.PackedB  { return idx.packedMouth(true) }

func (idx *Index) packedMouth(delta bool) *tensor.PackedB {
	idx.packedMu.Lock()
	defer idx.packedMu.Unlock()
	n, ok := arrayLength(idx.Frames, worldcodec.MouthDims, 4)
	if !ok || idx.Frames <= 0 || idx.Frames > maxFrames {
		return nil
	}
	if len(idx.MouthStatic) != n || len(idx.MouthDelta) != n {
		if idx.Validate() != nil {
			return nil
		}
		idx.computeDerived()
	}
	packed, values := &idx.packedStatic, idx.MouthStatic
	if delta {
		packed, values = &idx.packedDelta, idx.MouthDelta
	}
	if *packed == nil {
		*packed = tensor.PackB(values, idx.Frames, worldcodec.MouthDims, true)
	}
	return *packed
}

func (idx *Index) ResetPacked() {
	idx.packedMu.Lock()
	defer idx.packedMu.Unlock()
	idx.resetPacked()
}

func (idx *Index) resetPacked() {
	idx.packedChannel = nil
	idx.packedStatic, idx.packedDelta = nil, nil
}

func (idx *Index) reversedFrameMask() (int, []bool) {
	hasReversed := false
	for _, f := range idx.Files {
		if f.Reversed {
			hasReversed = true
			break
		}
	}
	if !hasReversed || len(idx.Files) == 0 {
		return idx.Frames, nil
	}
	mask := make([]bool, idx.Frames)
	forwardCount := 0
	for f, id := range idx.FileID {
		if id >= 0 && int(id) < len(idx.Files) && idx.Files[id].Reversed {
			mask[f] = true
		} else {
			forwardCount++
		}
	}
	if forwardCount == 0 {
		return idx.Frames, nil
	}
	return forwardCount, mask
}

func (idx *Index) computeStatistics() {
	_, revMask := idx.reversedFrameMask()
	var positive, allPositive []float64
	for row, f := range idx.F0 {
		if f <= 0 {
			continue
		}
		allPositive = append(allPositive, float64(f))
		frame := row / SubFrames
		if revMask == nil || frame >= len(revMask) || !revMask[frame] {
			positive = append(positive, float64(f))
		}
	}
	if len(positive) == 0 {
		positive = allPositive
	}
	idx.F0Median = worldcodec.Percentile(positive, 50)
	idx.computeDerived()
	idx.C35 = computeC35Masked(idx.Spectrum, idx.Voiced, revMask)
	idx.GlitchPenalty = ComputeGlitchPenalties(idx.F0, idx.Aperiodicity, idx.APBands, idx.Voiced, idx.FileID, idx.F0Median)
}
