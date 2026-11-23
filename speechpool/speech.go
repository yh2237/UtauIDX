package speechpool

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/yh2237/audiodsp/wav"
)

// SpeechOptions controls segmentation and level normalization for speech recordings.
type SpeechOptions struct {
	MaxSegmentSec float64 // Defaults to 10 seconds.
	MinSegmentSec float64 // Defaults to 0.4 seconds.
	SilenceDB     float64 // Relative to peak frame RMS. Defaults to -35 dB.
	MinSilenceMS  float64 // Defaults to 180 ms.
	TargetRMS     float64 // Defaults to 0.10. A negative value skips RMS gain.
}

func (opts SpeechOptions) withDefaults() SpeechOptions {
	if opts.MaxSegmentSec <= 0 {
		opts.MaxSegmentSec = 10
	}
	if opts.MinSegmentSec <= 0 {
		opts.MinSegmentSec = 0.4
	}
	if opts.SilenceDB >= 0 {
		opts.SilenceDB = -35
	}
	if opts.MinSilenceMS <= 0 {
		opts.MinSilenceMS = 180
	}
	if opts.TargetRMS == 0 {
		opts.TargetRMS = 0.10
	}
	return opts
}

// PrepareSpeechDir finds WAVs recursively and writes mono segments to outDir.
// An empty outDir returns one entry per original recording. If processing fails,
// the returned entries describe the recordings already processed.
func PrepareSpeechDir(srcDir, outDir string, opts SpeechOptions) ([]Entry, error) {
	paths, err := speechWAVPaths(srcDir, outDir)
	if err != nil {
		return nil, err
	}
	if outDir != "" {
		for _, value := range []float64{opts.MaxSegmentSec, opts.MinSegmentSec, opts.SilenceDB, opts.MinSilenceMS, opts.TargetRMS} {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("speech options must be finite")
			}
		}
		opts = opts.withDefaults()
		if opts.MinSegmentSec > opts.MaxSegmentSec {
			return nil, fmt.Errorf("minimum speech segment duration exceeds maximum")
		}
		if len(paths) > 0 {
			if err := os.MkdirAll(outDir, 0o755); err != nil {
				return nil, err
			}
		}
	}

	var entries []Entry
	for fileIndex, path := range paths {
		pcm, err := wav.ReadFile(path)
		if err != nil {
			return entries, fmt.Errorf("read speech WAV %s: %w", path, err)
		}
		if len(pcm.Data) == 0 {
			continue
		}
		entry := Entry{Path: path, Layer: LayerSpeech, Penalty: CleanFilePenalty(path, false)}
		if outDir == "" {
			entries = append(entries, entry)
			continue
		}

		samples := monoSamples(pcm)
		ends, err := speechSegmentEnds(samples, pcm.SampleRate, opts)
		if err != nil {
			return entries, fmt.Errorf("split speech WAV %s: %w", path, err)
		}
		start := 0
		for segmentIndex, end := range ends {
			entry.Path = filepath.Join(outDir, fmt.Sprintf("seg_%04d_%03d.wav", fileIndex+1, segmentIndex))
			segment := normalizedPCM16(samples[start:end], pcm.SampleRate, opts.TargetRMS)
			if err := wav.WriteFile(entry.Path, segment); err != nil {
				return entries, fmt.Errorf("write speech WAV %s: %w", entry.Path, err)
			}
			entries = append(entries, entry)
			start = end
		}
	}
	return entries, nil
}

func speechWAVPaths(srcDir, outDir string) ([]string, error) {
	info, err := os.Stat(srcDir)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("speech source is not a directory: %s", srcDir)
	}

	var skipDir string
	if outDir != "" {
		src, err := filepath.Abs(srcDir)
		if err != nil {
			return nil, err
		}
		out, err := filepath.Abs(outDir)
		if err != nil {
			return nil, err
		}
		if outInfo, err := os.Stat(out); err == nil && os.SameFile(info, outInfo) {
			return nil, fmt.Errorf("speech source and output directories must differ")
		}
		skipDir, _ = filepath.Rel(src, out)
		if skipDir == "." {
			return nil, fmt.Errorf("speech source and output directories must differ")
		}
	}

	var paths []string
	err = filepath.WalkDir(srcDir, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if skipDir != "" {
				rel, _ := filepath.Rel(srcDir, path)
				if match, err := filepath.Rel(skipDir, rel); err == nil && match == "." {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if strings.EqualFold(filepath.Ext(entry.Name()), ".wav") {
			paths = append(paths, path)
		}
		return nil
	})
	sort.Strings(paths)
	return paths, err
}

func speechSegmentEnds(samples []float64, rate int, opts SpeechOptions) ([]int, error) {
	length := len(samples)
	if float64(length) <= opts.MaxSegmentSec*float64(rate) {
		return []int{length}, nil
	}
	maxSamples := int(opts.MaxSegmentSec * float64(rate))
	minSamples := int(math.Ceil(opts.MinSegmentSec * float64(rate)))
	if maxSamples < minSamples || !fitsSpeechSegments(length, minSamples, maxSamples) {
		return nil, fmt.Errorf("recording cannot be split within the configured segment durations")
	}

	frameSize := max(1, rate/100)
	frameRMS, silences := speechSilences(samples, rate, frameSize, opts)
	var ends []int
	start := 0
	for length-start > maxSamples {
		lo, hi := start+minSamples, min(start+maxSamples, length-minSamples)
		i := sort.SearchInts(silences, hi+1) - 1
		for ; i >= 0 && silences[i] >= lo; i-- {
			if fitsSpeechSegments(length-silences[i], minSamples, maxSamples) {
				break
			}
		}
		var end int
		if i >= 0 && silences[i] >= lo {
			end = silences[i]
		} else {
			// Leave enough samples for the remaining minimum-length segments.
			remainingSegments := speechSegmentCount(length-hi, maxSamples)
			hi = min(hi, length-remainingSegments*minSamples)
			lo = max(lo, length-remainingSegments*maxSamples)
			end = speechFallbackCut(samples, frameRMS, frameSize, max(lo, hi-rate/10), hi)
		}
		ends = append(ends, end)
		start = end
	}
	return append(ends, length), nil
}

func speechSegmentCount(length, maxSamples int) int {
	return (length-1)/maxSamples + 1
}

func fitsSpeechSegments(length, minSamples, maxSamples int) bool {
	return speechSegmentCount(length, maxSamples) <= length/minSamples
}

func speechSilences(samples []float64, rate, frameSize int, opts SpeechOptions) ([]float64, []int) {
	rms := make([]float64, 0, speechSegmentCount(len(samples), frameSize))
	var peak float64
	for start := 0; start < len(samples); start += frameSize {
		frame := samples[start:min(start+frameSize, len(samples))]
		var sumSquares float64
		for _, v := range frame {
			sumSquares += v * v
		}
		level := math.Sqrt(sumSquares / float64(len(frame)))
		rms = append(rms, level)
		peak = max(peak, level)
	}
	threshold := peak * math.Pow(10, opts.SilenceDB/20)
	minQuietSamples := opts.MinSilenceMS * float64(rate) / 1000
	var midpoints []int
	for first := 0; first < len(rms); {
		if rms[first] > threshold {
			first++
			continue
		}
		last := first + 1
		for last < len(rms) && rms[last] <= threshold {
			last++
		}
		start, end := first*frameSize, min(last*frameSize, len(samples))
		if float64(end-start) >= minQuietSamples {
			midpoints = append(midpoints, start+(end-start)/2)
		}
		first = last
	}
	return rms, midpoints
}

func speechFallbackCut(samples, rms []float64, frameSize, lo, hi int) int {
	quietest := hi / frameSize
	for i := quietest - 1; i >= lo/frameSize; i-- {
		if rms[i] < rms[quietest] {
			quietest = i
		}
	}
	lo = max(lo, quietest*frameSize)
	hi = min(hi, (quietest+1)*frameSize-1)

	best, bestMagnitude := hi, math.Inf(1)
	for i := hi; i >= lo; i-- {
		left, right := samples[i-1], samples[i]
		if left == 0 || right == 0 || math.Signbit(left) != math.Signbit(right) {
			return i
		}
		if magnitude := math.Abs(left) + math.Abs(right); magnitude < bestMagnitude {
			best, bestMagnitude = i, magnitude
		}
	}
	return best
}
