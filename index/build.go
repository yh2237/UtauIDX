package index

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"sync"

	"github.com/yh2237/UtauIDX/encoder"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
	"github.com/yh2237/UtauIDX/speechpool"
	"github.com/yh2237/audiodsp/wav"
	"github.com/yh2237/gograd/dsp"
)

type BuildOptions struct {
	TargetSampleRate int
	Source, CacheKey string
	Cache            *PartCache
	Progress         func(done, total int)
}

func RecordingSpan(regions []speechpool.Region, durMS float64) (start, end float64) {
	start, end = math.Inf(1), math.Inf(-1)
	for _, region := range regions {
		last := durMS - region.Blank
		if region.Blank < 0 {
			last = region.Offset - region.Blank
		}
		if last > region.Offset {
			start, end = min(start, region.Offset), max(end, last)
		}
	}
	if start >= end {
		return 0, durMS
	}
	return max(start, 0), min(end, durMS)
}

type buildEncoder struct {
	encoder.Encoder
	meta ChannelMeta
}

func describeEncoder(enc encoder.Encoder) (buildEncoder, error) {
	if enc == nil {
		return buildEncoder{}, errors.New("index: nil encoder")
	}
	meta := ChannelMeta{ID: enc.ID(), Dims: enc.Dims()}
	if meta.ID == "" || meta.Dims <= 0 || meta.Dims > maxDims {
		return buildEncoder{}, fmt.Errorf("index: invalid encoder %q with %d dimensions", meta.ID, meta.Dims)
	}
	return buildEncoder{Encoder: enc, meta: meta}, nil
}

type buildPart struct {
	WorldPart
	channels [][]float32
}

func Build(entries []speechpool.Entry, world World, encoders []encoder.Encoder, opts BuildOptions) (*Index, error) {
	if world == nil {
		return nil, errors.New("index: nil WORLD analyzer")
	}
	if len(encoders) == 0 {
		return nil, errors.New("index: at least one encoder is required")
	}
	idx := &Index{
		Header:   Header{Version: Version, Source: opts.Source, CacheKey: opts.CacheKey},
		Channels: make(map[string][]float32, len(encoders)),
	}
	models := make([]buildEncoder, 0, len(encoders))
	seen := make(map[string]bool, len(encoders))
	for _, enc := range encoders {
		model, err := describeEncoder(enc)
		if err != nil {
			return nil, err
		}
		if seen[model.meta.ID] {
			return nil, fmt.Errorf("index: duplicate encoder %q", model.meta.ID)
		}
		seen[model.meta.ID] = true
		models = append(models, model)
		idx.Header.Channels = append(idx.Header.Channels, model.meta)
	}
	rate := opts.TargetSampleRate
	if rate <= 0 {
		rate = chooseIndexRate(entries)
	}
	if rate <= 0 || rate > maxRate {
		return nil, fmt.Errorf("index: no usable target sample rate (%d)", rate)
	}
	idx.SampleRate = rate

	results, err := processFiles(len(entries), opts.Progress, func(i int) (buildPart, error) {
		entry := entries[i]
		part, err := buildRecording(entry, world, models, opts.Cache, rate)
		if err != nil {
			err = fmt.Errorf("%s: %w", entry.Path, err)
		}
		return part, err
	})
	if err != nil {
		return nil, err
	}
	for i, part := range results {
		if !part.Usable {
			continue
		}
		if idx.FFTSize == 0 {
			idx.FFTSize, idx.APBands = part.FFTSize, part.APBands
		}
		if idx.FFTSize != part.FFTSize || idx.APBands != part.APBands {
			return nil, fmt.Errorf("index: incompatible WORLD shape in %s", entries[i].Path)
		}
		if part.UsableFrames > maxFrames-idx.Frames {
			return nil, errors.New("index: too many frames")
		}
		entry := entries[i]
		fid := int32(len(idx.Files))
		idx.Files = append(idx.Files, FileEntry{Path: entry.Path, Layer: entry.Layer, Penalty: entry.Penalty, Reversed: entry.Reversed})
		idx.Frames += part.UsableFrames
		idx.Spectrum = append(idx.Spectrum, part.Spectrum...)
		idx.Aperiodicity = append(idx.Aperiodicity, part.Aperiodicity...)
		idx.F0 = append(idx.F0, part.F0...)
		idx.Position = append(idx.Position, part.Position...)
		for range part.UsableFrames {
			idx.FileID = append(idx.FileID, fid)
		}
		for c, meta := range idx.Header.Channels {
			idx.Channels[meta.ID] = append(idx.Channels[meta.ID], part.channels[c]...)
		}
		results[i] = buildPart{}
	}
	if idx.Frames == 0 {
		return nil, errors.New("index: no usable frames in the recordings")
	}
	if err := idx.Header.validateShape(); err != nil {
		return nil, err
	}
	for _, meta := range idx.Header.Channels {
		idx.Channels[meta.ID] = normalizedChannel(idx.Channels[meta.ID], idx.Frames, meta.Dims)
	}
	idx.Spectrum = worldcodec.RoundToHalf(idx.Spectrum)
	idx.Aperiodicity = worldcodec.RoundToHalf(idx.Aperiodicity)
	idx.F0 = worldcodec.RoundToHalf(idx.F0)
	idx.computeStatistics()
	return idx, nil
}

func buildRecording(entry speechpool.Entry, world World, models []buildEncoder, cache *PartCache, rate int) (buildPart, error) {
	audio, err := readRecording(entry.Path, entry.Reversed)
	if err != nil {
		return buildPart{}, nil
	}
	channels := make([]ChannelPart, len(models))
	var jobs inferenceBatch
	limit := maxFrames
	for i, model := range models {
		part, err := loadOrExtractChannel(model, audio, cache)
		if err != nil {
			if jobs.capture(err) {
				continue
			}
			return buildPart{}, fmt.Errorf("extract %q: %w", model.meta.ID, err)
		}
		channels[i] = part
		// A convolutional encoder can produce one fewer row at the trailing edge.
		if part.RawFrames == 0 {
			limit = 0
		} else {
			limit = min(limit, part.RawFrames+1)
		}
	}
	if jobs.pending {
		return buildPart{}, &jobs.need
	}
	if limit == 0 {
		return buildPart{}, nil
	}
	durMS := float64(len(audio.samples)) * 1000 / float64(audio.rate)
	start, end := RecordingSpan(entry.Regions, durMS)
	if entry.Reversed {
		start, end = max(0, durMS-end), min(durMS, durMS-start)
	}
	part, err := loadOrAnalyzeWorld(world, audio, cache, rate, start, end)
	if err != nil || !part.Usable {
		return buildPart{}, err
	}
	// WORLD cache positions are monotonic, so incompatible positions form a suffix.
	keep := 0
	for keep < len(part.Position) && int(part.Position[keep]) < limit {
		keep++
	}
	part.Usable, part.UsableFrames = keep > 0, keep
	part.Position = part.Position[:keep]
	part.Spectrum = part.Spectrum[:keep*SubFrames*SpectrumDims]
	part.Aperiodicity = part.Aperiodicity[:keep*SubFrames*part.APBands]
	part.F0 = part.F0[:keep*SubFrames]
	out := buildPart{WorldPart: part, channels: make([][]float32, len(channels))}
	for c, channel := range channels {
		out.channels[c] = make([]float32, keep*channel.Dims)
		for f, p := range part.Position {
			copy(out.channels[c][f*channel.Dims:], channel.row(int(p)))
		}
	}
	return out, nil
}

type recordingAudio struct {
	data     []byte
	samples  []float64
	rate     int
	reversed bool
	wav16k   []float32
}

func readRecording(path string, reversed bool) (*recordingAudio, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pcm, err := wav.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	samples := make([]float64, len(pcm.Data)/pcm.Channels)
	if pcm.SampleRate > maxRate || float64(len(samples)) < MinimumFileSec*float64(pcm.SampleRate) {
		return nil, errors.New("index: WAV is too short or has an unsupported sample rate")
	}
	for i := range samples {
		var sum float64
		for _, value := range pcm.Data[i*pcm.Channels : (i+1)*pcm.Channels] {
			sum += float64(value)
		}
		samples[i] = sum / (32768 * float64(pcm.Channels))
	}
	if reversed {
		slices.Reverse(samples)
	}
	return &recordingAudio{data: data, samples: samples, rate: pcm.SampleRate, reversed: reversed}, nil
}

func (audio *recordingAudio) resample(rate int) []float32 {
	input := make([]float32, len(audio.samples))
	for i, value := range audio.samples {
		input[i] = float32(value)
	}
	if rate == audio.rate {
		return input
	}
	return dsp.Resample(input, audio.rate, rate)
}

func loadOrExtractChannel(model buildEncoder, audio *recordingAudio, cache *PartCache) (ChannelPart, error) {
	var key string
	if cache != nil {
		key = channelKey(audio.data, model.meta.ID, audio.reversed)
		if part, ok := cache.loadChannel(key, model.meta.Dims); ok {
			return part, nil
		}
	}
	if audio.wav16k == nil {
		audio.wav16k = audio.resample(encoder.SampleRate)
	}
	values, frames, err := model.Extract(audio.wav16k)
	if err != nil {
		return ChannelPart{}, err
	}
	n, ok := arrayLength(frames, model.meta.Dims, 4)
	if !ok || frames > maxFrames || len(values) != n {
		return ChannelPart{}, fmt.Errorf("invalid encoder output (%d frames, %d dimensions, %d values)", frames, model.meta.Dims, len(values))
	}
	part := ChannelPart{RawFrames: frames, Dims: model.meta.Dims, Values: values}
	if cache != nil {
		cache.saveChannel(key, part)
	}
	return part, nil
}

func (part ChannelPart) row(position int) []float32 {
	start := min(position, part.RawFrames-1) * part.Dims
	return part.Values[start : start+part.Dims]
}

func loadOrAnalyzeWorld(world World, audio *recordingAudio, cache *PartCache, rate int, startMS, endMS float64) (WorldPart, error) {
	var key string
	if cache != nil {
		key = worldKey(audio.data, rate, startMS, endMS, audio.reversed)
		if part, ok := cache.loadWorld(key, rate); ok {
			return part, nil
		}
	}
	part, err := analyzeWorld(world, audio, rate, startMS, endMS)
	if err == nil && cache != nil {
		cache.saveWorld(key, part)
	}
	return part, err
}

func analyzeWorld(world World, audio *recordingAudio, rate int, startMS, endMS float64) (WorldPart, error) {
	samples := audio.samples
	if rate != audio.rate {
		resampled := audio.resample(rate)
		samples = make([]float64, len(resampled))
		for i, value := range resampled {
			samples[i] = float64(value)
		}
	}
	f0, spectrum, ap, fftSize, err := world.Analyze(samples, rate, WorldFrameMS, D4CThreshold)
	if err != nil {
		return WorldPart{}, err
	}
	if !validWorldShape(rate, fftSize, 0) {
		return WorldPart{}, fmt.Errorf("index: WORLD returned invalid FFT size %d", fftSize)
	}
	frames := len(f0) / SubFrames
	rows := frames * SubFrames
	bins := fftSize/2 + 1
	worldLen, worldOK := arrayLength(rows, bins, 8)
	codeLen, codeOK := arrayLength(rows, SpectrumDims, 8)
	if frames > maxFrames || !worldOK || !codeOK || len(spectrum) < worldLen || len(ap) < worldLen {
		return WorldPart{}, errors.New("index: WORLD analysis rows do not match")
	}
	codec := worldcodec.New(rate, fftSize, SpectrumDims)
	part := WorldPart{Rate: rate, FFTSize: fftSize, APBands: codec.APBands()}
	codes := make([]float64, codeLen)
	levels := make([]float64, rows)
	for row := range rows {
		code := codes[row*SpectrumDims : (row+1)*SpectrumDims]
		codec.CodeSpectrum(spectrum[row*bins:(row+1)*bins], code)
		levels[row] = code[0]
	}
	quietCut := worldcodec.Percentile(levels, 95) - QuietMarginC0
	apCode := make([]float64, part.APBands)
	for frame := range frames {
		center := (float64(frame) + 0.5) * FrameMS
		if center < startMS || center >= endMS {
			continue
		}
		first := frame * SubFrames
		loud := false
		for _, level := range levels[first : first+SubFrames] {
			loud = loud || level >= quietCut
		}
		if !loud {
			continue
		}
		part.Position = append(part.Position, int32(frame))
		for row := first; row < first+SubFrames; row++ {
			for _, value := range codes[row*SpectrumDims : (row+1)*SpectrumDims] {
				part.Spectrum = append(part.Spectrum, float32(value))
			}
			codec.CodeAperiodicity(ap[row*bins:(row+1)*bins], apCode)
			for _, value := range apCode {
				part.Aperiodicity = append(part.Aperiodicity, float32(value))
			}
			part.F0 = append(part.F0, float32(f0[row]))
		}
	}
	part.UsableFrames = len(part.Position)
	part.Usable = part.UsableFrames > 0
	return part, nil
}

func AddChannel(idx *Index, enc encoder.Encoder, cache *PartCache, progress func(done, total int)) error {
	if idx == nil {
		return errors.New("index: nil index")
	}
	idx.packedMu.Lock()
	err := idx.Validate()
	idx.packedMu.Unlock()
	if err != nil {
		return err
	}
	model, err := describeEncoder(enc)
	if err != nil {
		return err
	}
	if len(idx.Files) == 0 {
		return errors.New("index: cannot add a channel without recording paths")
	}
	n, ok := arrayLength(idx.Frames, model.meta.Dims, 4)
	if !ok {
		return errors.New("index: channel dimensions overflow")
	}
	rows := make([][]int, len(idx.Files))
	for f, id := range idx.FileID {
		if idx.Position[f] < 0 {
			return fmt.Errorf("index: frame %d has invalid position %d", f, idx.Position[f])
		}
		rows[id] = append(rows[id], f)
	}
	values := make([]float32, n)
	_, err = processFiles(len(idx.Files), progress, func(i int) (struct{}, error) {
		var result struct{}
		if len(rows[i]) == 0 {
			return result, nil
		}
		path := idx.Files[i].Path
		audio, err := readRecording(path, idx.Files[i].Reversed)
		if err != nil {
			return result, fmt.Errorf("%s: %w", path, err)
		}
		part, err := loadOrExtractChannel(model, audio, cache)
		if err != nil {
			return result, fmt.Errorf("%s: extract %q: %w", path, model.meta.ID, err)
		}
		if part.RawFrames == 0 {
			return result, fmt.Errorf("%s: encoder %q returned no frames", path, model.meta.ID)
		}
		for _, f := range rows[i] {
			copy(values[f*part.Dims:], part.row(int(idx.Position[f])))
		}
		return result, nil
	})
	if err != nil {
		return err
	}
	values = normalizedChannel(values, idx.Frames, model.meta.Dims)
	idx.packedMu.Lock()
	defer idx.packedMu.Unlock()
	idx.Channels[model.meta.ID] = values
	for i, ch := range idx.Header.Channels {
		if ch.ID == model.meta.ID {
			idx.Header.Channels[i] = model.meta
			idx.resetPacked()
			return nil
		}
	}
	idx.Header.Channels = append(idx.Header.Channels, model.meta)
	idx.resetPacked()
	return nil
}

type inferenceBatch struct {
	pending bool
	need    encoder.NeedInference
	seen    map[[2]string]bool
}

func (batch *inferenceBatch) capture(err error) bool {
	var need *encoder.NeedInference
	if !errors.As(err, &need) {
		return false
	}
	batch.pending = true
	if batch.seen == nil {
		batch.seen = make(map[[2]string]bool)
	}
	for _, job := range need.Jobs {
		key := [2]string{job.EncoderID, job.Key}
		if !batch.seen[key] {
			batch.seen[key] = true
			batch.need.Jobs = append(batch.need.Jobs, job)
		}
	}
	return true
}

func processFiles[T any](total int, progress func(done, total int), work func(int) (T, error)) ([]T, error) {
	results, errs := make([]T, total), make([]error, total)
	var mu sync.Mutex
	done := 0
	var wg sync.WaitGroup
	workers := min(runtime.GOMAXPROCS(0), total)
	for worker := range workers {
		wg.Go(func() {
			for i := worker; i < total; i += workers {
				results[i], errs[i] = work(i)
				if progress != nil {
					mu.Lock()
					done++
					progress(done, total)
					mu.Unlock()
				}
			}
		})
	}
	wg.Wait()
	var jobs inferenceBatch
	for _, err := range errs {
		if err != nil && !jobs.capture(err) {
			return nil, err
		}
	}
	if jobs.pending {
		return nil, &jobs.need
	}
	return results, nil
}
