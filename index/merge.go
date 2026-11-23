package index

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

func Merge(indexes ...*Index) (*Index, error) {
	if len(indexes) == 0 {
		return nil, errors.New("index: no indexes to merge")
	}
	first := indexes[0]
	if err := first.Validate(); err != nil {
		return nil, fmt.Errorf("index 0: %w", err)
	}
	shared := slices.Clone(first.Header.Channels)
	totalFrames := first.Frames
	for i, idx := range indexes[1:] {
		if err := idx.Validate(); err != nil {
			return nil, fmt.Errorf("index %d: %w", i+1, err)
		}
		if idx.SampleRate != first.SampleRate || idx.FFTSize != first.FFTSize || idx.APBands != first.APBands {
			return nil, fmt.Errorf("index %d: incompatible WORLD shape", i+1)
		}
		if idx.Frames > maxFrames-totalFrames {
			return nil, errors.New("index: too many merged frames")
		}
		totalFrames += idx.Frames
		shared = slices.DeleteFunc(shared, func(ch ChannelMeta) bool {
			return idx.ChannelDims(ch.ID) != ch.Dims
		})
	}
	if len(shared) == 0 {
		return nil, errors.New("index: no shared channels with compatible dimensions")
	}
	h := Header{
		Version: Version, SampleRate: first.SampleRate, FFTSize: first.FFTSize,
		APBands: first.APBands, Frames: totalFrames, Channels: shared,
	}
	if len(indexes) == 1 {
		h.Source, h.CacheKey = first.Source, first.CacheKey
	}
	if err := h.validateShape(); err != nil {
		return nil, err
	}
	out := &Index{
		Header: h, Channels: make(map[string][]float32, len(shared)),
		Spectrum:     make([]float32, 0, totalFrames*SubFrames*SpectrumDims),
		Aperiodicity: make([]float32, 0, totalFrames*SubFrames*h.APBands),
		F0:           make([]float32, 0, totalFrames*SubFrames),
		FileID:       make([]int32, 0, totalFrames),
		Position:     make([]int32, 0, totalFrames),
	}
	for _, ch := range shared {
		out.Channels[ch.ID] = make([]float32, 0, totalFrames*ch.Dims)
	}
	for _, idx := range indexes {
		if err := out.appendFileIDs(idx); err != nil {
			return nil, err
		}
		out.Position = append(out.Position, idx.Position...)
		out.Spectrum = append(out.Spectrum, idx.Spectrum...)
		out.Aperiodicity = append(out.Aperiodicity, idx.Aperiodicity...)
		out.F0 = append(out.F0, idx.F0...)
		for _, ch := range shared {
			out.Channels[ch.ID] = append(out.Channels[ch.ID], idx.Channels[ch.ID]...)
		}
	}
	out.computeStatistics()
	return out, nil
}

func (out *Index) appendFileIDs(idx *Index) error {
	if len(idx.Files) > 0 {
		if len(idx.Files) > math.MaxInt32-len(out.Files) {
			return errors.New("index: too many merged files")
		}
		base := int32(len(out.Files))
		out.Files = append(out.Files, idx.Files...)
		for _, id := range idx.FileID {
			out.FileID = append(out.FileID, base+id)
		}
		return nil
	}
	// Group anonymous recordings by input index and file ID.
	fileIDs := make(map[int32]int32)
	for _, id := range idx.FileID {
		mapped, ok := fileIDs[id]
		if !ok {
			if len(out.Files) == math.MaxInt32 {
				return errors.New("index: too many merged files")
			}
			mapped = int32(len(out.Files))
			fileIDs[id] = mapped
			out.Files = append(out.Files, FileEntry{})
		}
		out.FileID = append(out.FileID, mapped)
	}
	return nil
}
