package index

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/yh2237/UtauIDX/atomicfile"
	"github.com/yh2237/UtauIDX/internal/worldcodec"
)

const (
	maxHeaderBytes = 1 << 26
	blockElements  = 4096
)

var littleEndian = binary.LittleEndian

func QuantizeFeatures(values []float32, frames, dims int) ([]int8, []float32) {
	n, ok := arrayLength(frames, dims, 4)
	if !ok || dims <= 0 || len(values) != n {
		return nil, nil
	}
	quantized := make([]int8, n)
	scales := make([]float32, frames)
	for f := range frames {
		start := f * dims
		row := values[start : start+dims]
		var peak float32
		for _, value := range row {
			if absolute := float32(math.Abs(float64(value))); absolute > peak {
				peak = absolute
			}
		}
		if peak == 0 {
			scales[f] = 1
			continue
		}
		scale := max(peak/127, math.SmallestNonzeroFloat32)
		scales[f] = scale
		for i, value := range row {
			quantized[start+i] = int8(math.Round(float64(value / scale)))
		}
	}
	return quantized, scales
}

func DequantizeFeatures(quantized []int8, scales []float32, dims int) []float32 {
	n, ok := arrayLength(len(scales), dims, 4)
	if !ok || dims <= 0 || len(quantized) != n {
		return nil
	}
	values := make([]float32, n)
	for f, scale := range scales {
		start := f * dims
		for i, value := range quantized[start : start+dims] {
			values[start+i] = float32(value) * scale
		}
	}
	return values
}

func QuantizeRoundTrip(values []float32, dims int) []float32 {
	if dims <= 0 {
		return values
	}
	quantized, scales := QuantizeFeatures(values, len(values)/dims, dims)
	return DequantizeFeatures(quantized, scales, dims)
}

func normalizedChannel(values []float32, frames, dims int) []float32 {
	values = worldcodec.NormalizeRows(values, frames, dims)
	values = QuantizeRoundTrip(values, dims)
	normalizeChannelRows(values, dims)
	return values
}

func normalizeChannelRows(values []float32, dims int) {
	for start := 0; start < len(values); start += dims {
		row := values[start : start+dims]
		worldcodec.NormalizeInto(row, row)
	}
}

func expectedPayloadSize(h Header) (int64, bool) {
	width := int64(2*SubFrames*(SpectrumDims+h.APBands+1) + 12)
	for _, ch := range h.Channels {
		channelWidth := int64(ch.Dims) + 4
		if width > math.MaxInt64-channelWidth {
			return 0, false
		}
		width += channelWidth
	}
	return checkedMul(int64(h.Frames), width)
}

func (idx *Index) Save(path string) error {
	if err := idx.Validate(); err != nil {
		return err
	}
	if idx.Version != Version {
		return fmt.Errorf("index: unsupported version %d", idx.Version)
	}
	header, err := json.Marshal(idx.Header)
	if err != nil {
		return err
	}
	if len(header) > maxHeaderBytes {
		return fmt.Errorf("index: header exceeds %d bytes", maxHeaderBytes)
	}
	return atomicfile.Write(path, func(w io.Writer) error {
		out := bufio.NewWriter(w)
		if _, err := out.WriteString(Magic); err != nil {
			return err
		}
		if err := binary.Write(out, littleEndian, uint32(len(header))); err != nil {
			return err
		}
		if _, err := out.Write(header); err != nil {
			return err
		}
		for _, values := range [][]float32{idx.Spectrum, idx.Aperiodicity, idx.F0} {
			if err := writeHalf(out, values); err != nil {
				return err
			}
		}
		for _, values := range [][]int32{idx.FileID, idx.Position} {
			if err := writeArray(out, values); err != nil {
				return err
			}
		}
		if err := writeArray(out, idx.GlitchPenalty); err != nil {
			return err
		}
		for _, meta := range idx.Header.Channels {
			q, scales := QuantizeFeatures(idx.Channels[meta.ID], idx.Frames, meta.Dims)
			if err := writeArray(out, q); err != nil {
				return err
			}
			if err := writeArray(out, scales); err != nil {
				return err
			}
		}
		return out.Flush()
	})
}

func ReadHeader(path string) (Header, error) {
	file, err := os.Open(path)
	if err != nil {
		return Header{}, err
	}
	defer file.Close()
	h, _, err := readHeader(file)
	return h, err
}

func readHeader(file *os.File) (Header, int64, error) {
	var prefix [8]byte
	if _, err := io.ReadFull(file, prefix[:]); err != nil {
		return Header{}, 0, err
	}
	if string(prefix[:4]) != Magic {
		return Header{}, 0, fmt.Errorf("index: invalid magic %q", prefix[:4])
	}
	length := int64(littleEndian.Uint32(prefix[4:]))
	if length < 1 || length > maxHeaderBytes {
		return Header{}, 0, fmt.Errorf("index: invalid header length %d", length)
	}
	info, err := file.Stat()
	if err != nil {
		return Header{}, 0, err
	}
	remaining := info.Size() - int64(len(prefix)) - length
	if remaining < 0 {
		return Header{}, 0, fmt.Errorf("index: truncated header: %w", io.ErrUnexpectedEOF)
	}
	data := make([]byte, int(length))
	if _, err := io.ReadFull(file, data); err != nil {
		return Header{}, 0, err
	}
	var h Header
	if err := json.Unmarshal(data, &h); err != nil {
		return Header{}, 0, err
	}
	if h.Version != Version {
		return Header{}, 0, fmt.Errorf("index: unsupported version %d", h.Version)
	}
	if err := h.validateShape(); err != nil {
		return Header{}, 0, err
	}
	return h, remaining, nil
}

func Load(path string) (*Index, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	h, remaining, err := readHeader(file)
	if err != nil {
		return nil, err
	}
	expected, ok := expectedPayloadSize(h)
	if !ok {
		return nil, fmt.Errorf("index: payload size overflow")
	}
	if remaining < expected {
		return nil, fmt.Errorf("index: truncated payload (%d of %d bytes): %w", remaining, expected, io.ErrUnexpectedEOF)
	}
	in := bufio.NewReader(file)
	idx := &Index{Header: h, Channels: make(map[string][]float32, len(h.Channels))}
	rows := h.Frames * SubFrames
	if idx.Spectrum, err = readHalf(in, rows*SpectrumDims); err != nil {
		return nil, err
	}
	if idx.Aperiodicity, err = readHalf(in, rows*h.APBands); err != nil {
		return nil, err
	}
	if idx.F0, err = readHalf(in, rows); err != nil {
		return nil, err
	}
	if idx.FileID, err = readArray[int32](in, h.Frames); err != nil {
		return nil, err
	}
	if idx.Position, err = readArray[int32](in, h.Frames); err != nil {
		return nil, err
	}
	if idx.GlitchPenalty, err = readArray[float32](in, h.Frames); err != nil {
		return nil, err
	}
	for _, meta := range h.Channels {
		q, err := readArray[int8](in, h.Frames*meta.Dims)
		if err != nil {
			return nil, err
		}
		scales, err := readArray[float32](in, h.Frames)
		if err != nil {
			return nil, err
		}
		values := DequantizeFeatures(q, scales, meta.Dims)
		normalizeChannelRows(values, meta.Dims)
		idx.Channels[meta.ID] = values
	}
	if err := idx.Validate(); err != nil {
		return nil, err
	}
	idx.computeDerived()
	return idx, nil
}

type binaryElement interface {
	int8 | uint16 | int32 | float32
}

func writeArray[T binaryElement](w io.Writer, values []T) error {
	for start := 0; start < len(values); start += blockElements {
		if err := binary.Write(w, littleEndian, values[start:min(start+blockElements, len(values))]); err != nil {
			return err
		}
	}
	return nil
}

func readArray[T binaryElement](r io.Reader, n int) ([]T, error) {
	values := make([]T, n)
	for start := 0; start < n; start += blockElements {
		if err := binary.Read(r, littleEndian, values[start:min(start+blockElements, n)]); err != nil {
			return nil, err
		}
	}
	return values, nil
}

func writeHalf(w io.Writer, values []float32) error {
	var buffer [blockElements]uint16
	for start := 0; start < len(values); start += blockElements {
		end := min(start+blockElements, len(values))
		for i, value := range values[start:end] {
			buffer[i] = worldcodec.Float32ToHalf(value)
		}
		if err := binary.Write(w, littleEndian, buffer[:end-start]); err != nil {
			return err
		}
	}
	return nil
}

func readHalf(r io.Reader, n int) ([]float32, error) {
	values := make([]float32, n)
	var buffer [blockElements]uint16
	for start := 0; start < n; start += blockElements {
		block := buffer[:min(blockElements, n-start)]
		if err := binary.Read(r, littleEndian, block); err != nil {
			return nil, err
		}
		for i, value := range block {
			values[start+i] = worldcodec.HalfToFloat32(value)
		}
	}
	return values, nil
}
