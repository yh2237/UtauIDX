package index

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/yh2237/UtauIDX/atomicfile"
)

const (
	worldMagic = "UTSW1"
	chanMagic  = "UTSC1"
)

type PartCache struct {
	dir string
}

type WorldPart struct {
	Usable                     bool
	Rate, FFTSize, APBands     int
	UsableFrames               int
	Spectrum, Aperiodicity, F0 []float32
	Position                   []int32
}

type ChannelPart struct {
	RawFrames, Dims int
	Values          []float32
}

type worldPartHeader struct {
	Magic                                  [5]byte
	Usable, Rate, FFTSize, APBands, Frames uint32
}

type channelPartHeader struct {
	Magic        [5]byte
	Frames, Dims uint32
}

func OpenPartCache(dir string) *PartCache {
	if dir == "" || os.MkdirAll(dir, 0o755) != nil {
		return nil
	}
	return &PartCache{dir: dir}
}

func worldKey(data []byte, targetRate int, startMS, endMS float64) string {
	return partKey(fmt.Sprintf("world\x00v1\x00%d\x00%g\x00%g\x00", targetRate, startMS, endMS), data)
}

func channelKey(data []byte, encoderID string) string {
	return partKey("chan\x00v1\x00"+encoderID+"\x00", data)
}

func partKey(prefix string, data []byte) string {
	hash := sha256.New()
	io.WriteString(hash, prefix)
	hash.Write(data)
	return hex.EncodeToString(hash.Sum(nil))
}

func (c *PartCache) path(key, kind string) string {
	return filepath.Join(c.dir, key[:2], key+"."+kind)
}

func (c *PartCache) read(key, kind string, decode func(io.Reader, int64) error) bool {
	path := c.path(key, kind)
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() || decode(bufio.NewReader(file), info.Size()) != nil {
		return false
	}
	now := time.Now()
	_ = os.Chtimes(path, now, now)
	return true
}

func (c *PartCache) write(key, kind string, encode func(io.Writer) error) {
	path := c.path(key, kind)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	_ = atomicfile.Write(path, func(w io.Writer) error {
		out := bufio.NewWriter(w)
		if err := encode(out); err != nil {
			return err
		}
		return out.Flush()
	})
}

func (c *PartCache) loadWorld(key string, rate int) (WorldPart, bool) {
	var part WorldPart
	ok := c.read(key, "world", func(r io.Reader, size int64) error {
		var err error
		part, err = decodeWorldPart(r, size)
		if err == nil && part.Rate != rate {
			return fmt.Errorf("index: cached WORLD sample rate %d, want %d", part.Rate, rate)
		}
		return err
	})
	return part, ok
}

func decodeWorldPart(r io.Reader, size int64) (WorldPart, error) {
	var h worldPartHeader
	if err := binary.Read(r, littleEndian, &h); err != nil {
		return WorldPart{}, err
	}
	rate, fftSize, apBands, frames := int(h.Rate), int(h.FFTSize), int(h.APBands), int(h.Frames)
	if string(h.Magic[:]) != worldMagic || h.Usable > 1 || (h.Usable == 1) != (frames > 0) ||
		frames < 0 || frames > maxFrames || !validWorldShape(rate, fftSize, apBands) {
		return WorldPart{}, fmt.Errorf("index: invalid WORLD part header")
	}
	width := SubFrames*(SpectrumDims+apBands+1) + 1
	payload, ok := checkedMul(int64(frames), int64(width*4))
	if !ok || payload != size-int64(binary.Size(h)) {
		return WorldPart{}, fmt.Errorf("index: invalid WORLD part size")
	}
	specLen, specOK := arrayLength(frames, SubFrames*SpectrumDims, 4)
	apLen, apOK := arrayLength(frames, SubFrames*apBands, 4)
	if !specOK || !apOK {
		return WorldPart{}, fmt.Errorf("index: WORLD part dimensions overflow")
	}
	part := WorldPart{Usable: h.Usable == 1, Rate: rate, FFTSize: fftSize, APBands: apBands, UsableFrames: frames}
	var err error
	if part.Spectrum, err = readArray[float32](r, specLen); err != nil {
		return WorldPart{}, err
	}
	if part.Aperiodicity, err = readArray[float32](r, apLen); err != nil {
		return WorldPart{}, err
	}
	if part.F0, err = readArray[float32](r, frames*SubFrames); err != nil {
		return WorldPart{}, err
	}
	if part.Position, err = readArray[int32](r, frames); err != nil {
		return WorldPart{}, err
	}
	last := int32(-1)
	for _, position := range part.Position {
		if position <= last || position >= maxFrames {
			return WorldPart{}, fmt.Errorf("index: invalid WORLD part position %d", position)
		}
		last = position
	}
	return part, nil
}

func (c *PartCache) saveWorld(key string, part WorldPart) {
	c.write(key, "world", func(w io.Writer) error {
		h := worldPartHeader{
			Magic: [5]byte([]byte(worldMagic)), Rate: uint32(part.Rate),
			FFTSize: uint32(part.FFTSize), APBands: uint32(part.APBands), Frames: uint32(part.UsableFrames),
		}
		if part.Usable {
			h.Usable = 1
		}
		if err := binary.Write(w, littleEndian, h); err != nil {
			return err
		}
		for _, values := range [][]float32{part.Spectrum, part.Aperiodicity, part.F0} {
			if err := writeArray(w, values); err != nil {
				return err
			}
		}
		return writeArray(w, part.Position)
	})
}

func (c *PartCache) loadChannel(key string, dims int) (ChannelPart, bool) {
	var part ChannelPart
	ok := c.read(key, "chan", func(r io.Reader, size int64) error {
		var err error
		part, err = decodeChannelPart(r, size, dims)
		return err
	})
	return part, ok
}

func decodeChannelPart(r io.Reader, size int64, dims int) (ChannelPart, error) {
	var h channelPartHeader
	if err := binary.Read(r, littleEndian, &h); err != nil {
		return ChannelPart{}, err
	}
	frames := int(h.Frames)
	if string(h.Magic[:]) != chanMagic || frames < 0 || frames > maxFrames ||
		dims <= 0 || dims > maxDims || h.Dims != uint32(dims) {
		return ChannelPart{}, fmt.Errorf("index: invalid channel part header")
	}
	n, ok := arrayLength(frames, dims, 4)
	if !ok || int64(n)*4 != size-int64(binary.Size(h)) {
		return ChannelPart{}, fmt.Errorf("index: invalid channel part size")
	}
	values, err := readArray[float32](r, n)
	if err != nil {
		return ChannelPart{}, err
	}
	return ChannelPart{RawFrames: frames, Dims: dims, Values: values}, nil
}

func (c *PartCache) saveChannel(key string, part ChannelPart) {
	c.write(key, "chan", func(w io.Writer) error {
		h := channelPartHeader{Magic: [5]byte([]byte(chanMagic)), Frames: uint32(part.RawFrames), Dims: uint32(part.Dims)}
		if err := binary.Write(w, littleEndian, h); err != nil {
			return err
		}
		return writeArray(w, part.Values)
	})
}
