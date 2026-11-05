package refmatch

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/yh2237/UtauIDX/atomicfile"
)

const (
	partMagic   = "UTTPART1"
	partVersion = 3
)

var errBadPart = errors.New("part cache is not usable")

// PartCacheは録音1本ぶんの解析結果をファイルに保存する。
// 索引を作り直すとき、内容が変わっていない録音を解析し直さないために使う。
type PartCache struct {
	dir string
}

// OpenPartCacheは保存先のフォルダを用意する。作れなければnilを返し、キャッシュを使わない。
func OpenPartCache(dir string) *PartCache {
	if dir == "" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil
	}
	return &PartCache{dir: dir}
}

// partKeyは解析条件と録音の中身と使う範囲から決まる。音源の場所が変わっても同じ録音なら使い回せる。
// 標本化周波数は索引ごとに揃えるため、解析条件に含める。
func partKey(model string, data []byte, rate int, start, end float64) string {
	hash := sha256.New()
	fmt.Fprintf(hash, "%s\x00%d\x00%d\x00%g\x00%g\x00", model, partVersion, rate, start, end)
	hash.Write(data)
	return hex.EncodeToString(hash.Sum(nil))
}

func (c *PartCache) path(key string) string {
	if len(key) < 2 {
		return ""
	}
	return filepath.Join(c.dir, key[:2], key+".part")
}

// Loadは保存した解析結果を返す。ないか読めなければfalse。
func (c *PartCache) Load(key string) (fileFrames, bool) {
	path := c.path(key)
	if path == "" {
		return fileFrames{}, false
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return fileFrames{}, false
	}
	frames, err := decodePart(data)
	if err != nil {
		return fileFrames{}, false
	}
	// 使った記録を残し、しばらく使っていないものだけを整理できるようにする。
	now := time.Now()
	_ = os.Chtimes(path, now, now)
	return frames, true
}

// Saveは解析結果を保存する。失敗しても索引の作成は続ける。
func (c *PartCache) Save(key string, frames fileFrames) {
	path := c.path(key)
	if path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	_ = atomicfile.WriteFile(path, encodePart(frames))
}

func encodePart(frames fileFrames) []byte {
	var buffer bytes.Buffer
	buffer.WriteString(partMagic)
	usable := uint32(0)
	if frames.usable {
		usable = 1
	}
	for _, value := range []uint32{
		partVersion, usable, uint32(frames.rate), uint32(frames.fftSize),
		uint32(frames.dims), uint32(frames.apBands), uint32(frames.frames),
	} {
		binary.Write(&buffer, binary.LittleEndian, value)
	}
	for _, values := range [][]float32{frames.features, frames.spectrum, frames.aperiodicity, frames.f0} {
		binary.Write(&buffer, binary.LittleEndian, values)
	}
	binary.Write(&buffer, binary.LittleEndian, frames.position)
	return buffer.Bytes()
}

func decodePart(data []byte) (fileFrames, error) {
	reader := bytes.NewReader(data)
	magic := make([]byte, len(partMagic))
	if _, err := io.ReadFull(reader, magic); err != nil || string(magic) != partMagic {
		return fileFrames{}, errBadPart
	}
	var header [7]uint32
	for i := range header {
		if err := binary.Read(reader, binary.LittleEndian, &header[i]); err != nil {
			return fileFrames{}, errBadPart
		}
	}
	version, usable, rate, fftSize, dims, apBands, frames := header[0], header[1], header[2], header[3], header[4], header[5], header[6]
	if version != partVersion || usable > 1 || frames > 1<<24 || dims > 1<<12 || apBands > 1<<10 {
		return fileFrames{}, errBadPart
	}
	if usable == 1 && (rate == 0 || rate > maxIndexRate || fftSize < 2 || fftSize > maxIndexFFTSize || fftSize%2 != 0 || dims == 0 || frames == 0) {
		return fileFrames{}, errBadPart
	}
	// 宣言された配列の長さを実際の残りと比べ、壊れたキャッシュで
	// 巨大な領域を確保する前に弾く。
	expected, ok := partPayloadSize(frames, dims, apBands)
	if !ok || expected > int64(reader.Len()) {
		return fileFrames{}, errBadPart
	}
	result := fileFrames{
		rate: int(rate), fftSize: int(fftSize), dims: int(dims), apBands: int(apBands),
		frames: int(frames), usable: usable == 1,
	}
	sizes := []int{
		int(frames) * int(dims),
		int(frames) * subFrames * spectrumDims,
		int(frames) * subFrames * int(apBands),
		int(frames) * subFrames,
	}
	targets := []*[]float32{&result.features, &result.spectrum, &result.aperiodicity, &result.f0}
	for i, size := range sizes {
		values := make([]float32, size)
		if err := binary.Read(reader, binary.LittleEndian, values); err != nil {
			return fileFrames{}, errBadPart
		}
		*targets[i] = values
	}
	result.position = make([]int32, frames)
	if err := binary.Read(reader, binary.LittleEndian, result.position); err != nil {
		return fileFrames{}, errBadPart
	}
	return result, nil
}

// partPayloadSizeは、キャッシュのヘッダが宣言する配列のバイト数を返す。
func partPayloadSize(frames, dims, apBands uint32) (int64, bool) {
	n := int64(frames)
	rows, ok := checkedMul(n, subFrames)
	if !ok {
		return 0, false
	}
	sizes := []struct{ count, width int64 }{
		{n, int64(dims) * 4},
		{rows, spectrumDims * 4},
		{rows, int64(apBands) * 4},
		{rows, 4},
		{n, 4},
	}
	var total int64
	for _, size := range sizes {
		bytes, ok := checkedMul(size.count, size.width)
		if !ok {
			return 0, false
		}
		total, ok = checkedAdd(total, bytes)
		if !ok {
			return 0, false
		}
	}
	return total, true
}
