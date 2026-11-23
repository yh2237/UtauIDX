package index

import (
	"fmt"
	"io"
	"os"

	"github.com/yh2237/UtauIDX/speechpool"
)

func chooseIndexRate(entries []speechpool.Entry) int {
	const limit = 48000
	counts := make(map[int]int)
	hasHighRate := false
	for _, entry := range entries {
		rate, duration, err := wavInfo(entry.Path)
		if err != nil || duration < MinimumFileSec {
			continue
		}
		if rate <= limit {
			counts[rate]++
		} else {
			hasHighRate = true
		}
	}
	best, bestCount := 0, 0
	for rate, count := range counts {
		if count > bestCount || (count == bestCount && rate > best) {
			best, bestCount = rate, count
		}
	}
	if best == 0 && hasHighRate {
		return limit
	}
	return best
}

// wavInfo reads WAV chunk headers and seeks past PCM data.
func wavInfo(path string) (int, float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return 0, 0, err
	}
	if !stat.Mode().IsRegular() {
		return 0, 0, fmt.Errorf("index: WAV is not a regular file")
	}
	var riff [12]byte
	if _, err := io.ReadFull(file, riff[:]); err != nil {
		return 0, 0, err
	}
	if string(riff[:4]) != "RIFF" || string(riff[8:]) != "WAVE" {
		return 0, 0, fmt.Errorf("index: invalid RIFF/WAVE header")
	}

	var rate, channels int
	var format, bits uint16
	fmtFound, dataFound := false, false
	var samples int64
	for offset := int64(len(riff)); offset < stat.Size(); {
		var chunk [8]byte
		if _, err := io.ReadFull(file, chunk[:]); err != nil {
			return 0, 0, err
		}
		offset += int64(len(chunk))
		id := string(chunk[:4])
		size := int64(littleEndian.Uint32(chunk[4:]))
		end := offset + size
		if end > stat.Size() {
			return 0, 0, io.ErrUnexpectedEOF
		}

		switch {
		case id == "fmt ":
			var fields [40]byte
			if size < 16 {
				return 0, 0, fmt.Errorf("index: short WAV fmt chunk")
			}
			if _, err := io.ReadFull(file, fields[:16]); err != nil {
				return 0, 0, err
			}
			format = littleEndian.Uint16(fields[:])
			if format == 0xfffe {
				if size < int64(len(fields)) {
					return 0, 0, fmt.Errorf("index: short extensible WAV fmt chunk")
				}
				if _, err := io.ReadFull(file, fields[16:]); err != nil {
					return 0, 0, err
				}
				format = littleEndian.Uint16(fields[24:])
			}
			channels = int(littleEndian.Uint16(fields[2:]))
			rawRate := littleEndian.Uint32(fields[4:])
			if rawRate == 0 || rawRate > maxRate || channels == 0 {
				return 0, 0, fmt.Errorf("index: invalid WAV sample rate or channel count")
			}
			rate = int(rawRate)
			bits = littleEndian.Uint16(fields[14:])
			fmtFound = true
		case id == "data" && fmtFound:
			if !(format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32) ||
				format == 3 && (bits == 32 || bits == 64)) {
				return 0, 0, fmt.Errorf("index: unsupported WAV format %d with %d bits", format, bits)
			}
			samples = size / int64(bits/8)
			dataFound = true
		}
		next := end + size%2
		if next > stat.Size() {
			if id != "data" || !fmtFound {
				return 0, 0, io.ErrUnexpectedEOF
			}
			next = end
		}
		if _, err := file.Seek(next, io.SeekStart); err != nil {
			return 0, 0, err
		}
		offset = next
	}
	if !dataFound || samples%int64(channels) != 0 {
		return 0, 0, fmt.Errorf("index: missing or misaligned WAV data")
	}
	return rate, float64(samples/int64(channels)) / float64(rate), nil
}
