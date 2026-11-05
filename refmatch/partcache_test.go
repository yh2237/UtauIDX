package refmatch

import (
	"encoding/binary"
	"errors"
	"runtime"
	"testing"
)

func partHeader(usable, rate, fftSize, dims, apBands, frames uint32) []byte {
	var buffer []byte
	buffer = append(buffer, []byte(partMagic)...)
	for _, value := range []uint32{partVersion, usable, rate, fftSize, dims, apBands, frames} {
		buffer = binary.LittleEndian.AppendUint32(buffer, value)
	}
	return buffer
}

func TestDecodePartRejectsHugeDeclaredSize(t *testing.T) {
	// 64Kフレーム×1024次元を宣言するが、中身はヘッダだけ。以前は256MB以上を
	// 確保してから読めないと分かっていた。
	data := partHeader(1, 16000, 1024, 1<<10, 1, 1<<16)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	_, err := decodePart(data)
	runtime.ReadMemStats(&after)
	if !errors.Is(err, errBadPart) {
		t.Fatalf("err = %v, want errBadPart", err)
	}
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("allocated %d bytes for a truncated part", allocated)
	}
}

func TestDecodePartRejectsInvalidUsableFields(t *testing.T) {
	for _, test := range []struct {
		name                            string
		rate, fftSize, dims, apBands, n uint32
	}{
		{"rate", 0, 1024, 1, 1, 1},
		{"fft size", 16000, 0, 1, 1, 1},
		{"dims", 16000, 1024, 0, 1, 1},
		{"frames", 16000, 1024, 1, 1, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			data := partHeader(1, test.rate, test.fftSize, test.dims, test.apBands, test.n)
			if _, err := decodePart(data); !errors.Is(err, errBadPart) {
				t.Fatalf("err = %v, want errBadPart", err)
			}
		})
	}
}

func TestPartCacheRejectsShortKey(t *testing.T) {
	cache := OpenPartCache(t.TempDir())
	if cache == nil {
		t.Fatal("cache was not created")
	}
	if _, ok := cache.Load(""); ok {
		t.Fatal("loaded an empty key")
	}
	if _, ok := cache.Load("a"); ok {
		t.Fatal("loaded a one-character key")
	}
	cache.Save("", fileFrames{})
	cache.Save("a", fileFrames{})
}
