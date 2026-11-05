package refmatch

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeWavInfoFile(t *testing.T, fmtSize uint32, fmtBody []byte, dataSize uint32) string {
	t.Helper()
	var buffer bytes.Buffer
	buffer.WriteString("RIFF")
	binary.Write(&buffer, binary.LittleEndian, uint32(0))
	buffer.WriteString("WAVE")
	buffer.WriteString("fmt ")
	binary.Write(&buffer, binary.LittleEndian, fmtSize)
	buffer.Write(fmtBody)
	if fmtSize%2 == 1 {
		buffer.WriteByte(0)
	}
	buffer.WriteString("data")
	binary.Write(&buffer, binary.LittleEndian, dataSize)
	buffer.Write(make([]byte, dataSize))
	out := buffer.Bytes()
	binary.LittleEndian.PutUint32(out[4:], uint32(len(out)-8))
	path := filepath.Join(t.TempDir(), "test.wav")
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestWavInfoReadsOddFmtChunk(t *testing.T) {
	body := make([]byte, 17)
	binary.LittleEndian.PutUint16(body[0:], 1)
	binary.LittleEndian.PutUint16(body[2:], 1)
	binary.LittleEndian.PutUint32(body[4:], 16000)
	binary.LittleEndian.PutUint32(body[8:], 32000)
	binary.LittleEndian.PutUint16(body[12:], 2)
	binary.LittleEndian.PutUint16(body[14:], 16)
	path := writeWavInfoFile(t, 17, body, 32000)
	rate, duration, err := wavInfo(path)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 16000 || duration != 1 {
		t.Fatalf("rate=%d duration=%v, want 16000 and 1", rate, duration)
	}
}

func TestWavInfoRejectsOversizedFmtChunk(t *testing.T) {
	body := make([]byte, 16)
	binary.LittleEndian.PutUint16(body[0:], 1)
	binary.LittleEndian.PutUint16(body[2:], 1)
	binary.LittleEndian.PutUint32(body[4:], 16000)
	binary.LittleEndian.PutUint32(body[8:], 32000)
	binary.LittleEndian.PutUint16(body[12:], 2)
	binary.LittleEndian.PutUint16(body[14:], 16)
	path := writeWavInfoFile(t, 1<<28, body, 0)
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	if _, _, err := wavInfo(path); err == nil {
		t.Fatal("accepted an oversized fmt chunk")
	}
	runtime.ReadMemStats(&after)
	if allocated := after.TotalAlloc - before.TotalAlloc; allocated > 8<<20 {
		t.Fatalf("allocated %d bytes for an oversized fmt chunk", allocated)
	}
}
