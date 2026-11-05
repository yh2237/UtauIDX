package audio

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteWavRejectsEmptyPCMWithoutReplacingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "output.wav")
	if err := os.WriteFile(path, []byte("existing"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := WriteWav(path, &PCM{SampleRate: 16000, Channels: 1}); err == nil {
		t.Fatal("empty PCM accepted")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "existing" {
		t.Fatalf("failed write replaced the file: %q, %v", data, err)
	}
}

func TestWriteWavRejectsMisalignedPCMWithoutReplacingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "output.wav")
	if err := os.WriteFile(path, []byte("existing"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := WriteWav(path, &PCM{SampleRate: 16000, Channels: 2, Data: make([]int16, 3)}); err == nil {
		t.Fatal("misaligned PCM accepted")
	}
	data, err := os.ReadFile(path)
	if err != nil || string(data) != "existing" {
		t.Fatalf("failed write replaced the file: %q, %v", data, err)
	}
}

func TestWriteWavRejectsNilPCM(t *testing.T) {
	if err := WriteWav(filepath.Join(t.TempDir(), "nil.wav"), nil); err == nil {
		t.Fatal("nil PCM accepted")
	}
}

func TestPCMToWavBytesRejectsMisalignedPCM(t *testing.T) {
	if data := PCMToWavBytes(&PCM{SampleRate: 16000, Channels: 2, Data: make([]int16, 3)}); data != nil {
		t.Fatalf("encoded %d bytes of misaligned PCM", len(data))
	}
}
