package encoder

import (
	"encoding/binary"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

func requireDevice(t *testing.T, device tensor.Device) {
	t.Helper()
	if !gpuEnabled(device) {
		t.Skipf("%s disabled by environment", deviceLabel(device))
	}
	leave, err := enterGPUDevice(device)
	if err != nil {
		t.Skipf("%s unavailable: %v", deviceLabel(device), err)
	}
	leave()
}

func readF32Fixture(t *testing.T, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data)%4 != 0 {
		t.Fatalf("%s: unaligned float32 fixture", path)
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values
}

func assertFeatureParity(t *testing.T, got, want []float32, dims int, minCosine, maxRelativeRMS float64) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 || len(want)%dims != 0 {
		t.Fatalf("invalid parity shapes: got %d, want %d, width %d", len(got), len(want), dims)
	}
	frames := len(want) / dims
	var cosine, worst, errorSq, referenceSq, maxDiff float64
	worst = 1
	for frame := 0; frame < frames; frame++ {
		var dot, a2, b2 float64
		for i := frame * dims; i < (frame+1)*dims; i++ {
			a, b := float64(got[i]), float64(want[i])
			diff := a - b
			if math.IsInf(a, 0) || math.IsNaN(a) || math.IsInf(b, 0) || math.IsNaN(b) {
				t.Fatalf("non-finite value at %d: got %g, want %g", i, a, b)
			}
			dot += a * b
			a2 += a * a
			b2 += b * b
			errorSq += diff * diff
			maxDiff = max(maxDiff, math.Abs(diff))
		}
		rowCosine := 1.0
		if a2 != 0 && b2 != 0 {
			rowCosine = dot / math.Sqrt(a2*b2)
		} else if a2 != b2 {
			rowCosine = 0
		}
		cosine += rowCosine
		worst = min(worst, rowCosine)
		referenceSq += b2
	}
	meanCosine := cosine / float64(frames)
	relativeRMS := math.Sqrt(errorSq / max(referenceSq, math.SmallestNonzeroFloat64))
	t.Logf("frames=%d dims=%d meanCosine=%.9f minCosine=%.9f relativeRMS=%.6g maxAbs=%.6g", frames, dims, meanCosine, worst, relativeRMS, maxDiff)
	if meanCosine+1e-14 < minCosine || relativeRMS > maxRelativeRMS {
		t.Errorf("need mean cosine >= %.6f and relative RMS <= %.6g", minCosine, maxRelativeRMS)
	}
}

func TestExternalEncoderParity(t *testing.T) {
	weights := os.Getenv("ENC_WEIGHTS")
	if weights == "" {
		t.Skip("set ENC_WEIGHTS, ENC_TYPE, ENC_WAVE_F32 and ENC_REF_F32")
	}
	wavePath, refPath := os.Getenv("ENC_WAVE_F32"), os.Getenv("ENC_REF_F32")
	if wavePath == "" || refPath == "" {
		t.Fatal("set ENC_WAVE_F32 and ENC_REF_F32")
	}
	var layer int
	if value := os.Getenv("ENC_LAYER"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil {
			t.Fatalf("ENC_LAYER: %v", err)
		}
		layer = parsed
	}
	id := os.Getenv("ENC_ID")
	cfgPath, prePath := os.Getenv("ENC_CONFIG"), os.Getenv("ENC_PRE")
	var model deviceEncoder
	var err error
	switch modelType := os.Getenv("ENC_TYPE"); modelType {
	case "wav2vec2":
		model, err = LoadWav2Vec2(weights, Wav2Vec2Options{ID: id, Layer: layer, ConfigPath: cfgPath, PreprocessorPath: prePath})
	case "whisper":
		model, err = LoadWhisperEncoder(weights, WhisperOptions{ID: id, Layer: layer, ConfigPath: cfgPath, PreprocessorPath: prePath})
	default:
		t.Fatalf("ENC_TYPE must be wav2vec2 or whisper, got %q", modelType)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})
	wave, want := readF32Fixture(t, wavePath), readF32Fixture(t, refPath)
	for _, device := range []tensor.Device{tensor.CPU, tensor.CUDA} {
		t.Run(deviceLabel(device), func(t *testing.T) {
			if device != tensor.CPU {
				requireDevice(t, device)
			}
			got, frames, err := model.ExtractOnDevice(device, wave)
			if err != nil {
				t.Fatal(err)
			}
			if frames*model.Dims() != len(want) {
				t.Fatalf("%d frames of %d dimensions, want %d values", frames, model.Dims(), len(want))
			}
			assertFeatureParity(t, got, want, model.Dims(), 0.999, 0.045)
		})
	}
}
