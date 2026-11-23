package encoder

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	"github.com/yh2237/gograd/tensor"
)

type registryEncoder struct {
	Encoder
	closed bool
	err    error
}

func (e *registryEncoder) Close() error {
	e.closed = true
	return e.err
}

func TestRegistry(t *testing.T) {
	s := NewExternalStore()
	a := s.Encoder("a", 2, 20, nil)
	closeErr := errors.New("close failed")
	b := &registryEncoder{Encoder: s.Encoder("b", 3, 20, nil), err: closeErr}
	r := NewRegistry(b, a)
	if !slices.Equal(r.IDs(), []string{"a", "b"}) ||
		!slices.Equal(r.List(), []Encoder{a, b}) || r.Get("none") != nil || r.Get("b") != b {
		t.Fatalf("registry lookup, list or IDs are incorrect: %v", r.IDs())
	}
	replacement := s.Encoder("a", 5, 20, nil)
	r.Register(replacement)
	if got := r.Get("a"); got != replacement {
		t.Fatalf("Get(a) = %v, want replacement", got)
	}
	if err := r.Close(); !errors.Is(err, closeErr) || !b.closed || len(r.List()) != 0 {
		t.Fatalf("Close() = %v, closed = %v, IDs = %v", err, b.closed, r.IDs())
	}
}

func TestExternalStore(t *testing.T) {
	s := NewExternalStore()
	a := s.Encoder("a", 3, 20, ConvFrameCount)
	b := s.Encoder("b", 2, 20, WhisperFrameCount)
	c := s.Encoder("c", 1, 20, nil)
	wave := make([]float32, 721)
	wave[0] = 1
	key := AudioKey(wave)
	negativeZero := math.Float32frombits(1 << 31)
	if AudioKey([]float32{0}) == AudioKey([]float32{negativeZero}) || AudioKey(wave[:720]) == key {
		t.Fatal("audio key aliases a different waveform")
	}
	_, _, err := a.Extract(wave)
	var need *NeedInference
	if !errors.As(err, &need) || len(need.Jobs) != 1 || need.Jobs[0].Key != key ||
		need.Jobs[0].EncoderID != "a" || !slices.Equal(need.Jobs[0].Samples, wave) {
		t.Fatalf("missing result error = %v", err)
	}
	jobs := s.Collect(func() {
		for _, tc := range []struct {
			enc    Encoder
			frames int
		}{{a, 2}, {a, 2}, {b, 2}, {c, 0}} {
			values, frames, err := tc.enc.Extract(wave)
			if err != nil || frames != tc.frames || !slices.Equal(values, make([]float32, frames*tc.enc.Dims())) {
				t.Fatalf("%s returned %v, %d, %v", tc.enc.ID(), values, frames, err)
			}
		}
	})
	if len(jobs) != 3 || jobs[0].EncoderID != "a" || jobs[1].EncoderID != "b" || jobs[2].EncoderID != "c" {
		t.Fatalf("collected jobs = %v", jobs)
	}
	wave[0] = 2
	if jobs[0].Samples[0] != 1 || need.Jobs[0].Samples[0] != 1 {
		t.Fatal("jobs share the caller's buffer")
	}
	wave[0] = 1
	result := []float32{2, 3, 4, 5, 6, 7}
	if err := s.Provide("a", key, result, 3, 3); err == nil {
		t.Fatal("Provide accepted an invalid shape")
	}
	if err := s.Provide("a", key, result, 2, 3); err != nil {
		t.Fatal(err)
	}
	result[0] = 99
	jobs = s.Collect(func() {
		values, frames, err := a.Extract(wave)
		if err != nil || frames != 2 || len(values) != 6 || values[0] != 2 {
			t.Fatalf("provided result = %v, %d, %v", values, frames, err)
		}
		values[0] = 99
	})
	if next, _, _ := a.Extract(wave); len(jobs) != 0 || next[0] != 2 {
		t.Fatal("store exposed its buffer or queued a provided result")
	}
	s.Clear()
	if _, _, err := a.Extract(wave); !errors.As(err, &need) {
		t.Fatalf("Extract after Clear = %v", err)
	}
}

func TestFrameCounts(t *testing.T) {
	for _, tc := range []struct{ samples, conv, whisper int }{
		{0, 0, 0}, {1, 0, 1}, {399, 0, 1}, {400, 1, 1}, {479, 1, 1},
		{480, 1, 2}, {719, 1, 2}, {720, 2, 2}, {800, 2, 3}, {16000, 49, 50},
	} {
		if c, w := ConvFrameCount(tc.samples), WhisperFrameCount(tc.samples); c != tc.conv || w != tc.whisper {
			t.Errorf("%d samples: got %d/%d, want %d/%d", tc.samples, c, w, tc.conv, tc.whisper)
		}
	}
}

type testWeight struct {
	shape  []int
	values []float32
}

type testWeights map[string]testWeight

func (w testWeights) put(key string, shape ...int) {
	n := 1
	for _, dim := range shape {
		n *= dim
	}
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(0.27 * math.Sin(float64((i+1)*(len(key)+3))*0.713))
	}
	w[key] = testWeight{shape, values}
}

func (w testWeights) affine(key string, in, out int, bias bool) {
	w.put(key+".weight", out, in)
	if bias {
		w.put(key+".bias", out)
	}
}

func (w testWeights) norm(key string, width int) {
	w.put(key+".weight", width)
	for i := range w[key+".weight"].values {
		w[key+".weight"].values[i] += 1
	}
	w.put(key+".bias", width)
}

func (w testWeights) attention(key string, width int, keyBias bool) {
	w.affine(key+".q_proj", width, width, true)
	w.affine(key+".k_proj", width, width, keyBias)
	w.affine(key+".v_proj", width, width, true)
	w.affine(key+".out_proj", width, width, true)
}

func saveTestWeights(t *testing.T, w testWeights, prefix string, config any) string {
	t.Helper()
	base := filepath.Join(os.TempDir(), "utauidx-enc-test")
	if err := os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "fixture-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	keys := make([]string, 0, len(w))
	for key := range w {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	header := make(map[string]any)
	var data []byte
	for _, key := range keys {
		item := w[key]
		offset := len(data)
		for _, value := range item.values {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
		}
		header[prefix+key] = map[string]any{
			"dtype": "F32", "shape": item.shape, "data_offsets": []int{offset, len(data)},
		}
	}
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	file := binary.LittleEndian.AppendUint64(nil, uint64(len(headerBytes)))
	file = append(file, headerBytes...)
	file = append(file, data...)
	path := filepath.Join(dir, "model.safetensors")
	if err := os.WriteFile(path, file, 0600); err != nil {
		t.Fatal(err)
	}
	configBytes, err := json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), configBytes, 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

type deviceEncoder interface {
	Encoder
	ExtractCPU([]float32) ([]float32, int, error)
	ExtractOnDevice(tensor.Device, []float32) ([]float32, int, error)
	LastDevice() string
	Close() error
}

func testDeviceParity(t *testing.T, model deviceEncoder, wave []float32, frames int) {
	t.Helper()
	t.Cleanup(func() {
		if err := model.Close(); err != nil {
			t.Error(err)
		}
	})
	cpu, n, err := model.ExtractCPU(wave)
	if err != nil || n != frames || len(cpu) != n*model.Dims() || model.LastDevice() != "CPU" {
		t.Fatalf("CPU extraction: %d frames, %d values, device %q, %v", n, len(cpu), model.LastDevice(), err)
	}
	assertFeatureParity(t, cpu, cpu, model.Dims(), 1, 0)
	var energy float64
	for _, v := range cpu {
		energy += float64(v) * float64(v)
	}
	if energy == 0 || math.Abs(energy-float64(n)) < 1e-3 {
		t.Fatalf("expected raw non-unit rows, output squared norm = %g", energy)
	}
	for _, device := range acceleratedDevices {
		t.Run(deviceLabel(device), func(t *testing.T) {
			requireDevice(t, device)
			got, gotFrames, err := model.ExtractOnDevice(device, wave)
			if err != nil || gotFrames != frames || model.LastDevice() != deviceLabel(device) {
				t.Fatalf("%s extraction: %d frames, device %q, %v", device, gotFrames, model.LastDevice(), err)
			}
			assertFeatureParity(t, got, cpu, model.Dims(), 0.99999, 0.001)
		})
	}
	t.Run("CPUFallback", func(t *testing.T) {
		t.Setenv("UTAUTTS_NO_GPU", "1")
		values, gotFrames, err := model.Extract(wave)
		if err != nil || gotFrames != frames || !slices.Equal(values, cpu) || model.LastDevice() != "CPU" || LastDevice() != "CPU" {
			t.Fatalf("fallback: %d frames, device %q, %v", gotFrames, model.LastDevice(), err)
		}
	})
}

func testWave(size int) []float32 {
	wave := make([]float32, size)
	for i := range wave {
		wave[i] = float32(0.08 + 0.2*math.Sin(float64(i)*0.059) + 0.07*math.Cos(float64(i*i)*0.0011))
	}
	return wave
}

func TestWav2Vec2Synthetic(t *testing.T) {
	for _, tc := range []struct {
		name, featNorm, modelType, prefix string
		stable, convBias                  bool
		posKernel, layer                  int
	}{
		{"WavLMPostNorm", "group", "wavlm", "wavlm.", false, false, 4, 2},
		{"EspeakPreNormIntermediate", "layer", "wav2vec2", "wav2vec2.", true, true, 3, 1},
		{"EspeakPreNormFinal", "layer", "wav2vec2", "wav2vec2.", true, true, 3, 2},
		{"EspeakLayerZero", "layer", "wav2vec2", "wav2vec2.", true, true, 3, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const width, inter, heads, layers, groups = 4, 7, 2, 2, 2
			kernel := tc.posKernel
			c := Wav2Vec2Config{
				HiddenSize: width, Heads: heads, Intermediate: inter, Layers: layers,
				ConvDim: []int{3, 4}, ConvKernel: []int{5, 3}, ConvStride: []int{2, 2},
				ConvBias: tc.convBias, FeatNorm: tc.featNorm, Stable: tc.stable,
				ModelType: tc.modelType, PosKernel: kernel, PosGroups: groups, LayerNormEps: 1e-4,
				GroupNormGroups: 1, NumBuckets: 16, MaxBucketDistance: 32,
			}
			w := make(testWeights)
			w.put("feature_extractor.conv_layers.0.conv.weight", 3, 1, 5)
			w.put("feature_extractor.conv_layers.1.conv.weight", 4, 3, 3)
			w.norm("feature_extractor.conv_layers.0.layer_norm", 3)
			if tc.convBias {
				w.put("feature_extractor.conv_layers.0.conv.bias", 3)
				w.put("feature_extractor.conv_layers.1.conv.bias", 4)
				w.norm("feature_extractor.conv_layers.1.layer_norm", 4)
			}
			w.norm("feature_projection.layer_norm", 4)
			w.affine("feature_projection.projection", 4, width, true)
			if tc.stable {
				w.put("encoder.pos_conv_embed.conv.parametrizations.weight.original1", width, width/groups, kernel)
				w.put("encoder.pos_conv_embed.conv.parametrizations.weight.original0", width, 1, 1)
			} else {
				w.put("encoder.pos_conv_embed.conv.weight_v", width, width/groups, kernel)
				w.put("encoder.pos_conv_embed.conv.weight_g", 1, 1, kernel)
			}
			w.put("encoder.pos_conv_embed.conv.bias", width)
			w.norm("encoder.layer_norm", width)
			for i := 0; i < layers; i++ {
				base := fmt.Sprintf("encoder.layers.%d", i)
				w.attention(base+".attention", width, true)
				w.norm(base+".layer_norm", width)
				w.norm(base+".final_layer_norm", width)
				w.affine(base+".feed_forward.intermediate_dense", width, inter, true)
				w.affine(base+".feed_forward.output_dense", inter, width, true)
				if tc.modelType == "wavlm" {
					w.affine(base+".attention.gru_rel_pos_linear", width/heads, 8, true)
					w.put(base+".attention.gru_rel_pos_const", 1, heads, 1, 1)
				}
			}
			if tc.modelType == "wavlm" {
				w.put("encoder.layers.0.attention.rel_attn_embed.weight", c.NumBuckets, heads)
			}
			path := saveTestWeights(t, w, tc.prefix, c)
			if err := os.WriteFile(filepath.Join(filepath.Dir(path), "preprocessor_config.json"), []byte(fmt.Sprintf("{\"do_normalize\":%t}", tc.convBias)), 0600); err != nil {
				t.Fatal(err)
			}
			model, err := LoadWav2Vec2(path, Wav2Vec2Options{Layer: tc.layer})
			if err != nil {
				t.Fatal(err)
			}
			if model.Dims() != width || model.FrameMS() != 0.25 || model.doNormalize != tc.convBias {
				t.Fatalf("metadata: dims %d, frame ms %g, normalize %v", model.Dims(), model.FrameMS(), model.doNormalize)
			}
			testDeviceParity(t, model, testWave(61), 14)
			if values, frames, err := model.ExtractCPU(make([]float32, 8)); len(values) != 0 || frames != 0 || err != nil {
				t.Fatalf("short wave: %d frames, %d values, %v", frames, len(values), err)
			}
		})
	}
}

func TestWhisperSynthetic(t *testing.T) {
	const width, inter, heads, layers, mels, positions = 4, 7, 2, 2, 3, 16
	c := whisperConfig{width, heads, inter, layers, mels, positions}
	w := make(testWeights)
	w.put("conv1.weight", width, mels, 3)
	w.put("conv1.bias", width)
	w.put("conv2.weight", width, width, 3)
	w.put("conv2.bias", width)
	w.put("embed_positions.weight", positions, width)
	w.norm("layer_norm", width)
	for i := 0; i < layers; i++ {
		base := fmt.Sprintf("layers.%d", i)
		w.attention(base+".self_attn", width, false)
		w.norm(base+".self_attn_layer_norm", width)
		w.norm(base+".final_layer_norm", width)
		w.affine(base+".fc1", width, inter, true)
		w.affine(base+".fc2", inter, width, true)
	}
	path := saveTestWeights(t, w, "encoder.", c)
	model, err := LoadWhisperEncoder(path, WhisperOptions{WindowFrames: 8})
	if err != nil {
		t.Fatal(err)
	}
	if model.ID() != IDWhisperSmallL12PosInv || model.Dims() != width || model.FrameMS() != DefaultFrameMS {
		t.Fatalf("metadata: %q, %d dimensions, %g ms", model.ID(), model.Dims(), model.FrameMS())
	}
	testDeviceParity(t, model, testWave(2851), 9)
}
