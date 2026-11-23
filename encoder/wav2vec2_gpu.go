package encoder

import (
	"fmt"
	"math"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

// CUDA operations and resource cleanup use a thread-local device context.
var gpuMu sync.Mutex
var lastDevice atomic.Value

// LastDevice reports the device used by the most recent successful extraction.
func LastDevice() string { return deviceName(&lastDevice) }

func deviceName(value *atomic.Value) string {
	name, _ := value.Load().(string)
	return name
}

func deviceLabel(device tensor.Device) string {
	switch device {
	case tensor.CPU:
		return "CPU"
	case tensor.CUDA:
		return "CUDA"
	case tensor.DirectML:
		return "DirectML"
	case tensor.Vulkan:
		return "Vulkan"
	default:
		return string(device)
	}
}

func gpuEnabled(device tensor.Device) bool {
	return os.Getenv("UTAUTTS_NO_GPU") == "" &&
		os.Getenv("UTAUTTS_NO_"+strings.ToUpper(string(device))) == ""
}

type modelWeights struct {
	mu         sync.Mutex
	file       *autograd.SafeTensorFile
	prefix     string
	states     map[tensor.Device]*gpuState
	lastDevice atomic.Value
	closed     bool
}

type gpuState struct {
	device  tensor.Device
	weights map[string]*autograd.Tensor
}

func (s *gpuState) close() {
	for _, w := range s.weights {
		w.Close()
	}
}

func openWeights(path, probe string, prefixes ...string) (*modelWeights, error) {
	file, err := autograd.OpenSafeTensorFile(path)
	if err != nil {
		return nil, err
	}
	for _, prefix := range prefixes {
		if _, err := file.Shape(prefix + probe); err == nil {
			return &modelWeights{file: file, prefix: prefix, states: make(map[tensor.Device]*gpuState)}, nil
		}
	}
	file.Close()
	return nil, fmt.Errorf("encoder: %s has no tensor matching %s under prefixes %q", path, probe, prefixes)
}

func (m *modelWeights) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	m.closed = true
	gpuMu.Lock()
	defer gpuMu.Unlock()
	for _, state := range m.states {
		state.close()
	}
	m.states = nil
	return m.file.Close()
}

type extraction func(tensor.Device, []float32) ([]float32, int, error)

func extractAccelerated(run extraction, wav []float32) ([]float32, int, error) {
	for _, device := range acceleratedDevices {
		if gpuEnabled(device) {
			if values, frames, err := run(device, wav); err == nil {
				return values, frames, nil
			}
		}
	}
	return run(tensor.CPU, wav)
}

func (m *modelWeights) run(device tensor.Device, frames int, forward func(*inference) ([]float32, error)) (values []float32, count int, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil, 0, fmt.Errorf("encoder: model is closed")
	}
	if device != tensor.CPU {
		if !gpuEnabled(device) {
			return nil, 0, fmt.Errorf("encoder: %s is disabled", deviceLabel(device))
		}
		gpuMu.Lock()
		defer gpuMu.Unlock()
		leave, deviceErr := enterGPUDevice(device)
		if deviceErr != nil {
			return nil, 0, deviceErr
		}
		defer leave()
		defer func() {
			if state := m.states[device]; err != nil && state != nil {
				state.close()
				delete(m.states, device)
			}
		}()
	}

	// Autograd reports dispatch failures by panicking. Convert those failures
	// to errors at the device boundary so Extract can retry on another device.
	defer func() {
		if failure := recover(); failure != nil {
			values, count = nil, 0
			if cause, ok := failure.(error); ok {
				err = fmt.Errorf("encoder: %s: %w", deviceLabel(device), cause)
			} else {
				err = fmt.Errorf("encoder: %s: %v", deviceLabel(device), failure)
			}
		}
	}()
	if frames > 0 {
		state := m.states[device]
		if state == nil {
			state = &gpuState{device, make(map[string]*autograd.Tensor)}
			m.states[device] = state
		}
		f := &inference{model: m, state: state}
		defer f.release()
		values, err = forward(f)
		if err != nil {
			return nil, 0, err
		}
	}
	m.lastDevice.Store(deviceLabel(device))
	lastDevice.Store(deviceLabel(device))
	return values, frames, nil
}

// inference owns the temporary tensors for one stage of a forward pass.
type inference struct {
	model     *modelWeights
	state     *gpuState
	temporary []*autograd.Tensor
}

func (f *inference) keep(t *autograd.Tensor) *autograd.Tensor {
	f.temporary = append(f.temporary, t)
	return t
}

func (f *inference) release() {
	for i := len(f.temporary) - 1; i >= 0; i-- {
		t := f.temporary[i]
		t.ReleaseGraph()
		t.Close()
	}
	f.temporary = f.temporary[:0]
}

// finish keeps the stage output and releases its dependency graph.
func (f *inference) finish(x *autograd.Tensor) *autograd.Tensor {
	x.RetainData()
	x.ReleaseGraph()
	for i := len(f.temporary) - 1; i >= 0; i-- {
		if t := f.temporary[i]; t != x {
			t.ReleaseGraph()
			t.Close()
		}
	}
	f.temporary = append(f.temporary[:0], x)
	return x
}

func (f *inference) upload(values []float32, shape ...int) *autograd.Tensor {
	t, err := autograd.New(values, shape, f.state.device, false)
	if err != nil {
		panic(err)
	}
	return t
}

func (f *inference) has(name string) bool {
	_, err := f.model.file.Shape(f.model.prefix + name)
	return err == nil
}

func (f *inference) read(name string) ([]float32, []int) {
	values, shape, err := f.model.file.ReadF32(f.model.prefix + name)
	if err != nil {
		panic(err)
	}
	return values, shape
}

func (f *inference) weight(name string, shape ...int) *autograd.Tensor {
	if w := f.state.weights[name]; w != nil {
		return w
	}
	values, actual := f.read(name)
	if !slices.Equal(shape, actual) {
		panic(fmt.Errorf("%s: shape %v, want %v", f.model.prefix+name, actual, shape))
	}
	w := f.upload(values, shape...)
	f.state.weights[name] = w
	return w
}

func (f *inference) conv(x *autograd.Tensor, name string, in, out, kernel, stride, pad int, bias bool) *autograd.Tensor {
	w := f.weight(name+".weight", out, in, kernel)
	var b *autograd.Tensor
	if bias {
		b = f.weight(name+".bias", out)
	}
	return f.keep(autograd.Conv1dGEMMStride(x, w, b, stride, pad, 1))
}

func (f *inference) normalize(x *autograd.Tensor, name string, width, groups int, eps float32) *autograd.Tensor {
	w := f.weight(name+".weight", width)
	b := f.weight(name+".bias", width)
	if groups > 0 {
		return f.keep(autograd.GroupNorm(x, w, b, groups, eps))
	}
	return f.keep(autograd.LayerNorm(x, w, b, eps))
}

func (f *inference) linear(x *autograd.Tensor, name string, in, out int, bias bool) *autograd.Tensor {
	w := f.weight(name+".weight", out, in)
	wt := f.keep(autograd.Transpose(w, 0, 1))
	product := f.keep(autograd.MatMul(x, wt))
	if bias {
		b := f.weight(name+".bias", out)
		return f.keep(autograd.Add(product, b))
	}
	return product
}

type transformerLayer struct {
	attention, attentionNorm, feedNorm, feedIn, feedOut string
	preNorm, keyBias                                    bool
}

func (f *inference) transformer(x *autograd.Tensor, layer transformerLayer, heads, intermediate int, eps float32, bias func(*autograd.Tensor) *autograd.Tensor) *autograd.Tensor {
	width := x.Shape[2]
	attentionInput := x
	if layer.preNorm {
		attentionInput = f.normalize(x, layer.attentionNorm, width, 0, eps)
	}
	var mask *autograd.Tensor
	if bias != nil {
		mask = bias(attentionInput)
	}
	attention := f.attention(attentionInput, layer.attention, heads, layer.keyBias, mask)
	x = f.keep(autograd.Add(x, attention))
	if !layer.preNorm {
		x = f.normalize(x, layer.attentionNorm, width, 0, eps)
	}
	x = f.finish(x)

	feedInput := x
	if layer.preNorm {
		feedInput = f.normalize(x, layer.feedNorm, width, 0, eps)
	}
	hidden := f.linear(feedInput, layer.feedIn, width, intermediate, true)
	hidden = f.keep(autograd.GELU(hidden, false))
	output := f.linear(hidden, layer.feedOut, intermediate, width, true)
	x = f.keep(autograd.Add(x, output))
	if !layer.preNorm {
		x = f.normalize(x, layer.feedNorm, width, 0, eps)
	}
	return f.finish(x)
}

func (f *inference) attention(x *autograd.Tensor, name string, heads int, keyBias bool, mask *autograd.Tensor) *autograd.Tensor {
	frames, width := x.Shape[1], x.Shape[2]
	headDim := width / heads
	project := func(projection string, bias bool) *autograd.Tensor {
		z := f.linear(x, name+projection, width, width, bias)
		z = f.keep(autograd.Reshape(z, 1, frames, heads, headDim))
		return f.keep(autograd.Permute(z, 0, 2, 1, 3))
	}
	q, k, v := project(".q_proj", true), project(".k_proj", keyBias), project(".v_proj", true)
	var context *autograd.Tensor
	if f.state.device == tensor.CUDA && mask == nil {
		context = f.keep(autograd.ScaledDotProductAttention(q, k, v, nil))
	} else {
		kt := f.keep(autograd.Transpose(k, 2, 3))
		scores := f.keep(autograd.MatMul(q, kt))
		scores = f.keep(autograd.MulScalar(scores, float32(1/math.Sqrt(float64(headDim)))))
		if mask != nil {
			scores = f.keep(autograd.Add(scores, mask))
		}
		weights := f.keep(autograd.Softmax(scores, 3))
		context = f.keep(autograd.MatMul(weights, v))
	}
	ordered := f.keep(autograd.Permute(context, 0, 2, 1, 3))
	ordered = f.keep(autograd.Reshape(ordered, 1, frames, width))
	return f.linear(ordered, name+".out_proj", width, width, true)
}
