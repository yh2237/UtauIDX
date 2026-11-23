package encoder

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/autograd"
	"github.com/yh2237/gograd/tensor"
)

type Wav2Vec2Encoder struct {
	model       *modelWeights
	config      Wav2Vec2Config
	id          string
	layer       int
	doNormalize bool
}

func LoadWav2Vec2(weightsPath string, opts Wav2Vec2Options) (*Wav2Vec2Encoder, error) {
	config, err := loadWav2Vec2Config(weightsPath, opts.ConfigPath)
	if err != nil {
		return nil, err
	}
	if opts.Layer < 0 || opts.Layer > config.Layers {
		return nil, fmt.Errorf("encoder: wav2vec2 layer must be in 0..%d, got %d", config.Layers, opts.Layer)
	}

	var preprocessor struct {
		DoNormalize bool `json:"do_normalize"`
	}
	if opts.DoNormalize != nil {
		preprocessor.DoNormalize = *opts.DoNormalize
	} else if err := readConfig(weightsPath, opts.PreprocessorPath, "preprocessor_config.json", true, &preprocessor); err != nil {
		return nil, err
	}

	id := opts.ID
	if id == "" {
		switch {
		case config.ModelType == "wavlm" && opts.Layer == 6:
			id = IDWavLMBasePlusL6
		case config.HiddenSize == 1024 && opts.Layer == 20:
			id = IDWav2Vec2EspeakL20
		default:
			id = config.ModelType
		}
	}
	model, err := openWeights(weightsPath, "feature_extractor.conv_layers.0.conv.weight", "", "wav2vec2.", "hubert.", "wavlm.")
	if err != nil {
		return nil, err
	}
	return &Wav2Vec2Encoder{model, config, id, opts.Layer, preprocessor.DoNormalize}, nil
}

func (m *Wav2Vec2Encoder) ID() string { return m.id }
func (m *Wav2Vec2Encoder) Dims() int  { return m.config.HiddenSize }
func (m *Wav2Vec2Encoder) FrameMS() float64 {
	ms := 1000.0 / SampleRate
	for _, stride := range m.config.ConvStride {
		ms *= float64(stride)
	}
	return ms
}
func (m *Wav2Vec2Encoder) Close() error       { return m.model.Close() }
func (m *Wav2Vec2Encoder) LastDevice() string { return deviceName(&m.model.lastDevice) }

func (m *Wav2Vec2Encoder) Extract(wav []float32) ([]float32, int, error) {
	return extractAccelerated(m.ExtractOnDevice, wav)
}

func (m *Wav2Vec2Encoder) ExtractCPU(wav []float32) ([]float32, int, error) {
	return m.ExtractOnDevice(tensor.CPU, wav)
}

// ExtractOnDevice runs inference on one device and reports dispatch errors.
func (m *Wav2Vec2Encoder) ExtractOnDevice(device tensor.Device, wav []float32) ([]float32, int, error) {
	return m.model.run(device, m.config.frameCount(len(wav)), func(f *inference) ([]float32, error) {
		if m.doNormalize {
			wav = normalizeWaveform(wav)
		}
		return m.forward(f, wav)
	})
}

func normalizeWaveform(wav []float32) []float32 {
	var mean, variance float64
	for _, v := range wav {
		mean += float64(v)
	}
	mean /= float64(len(wav))
	for _, v := range wav {
		diff := float64(v) - mean
		variance += diff * diff
	}
	scale := 1 / math.Sqrt(variance/float64(len(wav))+1e-7)
	out := make([]float32, len(wav))
	for i, v := range wav {
		out[i] = float32((float64(v) - mean) * scale)
	}
	return out
}

func (m *Wav2Vec2Encoder) forward(f *inference, wav []float32) ([]float32, error) {
	c := m.config
	x := f.keep(f.upload(wav, 1, len(wav), 1))
	channels := 1
	for i, out := range c.ConvDim {
		base := fmt.Sprintf("feature_extractor.conv_layers.%d", i)
		x = f.conv(x, base+".conv", channels, out, c.ConvKernel[i], c.ConvStride[i], 0, c.ConvBias)
		if c.FeatNorm == "layer" {
			x = f.normalize(x, base+".layer_norm", out, 0, 1e-5)
		} else if i == 0 {
			x = f.normalize(x, base+".layer_norm", out, c.GroupNormGroups, 1e-5)
		}
		x = f.finish(f.keep(autograd.GELU(x, false)))
		channels = out
	}
	x = f.normalize(x, "feature_projection.layer_norm", channels, 0, c.LayerNormEps)
	x = f.linear(x, "feature_projection.projection", channels, c.HiddenSize, true)
	pos := m.positionalConv(f, x)
	x = f.keep(autograd.Add(x, pos))
	if !c.Stable {
		x = f.normalize(x, "encoder.layer_norm", c.HiddenSize, 0, c.LayerNormEps)
	}
	x = f.finish(x)

	var relative *autograd.Tensor
	if c.RelativeBias && m.layer > 0 {
		relative = m.relativePositionBias(f, x.Shape[1])
		defer relative.Close()
	}
	for i := 0; i < m.layer; i++ {
		base := fmt.Sprintf("encoder.layers.%d", i)
		layer := transformerLayer{
			attention:     base + ".attention",
			attentionNorm: base + ".layer_norm",
			feedNorm:      base + ".final_layer_norm",
			feedIn:        base + ".feed_forward.intermediate_dense",
			feedOut:       base + ".feed_forward.output_dense",
			preNorm:       c.Stable, keyBias: true,
		}
		var bias func(*autograd.Tensor) *autograd.Tensor
		if relative != nil {
			bias = func(input *autograd.Tensor) *autograd.Tensor {
				return m.gatedPositionBias(f, input, relative, layer.attention)
			}
		}
		x = f.transformer(x, layer, c.Heads, c.Intermediate, c.LayerNormEps, bias)
	}
	if c.Stable && m.layer == c.Layers {
		x = f.normalize(x, "encoder.layer_norm", c.HiddenSize, 0, c.LayerNormEps)
	}
	return x.ToHost()
}

func (m *Wav2Vec2Encoder) positionalConv(f *inference, x *autograd.Tensor) *autograd.Tensor {
	c := m.config
	width, kernel, groups := c.HiddenSize, c.PosKernel, c.PosGroups
	channels, frames := width/groups, x.Shape[1]
	bias := f.weight("encoder.pos_conv_embed.conv.bias", width)
	parts := make([]*autograd.Tensor, groups)
	for group := range parts {
		start, end := group*channels, (group+1)*channels
		w := m.positionalGroup(f, group)
		in, b := x, bias
		if groups != 1 {
			in = f.keep(autograd.Slice(x, 2, start, end))
			b = f.keep(autograd.Slice(bias, 0, start, end))
		}
		conv := f.keep(autograd.Conv1dGEMMStride(in, w, b, 1, kernel/2, 1))
		if kernel%2 == 0 {
			conv = f.keep(autograd.Slice(conv, 1, 0, frames))
		}
		parts[group] = conv
	}
	pos := parts[0]
	if groups > 1 {
		pos = f.keep(autograd.Concat(2, parts...))
	}
	return f.keep(autograd.GELU(pos, false))
}

func (m *Wav2Vec2Encoder) positionalGroup(f *inference, group int) *autograd.Tensor {
	const base = "encoder.pos_conv_embed.conv"
	key := fmt.Sprintf("%s.group.%d", base, group)
	if w := f.state.weights[key]; w != nil {
		return w
	}
	c := m.config
	width, kernel := c.HiddenSize, c.PosKernel
	channels := width / c.PosGroups
	var values []float32
	if f.has(base + ".weight") {
		values, _ = f.read(base + ".weight")
	} else {
		vKey, gKey := base+".weight_v", base+".weight_g"
		if !f.has(vKey) {
			vKey = base + ".parametrizations.weight.original1"
			gKey = base + ".parametrizations.weight.original0"
		}
		v, _ := f.read(vKey)
		g, shape := f.read(gKey)
		values = positionalWeights(v, g, shape, width, channels, kernel)
	}
	if len(values) != width*channels*kernel {
		panic(fmt.Errorf("%s: invalid weight shape", base))
	}
	groupSize := channels * channels * kernel
	for i := 0; i < c.PosGroups; i++ {
		name := fmt.Sprintf("%s.group.%d", base, i)
		if f.state.weights[name] == nil {
			f.state.weights[name] = f.upload(values[i*groupSize:(i+1)*groupSize], channels, channels, kernel)
		}
	}
	return f.state.weights[key]
}

func positionalWeights(v, g []float32, shape []int, width, channels, kernel int) []float32 {
	if len(v) != width*channels*kernel {
		panic(fmt.Errorf("encoder: invalid positional weight_v shape"))
	}
	// Hugging Face uses weight normalization across all axes except the kernel
	// axis. Some exports instead preserve one norm per output channel.
	tapNorm := len(g) == kernel && (len(g) != width || len(shape) == 1 || shape[0] == 1)
	if !tapNorm && len(g) != width {
		panic(fmt.Errorf("encoder: invalid positional weight_g shape %v", shape))
	}
	index := func(i int) int {
		if tapNorm {
			return i % kernel
		}
		return i / (channels * kernel)
	}
	squares := make([]float64, len(g))
	for i, value := range v {
		squares[index(i)] += float64(value) * float64(value)
	}
	scale := make([]float32, len(g))
	for i, square := range squares {
		if square != 0 {
			scale[i] = g[i] / float32(math.Sqrt(square))
		}
	}
	for i := range v {
		v[i] *= scale[index(i)]
	}
	return v
}

func (m *Wav2Vec2Encoder) relativePositionBias(f *inference, frames int) *autograd.Tensor {
	c := m.config
	embed, _ := f.read("encoder.layers.0.attention.rel_attn_embed.weight")
	if len(embed) != c.NumBuckets*c.Heads {
		panic(fmt.Errorf("encoder: invalid WavLM relative position embedding shape"))
	}
	offsets := make([]int, 2*frames-1)
	half := c.NumBuckets / 2
	exact := half / 2
	denominator := math.Log(float64(c.MaxBucketDistance) / float64(exact))
	for relative := 1 - frames; relative < frames; relative++ {
		sign, distance := 0, -relative
		if relative > 0 {
			sign, distance = half, relative
		}
		bucket := distance
		if distance >= exact {
			bucket = min(half-1, exact+int(math.Log(float64(distance)/float64(exact))/denominator*float64(half-exact)))
		}
		offsets[relative+frames-1] = sign + bucket
	}
	values := make([]float32, c.Heads*frames*frames)
	for h := 0; h < c.Heads; h++ {
		for i := 0; i < frames; i++ {
			row := values[(h*frames+i)*frames:][:frames]
			for j := range row {
				row[j] = embed[offsets[j-i+frames-1]*c.Heads+h]
			}
		}
	}
	return f.upload(values, 1, c.Heads, frames, frames)
}

func (m *Wav2Vec2Encoder) gatedPositionBias(f *inference, x, relative *autograd.Tensor, base string) *autograd.Tensor {
	c := m.config
	frames, headDim := x.Shape[1], c.HiddenSize/c.Heads
	hidden := f.keep(autograd.Reshape(x, 1, frames, c.Heads, headDim))
	hidden = f.keep(autograd.Permute(hidden, 0, 2, 1, 3))
	proj := f.linear(hidden, base+".gru_rel_pos_linear", headDim, 8, true)
	// DirectML and Vulkan lack the reduction and sigmoid operations. Only the
	// eight projected values per head and frame make the round trip to the host.
	projected, err := proj.ToHost()
	if err != nil {
		panic(err)
	}
	constant, _ := f.read(base + ".gru_rel_pos_const")
	if len(constant) != c.Heads {
		panic(fmt.Errorf("encoder: invalid WavLM gate constant shape"))
	}
	gates := make([]float32, c.Heads*frames)
	for i := range gates {
		row := projected[i*8:][:8]
		var first, second float32
		for j := 0; j < 4; j++ {
			first += row[j]
			second += row[j+4]
		}
		a := float32(1 / (1 + math.Exp(-float64(first))))
		b := float32(1 / (1 + math.Exp(-float64(second))))
		gates[i] = a*(b*constant[i/frames]-1) + 2
	}
	value := f.keep(f.upload(gates, 1, c.Heads, frames, 1))
	return f.keep(autograd.Mul(relative, value))
}
