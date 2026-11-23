package encoder

import (
	"fmt"

	"github.com/yh2237/gograd/autograd"
)

func (m *WhisperEncoder) forwardWindow(f *inference, mel []float32, start, count int) ([]float32, error) {
	c := m.config
	x := f.keep(f.upload(mel, 1, 2*c.Positions, c.Mels))
	x = f.conv(x, "conv1", c.Mels, c.Width, 3, 1, 1, true)
	x = f.finish(f.keep(autograd.GELU(x, false)))
	x = f.conv(x, "conv2", c.Width, c.Width, 3, 2, 1, true)
	x = f.keep(autograd.GELU(x, false))
	pos := f.weight("embed_positions.weight", c.Positions, c.Width)
	x = f.finish(f.keep(autograd.Add(x, pos)))

	for i := 0; i < m.layer; i++ {
		base := fmt.Sprintf("layers.%d", i)
		layer := transformerLayer{
			attention:     base + ".self_attn",
			attentionNorm: base + ".self_attn_layer_norm",
			feedNorm:      base + ".final_layer_norm",
			feedIn:        base + ".fc1",
			feedOut:       base + ".fc2",
			preNorm:       true,
			keyBias:       f.has(base + ".self_attn.k_proj.bias"),
		}
		x = f.transformer(x, layer, c.Heads, c.Intermediate, 1e-5, nil)
	}
	if m.layer == c.Layers {
		x = f.normalize(x, "layer_norm", c.Width, 0, 1e-5)
	}
	x = f.keep(autograd.Slice(x, 1, start, start+count))
	return x.ToHost()
}
