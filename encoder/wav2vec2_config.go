package encoder

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

type Wav2Vec2Config struct {
	HiddenSize        int     `json:"hidden_size"`
	Layers            int     `json:"num_hidden_layers"`
	Heads             int     `json:"num_attention_heads"`
	Intermediate      int     `json:"intermediate_size"`
	ConvDim           []int   `json:"conv_dim"`
	ConvKernel        []int   `json:"conv_kernel"`
	ConvStride        []int   `json:"conv_stride"`
	ConvBias          bool    `json:"conv_bias"`
	FeatNorm          string  `json:"feat_extract_norm"`
	Stable            bool    `json:"do_stable_layer_norm"`
	PosKernel         int     `json:"num_conv_pos_embeddings"`
	PosGroups         int     `json:"num_conv_pos_embedding_groups"`
	ModelType         string  `json:"model_type"`
	LayerNormEps      float32 `json:"layer_norm_eps"`
	RelativeBias      bool    `json:"-"`
	GroupNormGroups   int     `json:"group_norm_groups"`
	NumBuckets        int     `json:"num_buckets"`
	MaxBucketDistance int     `json:"max_bucket_distance"`
}

type Wav2Vec2Options struct {
	ID string
	// Layer selects hidden_states[Layer], including index 0.
	Layer            int
	ConfigPath       string
	PreprocessorPath string
	DoNormalize      *bool
}

func readConfig(weightsPath, path, name string, optional bool, dest any) error {
	if path == "" {
		path = filepath.Join(filepath.Dir(weightsPath), name)
	} else {
		optional = false
	}
	data, err := os.ReadFile(path)
	if optional && os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, dest); err != nil {
		return fmt.Errorf("encoder: reading %s: %w", path, err)
	}
	return nil
}

func loadWav2Vec2Config(weightsPath, configPath string) (Wav2Vec2Config, error) {
	c := Wav2Vec2Config{
		ConvDim:    []int{512, 512, 512, 512, 512, 512, 512},
		ConvKernel: []int{10, 3, 3, 3, 3, 2, 2},
		ConvStride: []int{5, 2, 2, 2, 2, 2, 2},
		FeatNorm:   "group", PosKernel: 128, PosGroups: 16,
		ModelType: "wav2vec2", LayerNormEps: 1e-5,
		NumBuckets: 320, MaxBucketDistance: 800,
	}
	if err := readConfig(weightsPath, configPath, "config.json", false, &c); err != nil {
		return c, err
	}
	if c.LayerNormEps == 0 {
		c.LayerNormEps = 1e-5
	}
	if c.ModelType == "" {
		c.ModelType = "wav2vec2"
	}
	if c.NumBuckets == 0 {
		c.NumBuckets = 320
	}
	if c.MaxBucketDistance == 0 {
		c.MaxBucketDistance = 800
	}
	switch {
	case c.HiddenSize <= 0 || c.Heads <= 0 || c.HiddenSize%c.Heads != 0 ||
		c.Layers < 0 || c.Intermediate <= 0:
		return c, fmt.Errorf("encoder: invalid wav2vec2 transformer dimensions")
	case len(c.ConvDim) == 0 || len(c.ConvDim) != len(c.ConvKernel) || len(c.ConvDim) != len(c.ConvStride):
		return c, fmt.Errorf("encoder: wav2vec2 convolution configuration lengths must agree")
	case c.PosKernel <= 0 || c.PosGroups <= 0 || c.HiddenSize%c.PosGroups != 0:
		return c, fmt.Errorf("encoder: invalid wav2vec2 positional convolution dimensions")
	case c.FeatNorm != "layer" && c.FeatNorm != "group":
		return c, fmt.Errorf("encoder: unsupported wav2vec2 feature normalization %q", c.FeatNorm)
	case c.LayerNormEps <= 0 || math.IsNaN(float64(c.LayerNormEps)) || math.IsInf(float64(c.LayerNormEps), 0):
		return c, fmt.Errorf("encoder: invalid wav2vec2 layer_norm_eps %g", c.LayerNormEps)
	}
	for i, dim := range c.ConvDim {
		if dim <= 0 || c.ConvKernel[i] <= 0 || c.ConvStride[i] <= 0 {
			return c, fmt.Errorf("encoder: invalid wav2vec2 feature convolution %d dimensions", i)
		}
	}
	if c.GroupNormGroups == 0 {
		c.GroupNormGroups = c.ConvDim[0]
	}
	if c.GroupNormGroups < 0 || c.ConvDim[0]%c.GroupNormGroups != 0 {
		return c, fmt.Errorf("encoder: invalid wav2vec2 group_norm_groups %d", c.GroupNormGroups)
	}
	c.RelativeBias = c.ModelType == "wavlm"
	if c.RelativeBias && (c.NumBuckets < 4 || c.NumBuckets%2 != 0 || c.MaxBucketDistance <= c.NumBuckets/4) {
		return c, fmt.Errorf("encoder: invalid WavLM relative position bucket configuration")
	}
	return c, nil
}

func (c Wav2Vec2Config) frameCount(samples int) int {
	for i, kernel := range c.ConvKernel {
		if samples < kernel {
			return 0
		}
		samples = (samples-kernel)/c.ConvStride[i] + 1
	}
	return samples
}
