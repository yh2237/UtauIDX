package encoder

import (
	"fmt"
	"math"

	"github.com/yh2237/gograd/tensor"
)

type WhisperOptions struct {
	ID string
	// Layer selects the hidden-state index. Zero selects layer 12, or the last
	// encoder layer for a checkpoint with fewer than 12 layers.
	Layer            int
	ConfigPath       string
	PreprocessorPath string
	// WindowFrames is the number of 20 ms frames in a sliding audio segment.
	// The default is 100. The middle half contributes to the result.
	WindowFrames int
}

type whisperConfig struct {
	Width        int `json:"d_model"`
	Heads        int `json:"encoder_attention_heads"`
	Intermediate int `json:"encoder_ffn_dim"`
	Layers       int `json:"encoder_layers"`
	Mels         int `json:"num_mel_bins"`
	Positions    int `json:"max_source_positions"`
}

type WhisperEncoder struct {
	model        *modelWeights
	config       whisperConfig
	id           string
	layer        int
	windowFrames int
	filters      [][]float64
}

func LoadWhisperEncoder(weightsPath string, opts WhisperOptions) (*WhisperEncoder, error) {
	var c whisperConfig
	if err := readConfig(weightsPath, opts.ConfigPath, "config.json", false, &c); err != nil {
		return nil, err
	}
	if c.Width <= 0 || c.Heads <= 0 || c.Width%c.Heads != 0 || c.Intermediate <= 0 ||
		c.Layers < 1 || c.Mels <= 0 || c.Positions < 4 || c.Positions > math.MaxInt/320 {
		return nil, fmt.Errorf("encoder: invalid Whisper encoder configuration")
	}
	if opts.ID == "" {
		opts.ID = IDWhisperSmallL12PosInv
	}
	if opts.Layer == 0 {
		opts.Layer = min(12, c.Layers)
	}
	if opts.Layer < 1 || opts.Layer > c.Layers {
		return nil, fmt.Errorf("encoder: Whisper layer must be in 1..%d, got %d", c.Layers, opts.Layer)
	}
	if opts.WindowFrames == 0 {
		opts.WindowFrames = min(100, c.Positions/4*4)
	}
	if opts.WindowFrames < 4 || opts.WindowFrames%4 != 0 || opts.WindowFrames > c.Positions {
		return nil, fmt.Errorf("encoder: Whisper window must be a multiple of 4 frames in 4..%d", c.Positions)
	}

	var preprocessor struct {
		Filters [][]float64 `json:"mel_filters"`
	}
	if err := readConfig(weightsPath, opts.PreprocessorPath, "preprocessor_config.json", true, &preprocessor); err != nil {
		return nil, err
	}
	filters := preprocessor.Filters
	if len(filters) == 0 {
		filters = slaneyMelFilters(c.Mels)
	} else if len(filters) == whisperBins {
		for i, row := range filters {
			if len(row) != c.Mels {
				return nil, fmt.Errorf("encoder: Whisper mel_filters row %d has %d values, want %d", i, len(row), c.Mels)
			}
		}
	} else if len(filters) == c.Mels {
		// Some Whisper checkpoints store the filters in mel-major order.
		filters = make([][]float64, whisperBins)
		for bin := range filters {
			filters[bin] = make([]float64, c.Mels)
		}
		for mel, row := range preprocessor.Filters {
			if len(row) != whisperBins {
				return nil, fmt.Errorf("encoder: Whisper mel_filters row %d has %d values, want %d", mel, len(row), whisperBins)
			}
			for bin, value := range row {
				filters[bin][mel] = value
			}
		}
	} else {
		return nil, fmt.Errorf("encoder: Whisper mel_filters has %d rows, want %d or %d", len(filters), whisperBins, c.Mels)
	}
	model, err := openWeights(weightsPath, "conv1.weight", "model.encoder.", "encoder.")
	if err != nil {
		return nil, err
	}
	return &WhisperEncoder{model, c, opts.ID, opts.Layer, opts.WindowFrames, filters}, nil
}

func (m *WhisperEncoder) ID() string         { return m.id }
func (m *WhisperEncoder) Dims() int          { return m.config.Width }
func (m *WhisperEncoder) FrameMS() float64   { return DefaultFrameMS }
func (m *WhisperEncoder) Close() error       { return m.model.Close() }
func (m *WhisperEncoder) LastDevice() string { return deviceName(&m.model.lastDevice) }

func (m *WhisperEncoder) Extract(wav []float32) ([]float32, int, error) {
	return extractAccelerated(m.ExtractOnDevice, wav)
}

func (m *WhisperEncoder) ExtractCPU(wav []float32) ([]float32, int, error) {
	return m.ExtractOnDevice(tensor.CPU, wav)
}

func (m *WhisperEncoder) ExtractOnDevice(device tensor.Device, wav []float32) ([]float32, int, error) {
	frames := WhisperFrameCount(len(wav))
	return m.model.run(device, frames, func(f *inference) ([]float32, error) {
		const samplesPerFrame = 2 * whisperHop
		center := m.windowFrames / 4
		step := m.windowFrames / 2
		segment := make([]float32, m.windowFrames*samplesPerFrame)
		width := m.config.Width
		values := make([]float32, frames*width)
		for f0 := 0; f0 < frames; f0 += step {
			clear(segment)
			start := (f0 - center) * samplesPerFrame
			end := min(len(wav), start+len(segment))
			offset := max(0, start)
			if offset < end {
				copy(segment[offset-start:], wav[offset:end])
			}
			mel := whisperLogMel(segment, 2*m.config.Positions, m.filters)
			take := min(step, frames-f0)
			rows, err := m.forwardWindow(f, mel, center, take)
			if err != nil {
				return nil, err
			}
			copy(values[f0*width:], rows)
			f.release()
		}
		return values, nil
	})
}
