package encoder

import (
	"errors"
	"io"
	"sort"
	"sync"
)

const (
	SampleRate              = 16000
	DefaultFrameMS          = 20.0
	IDWavLMBasePlusL6       = "wavlm-base-plus-l6"
	IDWav2Vec2EspeakL20     = "wav2vec2-espeak-l20"
	IDWhisperSmallL12PosInv = "whisper-small-l12-posinv"
)

// Encoder extracts raw, row-major features from 16 kHz mono audio.
type Encoder interface {
	ID() string
	Dims() int
	FrameMS() float64
	Extract(wav16k []float32) (feats []float32, frames int, err error)
}

type Registry struct {
	mu       sync.RWMutex
	encoders map[string]Encoder
}

func NewRegistry(encoders ...Encoder) *Registry {
	r := &Registry{}
	for _, enc := range encoders {
		r.Register(enc)
	}
	return r
}

// Register associates an encoder with its ID, replacing any previous entry.
func (r *Registry) Register(enc Encoder) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.encoders == nil {
		r.encoders = make(map[string]Encoder)
	}
	r.encoders[enc.ID()] = enc
}

func (r *Registry) Get(id string) Encoder {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.encoders[id]
}

// IDs returns the registered IDs in lexical order.
func (r *Registry) IDs() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return sortedIDs(r.encoders)
}

// List returns encoders in the same order as IDs.
func (r *Registry) List() []Encoder {
	r.mu.RLock()
	defer r.mu.RUnlock()
	encoders := make([]Encoder, 0, len(r.encoders))
	for _, id := range sortedIDs(r.encoders) {
		encoders = append(encoders, r.encoders[id])
	}
	return encoders
}

func sortedIDs(encoders map[string]Encoder) []string {
	ids := make([]string, 0, len(encoders))
	for id := range encoders {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Close removes all registered encoders and closes those implementing io.Closer.
func (r *Registry) Close() error {
	r.mu.Lock()
	encoders := r.encoders
	r.encoders = nil
	r.mu.Unlock()

	var errs []error
	for _, id := range sortedIDs(encoders) {
		if closer, ok := encoders[id].(io.Closer); ok {
			errs = append(errs, closer.Close())
		}
	}
	return errors.Join(errs...)
}
