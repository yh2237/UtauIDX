package encoder

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"slices"
	"sync"
)

type InferenceJob struct {
	EncoderID string    `json:"encoder_id"`
	Key       string    `json:"key"`
	Samples   []float32 `json:"-"`
}

type NeedInference struct {
	Jobs []InferenceJob
}

func (e *NeedInference) Error() string {
	return fmt.Sprintf("encoder: external inference required for %d job(s)", len(e.Jobs))
}

func AudioKey(wav16k []float32) string {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], uint64(len(wav16k)))
	h.Write(buf[:])
	for _, sample := range wav16k {
		binary.LittleEndian.PutUint32(buf[:4], math.Float32bits(sample))
		h.Write(buf[:4])
	}
	return fmt.Sprintf("f%d-%x", len(wav16k), h.Sum64())
}

func ConvFrameCount(samples int) int {
	if samples < 400 {
		return 0
	}
	return (samples-400)/320 + 1
}

func WhisperFrameCount(samples int) int {
	if samples <= 0 {
		return 0
	}
	return max(1, int(math.Round(float64(samples)/320.0)))
}

type externalKey struct {
	id, audio string
}

type externalResult struct {
	values       []float32
	frames, dims int
}

type ExternalStore struct {
	mu        sync.Mutex
	collectMu sync.Mutex
	results   map[externalKey]externalResult
	seen      map[externalKey]struct{}
	jobs      []InferenceJob
}

func NewExternalStore() *ExternalStore { return &ExternalStore{} }

func (s *ExternalStore) Provide(encoderID, key string, values []float32, frames, dims int) error {
	if encoderID == "" || key == "" || frames < 0 || dims <= 0 || frames > math.MaxInt/dims || len(values) != frames*dims {
		return fmt.Errorf("encoder: invalid external result for %q/%q (%d frames, %d dimensions, %d values)", encoderID, key, frames, dims, len(values))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.results == nil {
		s.results = make(map[externalKey]externalResult)
	}
	s.results[externalKey{encoderID, key}] = externalResult{slices.Clone(values), frames, dims}
	return nil
}

// Collect runs fn with missing inference lookups recorded in encounter order.
// Collection is store-wide. Calls to Collect on the same store are serialized.
func (s *ExternalStore) Collect(fn func()) (jobs []InferenceJob) {
	s.collectMu.Lock()
	defer s.collectMu.Unlock()
	s.mu.Lock()
	s.seen = make(map[externalKey]struct{})
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		jobs = s.jobs
		s.jobs = nil
		s.seen = nil
		s.mu.Unlock()
	}()
	fn()
	return
}

func (s *ExternalStore) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.results = nil
}

func (s *ExternalStore) Encoder(id string, dims int, frameMS float64, frameCount func(samples int) int) Encoder {
	return &externalEncoder{s, id, dims, frameMS, frameCount}
}

type externalEncoder struct {
	store      *ExternalStore
	id         string
	dims       int
	frameMS    float64
	frameCount func(int) int
}

func (e *externalEncoder) ID() string       { return e.id }
func (e *externalEncoder) Dims() int        { return e.dims }
func (e *externalEncoder) FrameMS() float64 { return e.frameMS }

func (e *externalEncoder) Extract(wav16k []float32) ([]float32, int, error) {
	key := externalKey{e.id, AudioKey(wav16k)}
	s := e.store
	s.mu.Lock()
	if result, ok := s.results[key]; ok {
		s.mu.Unlock()
		if result.dims != e.dims {
			return nil, 0, fmt.Errorf("encoder: %s external result has %d dimensions, want %d", e.id, result.dims, e.dims)
		}
		return slices.Clone(result.values), result.frames, nil
	}
	if s.seen == nil {
		s.mu.Unlock()
		job := InferenceJob{key.id, key.audio, slices.Clone(wav16k)}
		return nil, 0, &NeedInference{Jobs: []InferenceJob{job}}
	}
	if _, ok := s.seen[key]; !ok {
		s.seen[key] = struct{}{}
		s.jobs = append(s.jobs, InferenceJob{key.id, key.audio, slices.Clone(wav16k)})
	}
	s.mu.Unlock()
	if e.frameCount == nil {
		return nil, 0, nil
	}
	frames := e.frameCount(len(wav16k))
	if e.dims <= 0 || frames < 0 || frames > math.MaxInt/e.dims {
		return nil, 0, fmt.Errorf("encoder: invalid placeholder shape (%d frames, %d dimensions)", frames, e.dims)
	}
	return make([]float32, frames*e.dims), frames, nil
}
