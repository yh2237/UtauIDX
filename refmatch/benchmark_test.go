package refmatch

import (
	"math"
	"math/rand"
	"testing"
)

func benchmarkIndex(frames, dims int) *Index {
	rng := rand.New(rand.NewSource(1))
	index := &Index{
		Header:   Header{Frames: frames, Dims: dims, SampleRate: 44100, FFTSize: 2048, APBands: 5, F0Median: 200},
		Features: make([]float32, frames*dims), F0: make([]float32, frames*subFrames),
		File: make([]int32, frames), Position: make([]int32, frames),
	}
	for i := range index.Features {
		index.Features[i] = float32(rng.NormFloat64())
	}
	for i := range index.F0 {
		index.F0[i] = float32(100 + rng.Float64()*100)
	}
	for i := range index.File {
		index.File[i] = int32(i / 64)
	}
	for i := range index.Position {
		index.Position[i] = int32(i)
	}
	return index
}

func BenchmarkNearest(b *testing.B) {
	const frames, dims, queryFrames = 4096, 64, 256
	index := benchmarkIndex(frames, dims)
	rng := rand.New(rand.NewSource(2))
	query := make([]float32, queryFrames*dims)
	for i := range query {
		query[i] = float32(rng.NormFloat64())
	}
	voiced := make([]bool, queryFrames)
	for i := range voiced {
		voiced[i] = i%3 != 0
	}
	// Warm the packed feature cache so the benchmark measures steady-state matching.
	index.nearest(query, queryFrames, voiced)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := index.nearest(query, queryFrames, voiced); len(got) != queryFrames {
			b.Fatalf("frames = %d", len(got))
		}
	}
}

func BenchmarkAverage(b *testing.B) {
	const frames, dims = 1024, 60
	index := benchmarkIndex(frames, dims)
	rng := rand.New(rand.NewSource(3))
	values := make([]float32, frames*subFrames*dims)
	for i := range values {
		values[i] = float32(rng.NormFloat64())
	}
	nearest := make([][]int, frames)
	for t := range nearest {
		first := max(0, min(frames-neighbors, t-neighbors/2))
		nearest[t] = []int{first, first + 1, first + 2, first + 3}
	}
	rows := frames * subFrames
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if got := index.average(values, dims, nearest, rows); len(got) != rows*dims {
			b.Fatalf("values = %d", len(got))
		}
	}
}

func BenchmarkCodecSpectrum(b *testing.B) {
	const fft, dims = 2048, 60
	c := newCodec(44100, fft, dims)
	bins := fft/2 + 1
	spectrum := make([]float64, bins)
	for i := range spectrum {
		spectrum[i] = 1 + math.Exp(-math.Pow(float64(i-100)/40, 2))
	}
	coded := make([]float64, dims)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.codeSpectrum(spectrum, coded)
	}
}

func BenchmarkCodecDecode(b *testing.B) {
	const fft, dims = 2048, 60
	c := newCodec(44100, fft, dims)
	bins := fft/2 + 1
	coded := make([]float64, dims)
	for i := range coded {
		coded[i] = math.Sin(float64(i) * 0.1)
	}
	spectrum := make([]float64, bins)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.decodeSpectrum(coded, spectrum)
	}
}
