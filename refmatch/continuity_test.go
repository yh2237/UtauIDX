package refmatch

import (
	"math"
	"reflect"
	"testing"
)

func TestContinuityAcrossBlocks(t *testing.T) {
	const n, frames = 100, 70
	index := &Index{Header: Header{Frames: n, Dims: 1}, Features: make([]float32, n), File: make([]int32, n), Position: make([]int32, n), F0: make([]float32, n*subFrames)}
	for j := range n {
		index.Features[j] = 1
		index.Position[j] = int32(100 + j)
	}
	features := make([]float32, frames)
	for t := range features {
		features[t] = 1
	}
	got := index.nearestWithContinuity(features, frames, make([]bool, frames), 0.1)
	for frame, chosen := range got {
		want := []int{frame, frame + 1, frame + 2, frame + 3}
		if !reflect.DeepEqual(chosen, want) {
			t.Fatalf("frame %d: got %v want %v", frame, chosen, want)
		}
	}
	// Each request starts without selection history.
	again := index.nearestWithContinuity(features, frames, make([]bool, frames), 0.1)
	if !reflect.DeepEqual(got, again) {
		t.Fatal("selection history leaked between requests")
	}
}

func TestContinuityOnlyNextOriginalPosition(t *testing.T) {
	index := &Index{Header: Header{Frames: 7}, File: []int32{0, 0, 0, 0, 1, 0, 0}, Position: []int32{10, 11, 13, 9, 11, 10, 21}}
	// Repeat, reverse, skipped positions and a different recording must not receive a bonus.
	sim := []float32{0.5, 0.45, 0.5, 0.5, 0.5, 0.5, 0.44}
	for _, tt := range []struct {
		previous []int
		want     int
	}{
		{nil, 0}, {[]int{0}, 1}, {[]int{0, 0}, 1},
		{[]int{0, 6}, 1}, {[]int{5}, 1},
	} {
		got := index.topContextWithContinuity(sim, 0, 1, 0, tt.previous, 0.1)
		if got[0] != tt.want {
			t.Fatalf("previous %v: got %v want top %d", tt.previous, got, tt.want)
		}
	}
	for candidate := range index.Frames {
		values := make([]float32, index.Frames)
		values[6] = 0.5
		values[candidate] = 0.45
		got := index.topContextWithContinuity(values, 0, 1, 0, []int{0}, 0.1)
		want := 6
		if candidate == 1 {
			want = 1
		}
		if got[0] != want {
			t.Fatalf("candidate %d incorrectly received bonus: %v", candidate, got)
		}
	}
	// Duplicate history must not stack the bonus.
	values := []float32{0, 0.35, 0, 0, 0, 0, 0.5}
	if got := index.topContextWithContinuity(values, 0, 1, 0, []int{0, 0}, 0.1); got[0] != 6 {
		t.Fatalf("bonus stacked: %v", got)
	}
	// A lower-ranked previous candidate also qualifies.
	index.Position[6] = 20
	got := index.topContextWithContinuity(sim, 0, 1, 0, []int{6, 0}, 0.1)
	if got[0] != 1 {
		t.Fatalf("lower rank continuation: %v", got)
	}
}

// legacyTopContext preserves the pre-bonus selection as an independent zero-value oracle.
func legacyTopContext(index *Index, sim []float32, first, last, t int) []int {
	bestScore := make([]float32, 0, neighbors)
	best := make([]int, 0, neighbors)
	for j := range index.Frames {
		var total, count float32
		for delta := -contextFrames; delta <= contextFrames; delta++ {
			tt, jj := t+delta, j+delta
			if tt < first || tt >= last || jj < 0 || jj >= index.Frames || index.File[jj] != index.File[j] {
				continue
			}
			weight := float32(1)
			if delta != 0 {
				weight = contextWeight
			}
			total += weight * sim[(tt-first)*index.Frames+jj]
			count += weight
		}
		score := total / max(count, 1e-6)
		if len(best) == neighbors && score <= bestScore[neighbors-1] {
			continue
		}
		position := len(best)
		if position == neighbors {
			position--
		} else {
			best = append(best, 0)
			bestScore = append(bestScore, 0)
		}
		for position > 0 && bestScore[position-1] < score {
			best[position], bestScore[position] = best[position-1], bestScore[position-1]
			position--
		}
		best[position], bestScore[position] = j, score
	}
	return best
}

func TestZeroContinuityMatchesLegacy(t *testing.T) {
	for _, n := range []int{1, 3, 4, 19} {
		index := &Index{Header: Header{Frames: n}, File: make([]int32, n)}
		for j := range n {
			index.File[j] = int32(j / 5)
		}
		const frames = 71
		sim := make([]float32, frames*n)
		for j := range sim {
			sim[j] = float32((j*17+j/11)%23) / 23
		}
		for frame := range frames {
			want := legacyTopContext(index, sim, 0, frames, frame)
			got := index.topContextWithContinuity(sim, 0, frames, frame, []int{0}, 0)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("n=%d frame=%d: %v != %v", n, frame, got, want)
			}
		}
	}
}

func TestInvalidContinuity(t *testing.T) {
	index := &Index{Header: Header{Dims: 1}}
	for _, bonus := range []float64{-1, math.NaN(), math.Inf(1)} {
		_, err := index.Parameters(Request{Features: []float32{1}, Frames: 1, ReferenceF0: make([]float64, subFrames), ContinuityBonus: bonus})
		if err == nil {
			t.Fatalf("accepted %v", bonus)
		}
	}
}
