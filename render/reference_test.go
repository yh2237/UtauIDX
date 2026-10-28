package render

import (
	"math"
	"testing"
)

func TestPitchRatioAuto(t *testing.T) {
	reference := []float64{0, 200, 0, 400}
	bank := []float32{0, 300, 300, 0}
	ratio, referenceMedian, indexMedian, err := pitchRatio(reference, bank, nil)
	if err != nil {
		t.Fatalf("pitchRatio: %v", err)
	}
	if referenceMedian != 300 || indexMedian != 300 {
		t.Fatalf("medians = %v, %v", referenceMedian, indexMedian)
	}
	if math.Abs(ratio-1) > 1e-12 {
		t.Fatalf("ratio = %v, want 1", ratio)
	}
}

func TestPitchRatioClamps(t *testing.T) {
	semitones := 30.0
	ratio, _, _, err := pitchRatio([]float64{100}, []float32{100}, &semitones)
	if err != nil {
		t.Fatalf("pitchRatio: %v", err)
	}
	if math.Abs(ratio-2) > 1e-9 {
		t.Fatalf("ratio = %v, want 2", ratio)
	}

	reference := []float64{100}
	bank := []float32{1000}
	ratio, _, _, err = pitchRatio(reference, bank, nil)
	if err != nil {
		t.Fatalf("pitchRatio: %v", err)
	}
	if math.Abs(ratio-2) > 1e-9 {
		t.Fatalf("auto ratio = %v, want clamp to 2", ratio)
	}
}

func TestReferenceAPRowsUsesLowBands(t *testing.T) {
	fftSize := 512
	rate := 44100
	bins := fftSize/2 + 1
	ap := make([]float64, bins*2)
	lowBins := min(bins, int(1500*float64(fftSize)/float64(rate))+1)
	for row := range 2 {
		for bin := range bins {
			if bin < lowBins {
				ap[row*bins+bin] = 0.25
			} else {
				ap[row*bins+bin] = 1
			}
		}
	}
	rows, err := referenceAPRows(ap, rate, fftSize, 10, 4)
	if err != nil {
		t.Fatalf("referenceAPRows: %v", err)
	}
	if len(rows) != 4 {
		t.Fatalf("rows = %d, want 4", len(rows))
	}
	for _, value := range rows {
		if math.Abs(float64(value)-0.25) > 1e-6 {
			t.Fatalf("row value = %v, want 0.25", value)
		}
	}
}

func TestReferenceAPRowsRejectsBadShape(t *testing.T) {
	if _, err := referenceAPRows([]float64{0.1}, 44100, 512, 10, 4); err == nil {
		t.Fatal("expected an error for a mismatched AP length")
	}
}

func TestNormalizeFeatureRows(t *testing.T) {
	values := []float32{3, 4, 0, 0}
	normalizeFeatureRows(values, 2)
	if math.Abs(float64(values[0])-0.6) > 1e-6 || math.Abs(float64(values[1])-0.8) > 1e-6 {
		t.Fatalf("normalized row = %v", values[:2])
	}
	if values[2] != 0 || values[3] != 0 {
		t.Fatalf("zero row must stay zero: %v", values[2:])
	}
}
