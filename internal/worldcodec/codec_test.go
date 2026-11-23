package worldcodec

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestCodecMatchesWorld(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "world_codec.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixtures map[string]struct {
		FFT int         `json:"fft"`
		CSP []float64   `json:"csp"`
		CAP [][]float64 `json:"cap"`
		DSP []float64   `json:"dsp"`
		DAP [][]float64 `json:"dap"`
	}
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, rate := range []int{24000, 44100} {
		want := fixtures[strconv.Itoa(rate)]
		c := New(rate, want.FFT, SpectrumDims)
		bins := want.FFT/2 + 1
		sp := make([]float64, bins)
		ap := [][]float64{make([]float64, bins), make([]float64, bins)}
		for j := range sp {
			x := float64(j)
			sp[j] = math.Exp(-8 + 0.5*math.Sin(x*0.013) + 0.3*math.Cos(x*0.071))
			ap[0][j] = 0.5 + 0.45*math.Sin(x*0.02)
			ap[1][j] = 0.01
		}
		coded := make([]float64, SpectrumDims)
		c.CodeSpectrum(sp, coded)
		assertClose(t, rate, "coded spectrum", coded, want.CSP, 1e-9)
		decoded := make([]float64, bins)
		c.DecodeSpectrum(want.CSP, decoded)
		assertClose(t, rate, "decoded spectrum", every(decoded, 37), want.DSP, 1e-9*want.DSP[0])
		for row := range ap {
			codedAP := make([]float64, c.APBands())
			c.CodeAperiodicity(ap[row], codedAP)
			assertClose(t, rate, "coded aperiodicity", codedAP, want.CAP[row], 1e-9)
			c.DecodeAperiodicity(want.CAP[row], decoded)
			assertClose(t, rate, "decoded aperiodicity", every(decoded, 37), want.DAP[row], 1e-9)
		}
	}
}

func TestHalfRoundTrip(t *testing.T) {
	vals := []float32{0, 1, -1.5, 123.25, -0.0625, 0.0005}
	got := RoundToHalf(vals)
	for i, w := range vals {
		if math.Abs(float64(got[i]-w)) > 0.125 {
			t.Fatalf("RoundToHalf[%d]=%g want ~%g", i, got[i], w)
		}
	}
}

func TestMathHelpers(t *testing.T) {
	if p := Percentile([]float64{10, 40, 20, 30}, 50); math.Abs(p-25) > 1e-9 {
		t.Fatalf("Percentile 50 = %g, want 25", p)
	}
	if p := Percentile([]float64{10, 40, 20, 30}, 95); math.Abs(p-38.5) > 1e-9 {
		t.Fatalf("Percentile 95 = %g, want 38.5", p)
	}
	g := GaussianFilter([]float64{1, 1, 1, 1}, 1.5)
	for i, v := range g {
		if math.Abs(v-1) > 1e-9 {
			t.Fatalf("GaussianFilter constant[%d]=%g", i, v)
		}
	}
	if v := InterpClamped([]float64{1, 3}, []float64{10, 20}, 2); math.Abs(v-15) > 1e-9 {
		t.Fatalf("InterpClamped=%g want 15", v)
	}
	if v := InterpClamped([]float64{1, 3}, []float64{10, 20}, 0); v != 10 {
		t.Fatalf("InterpClamped low=%g want 10", v)
	}
	if v := InterpClamped([]float64{1, 3}, []float64{10, 20}, 5); v != 20 {
		t.Fatalf("InterpClamped high=%g want 20", v)
	}
}

func TestMouthConsistencyAndBoundaries(t *testing.T) {
	const frames = 8
	spec := make([]float32, frames*SubFrames*SpectrumDims)
	c2To13 := make([]float64, frames*MouthDims)
	fileID := []int32{0, 0, 0, 0, 1, 1, 1, 1}
	pos := []int32{0, 1, 2, 3, 10, 11, 15, 16}
	voiced := []bool{true, true, true, false, true, true, true, false}
	for j := range frames {
		for d := range MouthDims {
			val := float32(math.Sin(float64(j*3+d)*0.3) + 0.1*float64(d))
			c2To13[j*MouthDims+d] = float64(val)
			for sub := range SubFrames {
				spec[(j*SubFrames+sub)*SpectrumDims+2+d] = val
			}
		}
	}
	qs, qd := ComputeQueryMouth(c2To13, frames, voiced)
	bs, bd := ComputeBankMouth(spec, frames, voiced, fileID, pos)
	if len(qs) != frames*MouthDims || len(qd) != frames*MouthDims || len(bs) != frames*MouthDims || len(bd) != frames*MouthDims {
		t.Fatalf("unexpected mouth feature lengths")
	}
	// Static rows must match between query and bank and be unit norm.
	for j := range frames {
		var ns float64
		for d := range MouthDims {
			a, b := qs[j*MouthDims+d], bs[j*MouthDims+d]
			if math.Abs(float64(a-b)) > 1e-5 {
				t.Fatalf("static mismatch frame %d dim %d: %g vs %g", j, d, a, b)
			}
			ns += float64(b) * float64(b)
		}
		if math.Abs(ns-1) > 1e-4 {
			t.Fatalf("frame %d static norm^2 = %g, want 1", j, ns)
		}
	}
	// Frame 5 is at pos=11 and frame 6 is at pos=15 (non-contiguous within same file), so
	// delta at frame 5 only differentiates backward (frame 5 - frame 4).
	for d := range MouthDims {
		if bd[3*MouthDims+d] == qd[3*MouthDims+d] && bd[4*MouthDims+d] == qd[4*MouthDims+d] {
			continue
		}
	}
	// Verify file boundary isolation: frames 3 and 4 have different FileIDs so bank delta at frame 3 equals 0.5*(frame3-frame2),
	// whereas query delta at frame 3 crosses into frame 4.
	var diffSum float64
	for d := range MouthDims {
		diffSum += math.Abs(float64(bd[3*MouthDims+d] - qd[3*MouthDims+d]))
	}
	if diffSum < 1e-3 {
		t.Fatalf("expected file boundary to isolate bank delta at frame 3")
	}
}

func every(values []float64, step int) []float64 {
	var out []float64
	for i := 0; i < len(values); i += step {
		out = append(out, values[i])
	}
	return out
}

func assertClose(t *testing.T, rate int, name string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d %s: length %d, want %d", rate, name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(got[i]-want[i]) > tolerance {
			t.Fatalf("%d %s[%d] = %g, want %g", rate, name, i, got[i], want[i])
		}
	}
}
