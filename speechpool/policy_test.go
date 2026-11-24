package speechpool

import "testing"

func TestResolvePoolMode(t *testing.T) {
	for _, test := range []struct {
		name     string
		mode     PoolMode
		topology string
		p90Span  float64
		frames   int
		c35      float64
		want     PoolMode
	}{
		{"CV fallback when span unset", PoolAuto, "cv-diverse", 0, 10000, 0.5, PoolBabbleOnly},
		{"human CVVC long span overrides cv-diverse", PoolAuto, "cv-diverse", 2800, 10000, 0.5, PoolRawOnly},
		{"short span forces babble even if diphone", PoolAuto, "diphone-graph", 739, 10000, 0.5, PoolBabbleOnly},
		{"small frame count", PoolAuto, "diphone-graph", 2000, 7999, 0.5, PoolBabbleOnly},
		{"sinusoidal C35", PoolAuto, "diphone-graph", 2000, 8000, 1.26, PoolBabbleOnly},
		{"large VCV at thresholds", "", "diphone-graph", 1100, 8000, 1.25, PoolRawOnly},
		{"explicit babble", PoolBabbleOnly, "diphone-graph", 2000, 10000, 0.5, PoolBabbleOnly},
		{"explicit merge", PoolMerge, "cv-diverse", 500, 100, 1.75, PoolMerge},
		{"explicit raw", PoolRawOnly, "cv-diverse", 500, 100, 1.75, PoolRawOnly},
	} {
		t.Run(test.name, func(t *testing.T) {
			profile := BankProfile{Topology: test.topology, P90SpanMS: test.p90Span}
			if got := ResolvePoolMode(test.mode, profile, test.frames, test.c35); got != test.want {
				t.Errorf("ResolvePoolMode = %q, want %q", got, test.want)
			}
		})
	}
}
