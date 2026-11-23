package speechpool

import "testing"

func TestResolvePoolMode(t *testing.T) {
	for _, test := range []struct {
		name     string
		mode     PoolMode
		topology string
		frames   int
		c35      float64
		want     PoolMode
	}{
		{"CV", PoolAuto, "cv-diverse", 10000, 0.5, PoolBabbleOnly},
		{"small frame count", PoolAuto, "diphone-graph", 7999, 0.5, PoolBabbleOnly},
		{"sinusoidal C35", PoolAuto, "diphone-graph", 8000, 1.26, PoolBabbleOnly},
		{"large VCV at thresholds", "", "diphone-graph", 8000, 1.25, PoolRawOnly},
		{"explicit babble", PoolBabbleOnly, "diphone-graph", 10000, 0.5, PoolBabbleOnly},
		{"explicit merge", PoolMerge, "cv-diverse", 100, 1.75, PoolMerge},
		{"explicit raw", PoolRawOnly, "cv-diverse", 100, 1.75, PoolRawOnly},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := ResolvePoolMode(test.mode, BankProfile{Topology: test.topology}, test.frames, test.c35); got != test.want {
				t.Errorf("ResolvePoolMode = %q, want %q", got, test.want)
			}
		})
	}
}
