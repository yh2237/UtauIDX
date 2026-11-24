// Package vocoder renders matched speech-index frames with WORLD.
package vocoder

import (
	"math"

	"github.com/yh2237/UtauIDX/index"
	"github.com/yh2237/UtauIDX/matcher"
)

type BetaMode string

const (
	BetaFull     BetaMode = "full"
	BetaLowOrder BetaMode = "low"
	BetaMidOrder BetaMode = "mid"
)

type SynthConfig struct {
	UseSafeRefAP    bool
	APCliffGuard    bool
	BodySupportDB   float64
	LevelTrackDB    float64
	MaxSupportDB    float64
	CrossfadeMS     float64
	UnvoicedCFMS    float64
	PostfilterBeta  float64
	BetaMode        BetaMode
	HighOrderSmooth int
	ModBoost        float64
	NoSmooth        bool
	SpectrumSmooth  int
	PresenceDB      float64
	QuietLevelDB    float64
}

// AutoSynthConfig derives synthesis settings from the matching profile.
func AutoSynthConfig(profile matcher.AutoProfile) SynthConfig {
	if profile.IsRawPool && !profile.IsSinusoidal {
		return SynthConfig{
			LevelTrackDB:   10,
			CrossfadeMS:    30,
			PostfilterBeta: 0.40,
			BetaMode:       BetaFull,
			QuietLevelDB:   40,
		}
	}
	gap, isSin := profile.Gap, profile.IsSinusoidal
	cfg := SynthConfig{
		UseSafeRefAP:    gap >= 0.25,
		APCliffGuard:    isSin,
		HighOrderSmooth: 1,
		QuietLevelDB:    40,
		BodySupportDB:   5.5 * max(0, (gap-0.25)/0.75),
	}
	if isSin {
		cfg.BetaMode = BetaMidOrder
		cfg.PostfilterBeta = 0.40
		cfg.SpectrumSmooth = 1
		cfg.PresenceDB = 2.5
		cfg.CrossfadeMS = 18
		cfg.ModBoost = 0.25
	} else {
		cfg.BetaMode = BetaLowOrder
		cfg.PostfilterBeta = 0.40 + 0.08*gap
		cfg.NoSmooth = gap > 0.65
		cfg.PresenceDB = 0.5 + 1.2*gap
		cfg.CrossfadeMS = 22
		cfg.UnvoicedCFMS = 12
		cfg.ModBoost = 0.35
	}
	if gap >= 0.25 {
		cfg.LevelTrackDB = 10 - 2*gap
		cfg.MaxSupportDB = 4 + 7*gap
	} else {
		cfg.LevelTrackDB = 10 - 4.5*gap
		cfg.MaxSupportDB = 4 + 2*gap
		if isSin {
			cfg.MaxSupportDB = 5.5
		}
	}
	return cfg
}

type SynthRequest struct {
	ReferenceF0    []float64 // 5 ms F0 in Hz, with 0 for unvoiced rows
	OutputF0       []float64 // optional explicit 5 ms F0 in Hz
	ReferenceLevel []float64 // optional 5 ms dB relative to p95
	ReferenceAP    []float32 // optional 5 ms SafeRefAP
	CenterHz       float64   // values <= 0 use the bank F0 median
	PitchScale     float64   // 0 defaults to 1
	Config         *SynthConfig
	Unblended      bool // use rank one, retaining crossfades and any explicit positive beta
}

type Vocoder interface {
	Synthesize(idx *index.Index, match *matcher.Result, req SynthRequest) ([]float64, error)
}

func finite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}
