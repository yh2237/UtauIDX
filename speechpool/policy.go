package speechpool

// ResolvePoolMode selects a layer for auto mode and preserves explicit choices.
func ResolvePoolMode(mode PoolMode, profile BankProfile, rawFrames int, rawC35 float64) PoolMode {
	switch mode {
	case PoolBabbleOnly, PoolMerge, PoolRawOnly:
		return mode
	}
	if profile.Topology == topologyCV || rawFrames < 8000 || rawC35 > 1.25 {
		return PoolBabbleOnly
	}
	return PoolRawOnly
}
