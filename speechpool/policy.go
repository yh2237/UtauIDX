package speechpool

// ResolvePoolMode selects a layer for auto mode and preserves explicit choices.
func ResolvePoolMode(mode PoolMode, profile BankProfile, rawFrames int, rawC35 float64) PoolMode {
	switch mode {
	case PoolBabbleOnly, PoolMerge, PoolRawOnly:
		return mode
	}
	shortRecordings := profile.P90SpanMS < 1100
	if profile.P90SpanMS <= 0 {
		shortRecordings = profile.Topology == topologyCV
	}
	if shortRecordings || rawFrames < 8000 || rawC35 > 1.25 {
		return PoolBabbleOnly
	}
	return PoolRawOnly
}
