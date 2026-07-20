package manboml

// Options configures model opening. Zero values select documented defaults.
type Options struct {
	// ContextSize is the runtime context window in tokens. Zero uses 2048.
	ContextSize int
	// MaxConcurrent is the number of eagerly allocated inference sessions.
	// Zero uses one.
	MaxConcurrent int
	// Workers is the numerical worker count. Zero captures the current
	// GOMAXPROCS without changing it; one runs fully inline.
	Workers int
	// MemoryLimit is a conservative total memory budget in bytes covering
	// the model backing and all sessions. Zero means no caller-selected
	// limit; overflow and platform checks always apply.
	MemoryLimit uint64
	// ExpectedSHA256 optionally requires the model file's exact SHA-256
	// digest (hex, case-insensitive). Empty means structural compatibility
	// mode.
	ExpectedSHA256 string
}

func (o Options) withDefaults() Options {
	if o.ContextSize == 0 {
		o.ContextSize = 2048
	}
	if o.MaxConcurrent == 0 {
		o.MaxConcurrent = 1
	}
	return o
}
