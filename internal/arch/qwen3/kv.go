package qwen3

import "fmt"

func mulU64(first uint64, rest ...uint64) (uint64, bool) {
	for _, n := range rest {
		if n != 0 && first > ^uint64(0)/n {
			return 0, false
		}
		first *= n
	}
	return first, true
}

// kvLayer holds one layer's F16 key and value caches in head-major layout:
// [kvHead][position][headDim] binary16 values.
type kvLayer struct {
	keys   []uint16
	values []uint16
}

// kvBytes returns the exact F16 KV payload for one session:
// layers * context * kvHeads * (headDim + headDim) * 2 bytes.
func kvBytes(cfg Config, contextSize int) (uint64, error) {
	if contextSize <= 0 || contextSize > cfg.ContextLength {
		return 0, fmt.Errorf("%w: context size %d (model maximum %d)",
			ErrContextLimit, contextSize, cfg.ContextLength)
	}
	perLayer, ok := mulU64(uint64(cfg.KVHeads), uint64(contextSize), uint64(cfg.HeadDim), 2)
	if !ok {
		return 0, fmt.Errorf("%w: KV cache size overflows", ErrContextLimit)
	}
	total, ok := mulU64(perLayer*2, uint64(cfg.Layers))
	if !ok {
		return 0, fmt.Errorf("%w: KV cache size overflows", ErrContextLimit)
	}
	return total, nil
}

// newKVCache allocates the F16 KV cache for one session. K and V are separate
// per-layer allocations so no individual slice is enormous on 32-bit targets.
func newKVCache(cfg Config, contextSize int) ([]kvLayer, error) {
	if _, err := kvBytes(cfg, contextSize); err != nil {
		return nil, err
	}
	perHead := uint64(contextSize) * uint64(cfg.HeadDim)
	if perHead*uint64(cfg.KVHeads) > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("%w: KV cache slice overflows platform int", ErrContextLimit)
	}
	layers := make([]kvLayer, cfg.Layers)
	for i := range layers {
		layers[i] = kvLayer{
			keys:   make([]uint16, cfg.KVHeads*contextSize*cfg.HeadDim),
			values: make([]uint16, cfg.KVHeads*contextSize*cfg.HeadDim),
		}
	}
	return layers, nil
}
