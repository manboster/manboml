package ops

import "errors"

// AttentionHead computes single-query-head attention over seqLen cached
// positions stored as binary16 rows. query and dst must have length headDim,
// keys and values must hold at least seqLen*headDim F16 values, and scores
// must hold at least seqLen values. dst is overwritten, not accumulated.
func AttentionHead(dst, query []float32, keys, values []uint16, scores []float32, headDim, seqLen int, scale float32) error {
	if headDim <= 0 || seqLen <= 0 {
		return errors.New("ops: attention invalid dimensions")
	}
	if len(dst) != headDim || len(query) != headDim {
		return errors.New("ops: attention query length mismatch")
	}
	if len(keys) < seqLen*headDim || len(values) < seqLen*headDim {
		return errors.New("ops: attention KV cache too small")
	}
	if len(scores) < seqLen {
		return errors.New("ops: attention score buffer too small")
	}

	for pos := 0; pos < seqLen; pos++ {
		k := keys[pos*headDim : (pos+1)*headDim]
		var s float32
		for i := 0; i < headDim; i++ {
			s += query[i] * F16ToF32(k[i])
		}
		scores[pos] = s * scale
	}
	Softmax(scores[:seqLen])

	for i := range dst {
		dst[i] = 0
	}
	for pos := 0; pos < seqLen; pos++ {
		p := scores[pos]
		v := values[pos*headDim : (pos+1)*headDim]
		for i := 0; i < headDim; i++ {
			dst[i] += p * F16ToF32(v[i])
		}
	}
	return nil
}
