package ops

// Argmax returns the index of the largest logit. Equal logits resolve to the
// lowest index, matching llama.cpp's greedy sampler. It returns -1 for an
// empty slice.
func Argmax(logits []float32) int {
	if len(logits) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}
