package manboml

import (
	"os"
	"testing"
)

// realModelSHA256 is the pinned digest of the certified
// Qwen3Guard-Gen-0.6B.Q4_K_M.gguf artifact.
const realModelSHA256 = "a0d3385101ba362822d914ba40d9767aff634811ac99a41e4509de2d7a453b3e"

func realModelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("MANBOML_TEST_GGUF")
	if path == "" {
		t.Skip("MANBOML_TEST_GGUF not set")
	}
	return path
}
