package backing

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpenBytesClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.bin")
	want := []byte("GGUF\x03\x00\x00\x00payload")
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}

	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := b.Len(); got != int64(len(want)) {
		t.Fatalf("Len = %d, want %d", got, len(want))
	}
	if got := string(b.Bytes()); got != string(want) {
		t.Fatalf("Bytes = %q, want %q", got, want)
	}
	if err := b.CheckUnchanged(path); err != nil {
		t.Fatalf("CheckUnchanged: %v", err)
	}
	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := b.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestOpenRejectsNonRegularAndEmpty(t *testing.T) {
	dir := t.TempDir()
	if _, err := Open(dir); err == nil {
		t.Fatal("Open(directory) succeeded, want error")
	}
	empty := filepath.Join(dir, "empty.gguf")
	if err := os.WriteFile(empty, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(empty); err == nil {
		t.Fatal("Open(empty) succeeded, want error")
	}
}

func TestCheckUnchangedDetectsModification(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.bin")
	if err := os.WriteFile(path, []byte("0123456789"), 0o600); err != nil {
		t.Fatal(err)
	}
	b, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer b.Close()

	if err := os.WriteFile(path, []byte("0123456789abcdef"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := b.CheckUnchanged(path); err == nil {
		t.Fatal("CheckUnchanged succeeded after modification, want error")
	}
}

func TestReadFileFallback(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.bin")
	want := []byte("fallback bytes")
	if err := os.WriteFile(path, want, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := readFile(path, int64(len(want)))
	if err != nil {
		t.Fatalf("readFile: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("readFile = %q, want %q", got, want)
	}
}
