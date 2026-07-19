// Package backing owns the read-only bytes of one model file for the whole
// lifetime of a model. The same bytes are used for GGUF parsing and for
// inference, so tensor descriptors never have to be paired with a reopened
// path.
package backing

import (
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

// Backing is an immutable read-only view of a model file.
type Backing struct {
	data    []byte
	unmap   func([]byte) error
	size    int64
	modTime time.Time
	closed  bool
}

// Open opens path as a read-only backing. It prefers a whole-file memory
// mapping and falls back to reading the file into memory when mapping is not
// available or fails.
func Open(path string) (*Backing, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("backing: stat %q: %w", path, err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("backing: %q is not a regular file", path)
	}
	if info.Size() <= 0 {
		return nil, fmt.Errorf("backing: %q is empty", path)
	}

	data, unmap, err := mapFile(path)
	if err != nil {
		data, err = readFile(path, info.Size())
		if err != nil {
			return nil, err
		}
		unmap = nil
	}
	return &Backing{
		data:    data,
		unmap:   unmap,
		size:    info.Size(),
		modTime: info.ModTime(),
	}, nil
}

// Bytes returns the backing bytes. The slice is valid until Close.
func (b *Backing) Bytes() []byte { return b.data }

// Len returns the file size in bytes.
func (b *Backing) Len() int64 { return b.size }

// CheckUnchanged verifies that the file on disk still has the size and
// modification time observed at Open. It reduces, but cannot eliminate, the
// risk of parsing bytes that no longer match the path.
func (b *Backing) CheckUnchanged(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("backing: stat %q: %w", path, err)
	}
	if info.Size() != b.size || !info.ModTime().Equal(b.modTime) {
		return fmt.Errorf("backing: %q changed while open", path)
	}
	return nil
}

// Close releases the backing. It is idempotent.
func (b *Backing) Close() error {
	if b.closed {
		return nil
	}
	b.closed = true
	data := b.data
	b.data = nil
	if b.unmap != nil {
		return b.unmap(data)
	}
	return nil
}

func readFile(path string, size int64) ([]byte, error) {
	if size > int64(maxInt()) {
		return nil, fmt.Errorf("backing: %q of %d bytes does not fit this platform", path, size)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("backing: open %q: %w", path, err)
	}
	defer f.Close()
	data := make([]byte, int(size))
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, fmt.Errorf("backing: read %q: %w", path, err)
	}
	return data, nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}

var errMappingUnsupported = errors.New("backing: memory mapping unsupported")
