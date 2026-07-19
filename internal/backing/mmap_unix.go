//go:build unix

package backing

import (
	"fmt"
	"os"
	"syscall"
)

func mapFile(path string) ([]byte, func([]byte) error, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("backing: open %q: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, nil, fmt.Errorf("backing: stat %q: %w", path, err)
	}
	size := info.Size()
	if size <= 0 || size > int64(maxInt()) {
		return nil, nil, fmt.Errorf("backing: %q of %d bytes cannot be mapped", path, size)
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, nil, fmt.Errorf("backing: mmap %q: %w", path, err)
	}
	return data, syscall.Munmap, nil
}
