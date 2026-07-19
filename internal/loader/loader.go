// Package loader adapts the Ollama GGUF parser to ManboML's normalized,
// validated view of a model file. It is the only package that imports Ollama
// code; Ollama types must not leak into operator signatures or the public API.
package loader

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// maxArrayElements bounds each metadata array collected by the parser. It must
// cover the certified vocabulary and merge counts while remaining finite.
const maxArrayElements = 1 << 20

var (
	ErrInvalidFormat       = errors.New("loader: invalid GGUF format")
	ErrUnsupportedVersion  = errors.New("loader: unsupported GGUF version")
	ErrUnsupportedEndian   = errors.New("loader: unsupported byte order")
	ErrUnsupportedTensor   = errors.New("loader: unsupported tensor encoding")
	ErrDuplicateTensor     = errors.New("loader: duplicate tensor name")
	ErrMisalignedTensor    = errors.New("loader: misaligned tensor offset")
	ErrOverlappingTensor   = errors.New("loader: overlapping tensor data")
	ErrTensorOutOfBounds   = errors.New("loader: tensor data outside file")
	ErrInconsistentOffsets = errors.New("loader: inconsistent tensor data offset")
)

// Tensor is a normalized tensor descriptor. Offset and Size are absolute byte
// positions within the backing returned by Parse.
type Tensor struct {
	Name   string
	Kind   uint32
	Shape  []uint64
	Offset uint64
	Size   uint64
}

// Tokenizer holds the GGUF-embedded tokenizer data needed to build a concrete
// tokenizer. Missing token IDs are -1.
type Tokenizer struct {
	Model          string
	Pre            string
	Tokens         []string
	TokenTypes     []int32
	Scores         []float32
	Merges         []string
	Template       string
	AddBOS         bool
	AddEOS         bool
	BOSTokenID     int64
	EOSTokenID     int64
	EOTTokenID     int64
	EOMTokenID     int64
	UnknownTokenID int64
	PaddingTokenID int64
}

// Model is the normalized result of parsing a GGUF backing.
type Model struct {
	Architecture string
	Tensors      []Tensor
	Tokenizer    Tokenizer

	kv ggml.KV
}

// Parse parses and validates a GGUF file held in data. The returned tensor
// descriptors reference data and must not outlive it.
func Parse(data []byte) (*Model, error) {
	if err := validateHeader(data); err != nil {
		return nil, err
	}

	file, err := ggml.Decode(bytes.NewReader(data), maxArrayElements)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidFormat, err)
	}
	kv := file.KV()

	tensors, err := normalizeTensors(file, uint64(len(data)))
	if err != nil {
		return nil, err
	}

	m := &Model{
		Architecture: kv.Architecture(),
		Tensors:      tensors,
		Tokenizer:    extractTokenizer(kv),
		kv:           kv,
	}
	return m, nil
}

// Uint returns a numeric metadata value, resolving architecture-prefixed keys
// the same way the Ollama parser does.
func (m *Model) Uint(key string) (uint64, bool) {
	v, ok := m.kv[m.resolveKey(key)]
	if !ok {
		return 0, false
	}
	return numeric(v)
}

// String returns a string metadata value.
func (m *Model) String(key string) (string, bool) {
	v, ok := m.kv[m.resolveKey(key)]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// Float returns a floating-point metadata value.
func (m *Model) Float(key string) (float64, bool) {
	v, ok := m.kv[m.resolveKey(key)]
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case float32:
		return float64(n), true
	case float64:
		return n, true
	}
	return 0, false
}

func (m *Model) resolveKey(key string) string {
	if strings.HasPrefix(key, "general.") || strings.HasPrefix(key, "tokenizer.") {
		return key
	}
	return m.Architecture + "." + key
}

func validateHeader(data []byte) error {
	if len(data) < 16 {
		return fmt.Errorf("%w: file too small (%d bytes)", ErrInvalidFormat, len(data))
	}
	switch string(data[:4]) {
	case "GGUF":
	case "FUGG":
		return ErrUnsupportedEndian
	default:
		return fmt.Errorf("%w: bad magic %q", ErrInvalidFormat, data[:4])
	}
	switch version := binary.LittleEndian.Uint32(data[4:8]); version {
	case 2, 3:
	default:
		return fmt.Errorf("%w: %d", ErrUnsupportedVersion, version)
	}
	return nil
}

func normalizeTensors(file *ggml.GGML, backingSize uint64) ([]Tensor, error) {
	base := file.Tensors().Offset
	alignment := uint64(file.KV().Uint("general.alignment", 32))
	if alignment == 0 {
		alignment = 32
	}
	if base > backingSize {
		return nil, fmt.Errorf("%w: data section starts at %d of %d byte file",
			ErrTensorOutOfBounds, base, backingSize)
	}

	items := file.Tensors().Items()
	tensors := make([]Tensor, 0, len(items))
	seen := make(map[string]struct{}, len(items))
	ranges := make([][2]uint64, 0, len(items))

	for _, item := range items {
		if item == nil {
			return nil, fmt.Errorf("%w: nil tensor descriptor", ErrInvalidFormat)
		}
		if _, dup := seen[item.Name]; dup {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateTensor, item.Name)
		}
		seen[item.Name] = struct{}{}

		size := item.Size()
		if size == 0 && item.Elements() != 0 {
			return nil, fmt.Errorf("%w: %q kind %d", ErrUnsupportedTensor, item.Name, item.Kind)
		}
		if item.Offset%alignment != 0 {
			return nil, fmt.Errorf("%w: %q offset %d alignment %d",
				ErrMisalignedTensor, item.Name, item.Offset, alignment)
		}
		end, ok := checkedAdd3(base, item.Offset, size)
		if !ok || end > backingSize {
			return nil, fmt.Errorf("%w: %q range [%d, %d) of %d byte file",
				ErrTensorOutOfBounds, item.Name, base+item.Offset, end, backingSize)
		}

		shape := make([]uint64, len(item.Shape))
		copy(shape, item.Shape)
		tensors = append(tensors, Tensor{
			Name:   item.Name,
			Kind:   item.Kind,
			Shape:  shape,
			Offset: base + item.Offset,
			Size:   size,
		})
		ranges = append(ranges, [2]uint64{base + item.Offset, end})
	}

	slices.SortFunc(ranges, func(a, b [2]uint64) int {
		switch {
		case a[0] < b[0]:
			return -1
		case a[0] > b[0]:
			return 1
		default:
			return 0
		}
	})
	for i := 1; i < len(ranges); i++ {
		if ranges[i][0] < ranges[i-1][1] {
			return nil, fmt.Errorf("%w: range starting at %d", ErrOverlappingTensor, ranges[i][0])
		}
	}
	return tensors, nil
}

func extractTokenizer(kv ggml.KV) Tokenizer {
	return Tokenizer{
		Model:          kv.String("tokenizer.ggml.model"),
		Pre:            kv.String("tokenizer.ggml.pre"),
		Tokens:         kv.Strings("tokenizer.ggml.tokens"),
		TokenTypes:     kv.Ints("tokenizer.ggml.token_type"),
		Scores:         kv.Floats("tokenizer.ggml.scores"),
		Merges:         kv.Strings("tokenizer.ggml.merges"),
		Template:       kv.ChatTemplate(),
		AddBOS:         kv.Bool("tokenizer.ggml.add_bos_token"),
		AddEOS:         kv.Bool("tokenizer.ggml.add_eos_token"),
		BOSTokenID:     tokenID(kv, "tokenizer.ggml.bos_token_id"),
		EOSTokenID:     tokenID(kv, "tokenizer.ggml.eos_token_id"),
		EOTTokenID:     tokenID(kv, "tokenizer.ggml.eot_token_id"),
		EOMTokenID:     tokenID(kv, "tokenizer.ggml.eom_token_id"),
		UnknownTokenID: tokenID(kv, "tokenizer.ggml.unknown_token_id"),
		PaddingTokenID: tokenID(kv, "tokenizer.ggml.padding_token_id"),
	}
}

func tokenID(kv ggml.KV, key string) int64 {
	v, ok := kv[key]
	if !ok {
		return -1
	}
	n, ok := numeric(v)
	if !ok || n > uint64(1<<63-1) {
		return -1
	}
	return int64(n)
}

func numeric(v any) (uint64, bool) {
	switch n := v.(type) {
	case uint8:
		return uint64(n), true
	case int8:
		return uint64(int64(n)), n >= 0
	case uint16:
		return uint64(n), true
	case int16:
		return uint64(int64(n)), n >= 0
	case uint32:
		return uint64(n), true
	case int32:
		return uint64(int64(n)), n >= 0
	case uint64:
		return n, true
	case int64:
		return uint64(n), n >= 0
	}
	return 0, false
}

func checkedAdd3(a, b, c uint64) (uint64, bool) {
	if a > ^uint64(0)-b {
		return 0, false
	}
	ab := a + b
	if ab > ^uint64(0)-c {
		return 0, false
	}
	return ab + c, true
}
