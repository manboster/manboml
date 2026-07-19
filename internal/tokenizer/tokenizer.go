// Package tokenizer builds a concrete Qwen2 byte-level BPE tokenizer from
// GGUF-embedded vocabulary data. The BPE engine is Ollama's pure-Go
// implementation; ManboML owns the data binding, validation, special-token
// policy and chat formatting.
package tokenizer

import (
	"errors"
	"fmt"
	"math"

	"github.com/manboster/manboml/internal/loader"
	oltok "github.com/ollama/ollama/tokenizer"
)

// qwen2PreTokenizePattern is the Qwen2 pre-tokenization pattern, matching
// llama.cpp and Hugging Face tokenizers. It requires lookahead support that
// Go's standard regexp engine does not provide, so Ollama's regexp2-based
// pre-tokenizer executes it.
const qwen2PreTokenizePattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

var (
	ErrUnsupportedTokenizer = errors.New("tokenizer: unsupported tokenizer")
	ErrInvalidVocabulary    = errors.New("tokenizer: invalid vocabulary")
)

// Tokenizer is an immutable Qwen2 byte-level BPE tokenizer bound to one
// model's vocabulary.
type Tokenizer struct {
	bpe      *oltok.BytePairEncoding
	eog      map[int32]struct{}
	template string
	bos      int32
	size     int
}

// New builds a Tokenizer from parsed GGUF tokenizer metadata. It requires the
// GPT-2/Qwen2 byte-level BPE family.
func New(meta loader.Tokenizer) (*Tokenizer, error) {
	if meta.Model != "gpt2" {
		return nil, fmt.Errorf("%w: model %q", ErrUnsupportedTokenizer, meta.Model)
	}
	if len(meta.Tokens) == 0 {
		return nil, fmt.Errorf("%w: empty vocabulary", ErrInvalidVocabulary)
	}
	if uint64(len(meta.Tokens)) > math.MaxInt32 {
		return nil, fmt.Errorf("%w: vocabulary of %d tokens overflows token IDs",
			ErrInvalidVocabulary, len(meta.Tokens))
	}

	types := meta.TokenTypes
	if len(types) != len(meta.Tokens) {
		types = synthesizeTypes(meta)
	}

	vocab := &oltok.Vocabulary{
		Values: meta.Tokens,
		Types:  types,
		Scores: meta.Scores,
		Merges: meta.Merges,
		AddBOS: meta.AddBOS,
		AddEOS: meta.AddEOS,
	}
	if meta.BOSTokenID >= 0 {
		vocab.BOS = []int32{int32(meta.BOSTokenID)}
	}
	eog := make(map[int32]struct{})
	for _, id := range []int64{meta.EOSTokenID, meta.EOTTokenID, meta.EOMTokenID} {
		if id >= 0 {
			if id > math.MaxInt32 {
				return nil, fmt.Errorf("%w: token ID %d out of range", ErrInvalidVocabulary, id)
			}
			eog[int32(id)] = struct{}{}
			vocab.EOS = append(vocab.EOS, int32(id))
		}
	}

	bpe := oltok.NewBytePairEncoding(vocab, qwen2PreTokenizePattern)
	t := &Tokenizer{
		bpe:      &bpe,
		eog:      eog,
		template: meta.Template,
		size:     len(meta.Tokens),
	}
	if meta.BOSTokenID >= 0 {
		t.bos = int32(meta.BOSTokenID)
	} else {
		t.bos = -1
	}
	return t, nil
}

// synthesizeTypes provides a fallback when the GGUF file lacks a token_type
// array: everything is normal except known special-token IDs.
func synthesizeTypes(meta loader.Tokenizer) []int32 {
	types := make([]int32, len(meta.Tokens))
	for i := range types {
		types[i] = 1
	}
	for _, id := range []int64{
		meta.BOSTokenID, meta.EOSTokenID, meta.EOTTokenID, meta.EOMTokenID,
	} {
		if id >= 0 && id < int64(len(types)) {
			types[id] = 3
		}
	}
	return types
}

// Encode tokenizes text. Special-token strings present in the vocabulary are
// recognized and encoded as their IDs. BOS/EOS are never added implicitly;
// prompt construction decides which special tokens appear.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	return t.bpe.Encode(text, false)
}

// Decode converts token IDs back to text. Special tokens decode to their
// literal vocabulary strings.
func (t *Tokenizer) Decode(ids []int32) (string, error) {
	return t.bpe.Decode(ids)
}

// IsEOG reports whether id is an end-of-generation token.
func (t *Tokenizer) IsEOG(id int32) bool {
	_, ok := t.eog[id]
	return ok
}

// BOS returns the beginning-of-sequence token ID, or -1 when the model does
// not define one.
func (t *Tokenizer) BOS() int32 { return t.bos }

// Size returns the vocabulary size.
func (t *Tokenizer) Size() int { return t.size }
