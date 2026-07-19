package qwen3

import (
	"context"
	"fmt"
	"math"

	"github.com/manboster/manboml/internal/ops"
)

// Forward evaluates one token at the session's committed position. The
// position advances only after every layer succeeds, so a canceled or failed
// evaluation never becomes visible to later attention. When computeLogits is
// false, the final normalization and vocabulary projection are skipped; this
// is the fast path for non-final prefill tokens.
func (s *Session) Forward(ctx context.Context, token int32, computeLogits bool) error {
	m := s.model
	cfg := m.cfg
	if token < 0 || int(token) >= m.vocab {
		return fmt.Errorf("%w: token %d outside vocabulary", ErrInvalidMetadata, token)
	}
	if s.pos >= s.capacity {
		return fmt.Errorf("%w: position %d of %d", ErrContextLimit, s.pos, s.capacity)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ops.EmbeddingRow(s.hidden, m.weights.Embedding, uint32(token)); err != nil {
		return err
	}

	pos := s.pos
	scale := float32(1) / float32(math.Sqrt(float64(cfg.HeadDim)))
	for layer := range m.weights.Layers {
		if err := ctx.Err(); err != nil {
			return err
		}
		lw := &m.weights.Layers[layer]

		if err := ops.RMSNorm(s.norm, s.hidden, lw.AttnNorm, cfg.NormEps); err != nil {
			return err
		}
		qAct := s.q8.View(cfg.Hidden)
		ops.QuantizeQ8K(&qAct, s.norm)
		if err := ops.MatVecQ8K(ctx, m.executor, s.q, lw.Q, &qAct); err != nil {
			return err
		}
		if err := ops.MatVecQ8K(ctx, m.executor, s.k, lw.K, &qAct); err != nil {
			return err
		}
		if err := ops.MatVecQ8K(ctx, m.executor, s.v, lw.V, &qAct); err != nil {
			return err
		}
		if err := ops.RMSNormHeads(s.q, lw.QNorm, cfg.QueryHeads, cfg.HeadDim, cfg.NormEps); err != nil {
			return err
		}
		if err := ops.RMSNormHeads(s.k, lw.KNorm, cfg.KVHeads, cfg.HeadDim, cfg.NormEps); err != nil {
			return err
		}
		if err := m.rope.Apply(s.q, pos, cfg.QueryHeads, cfg.HeadDim); err != nil {
			return err
		}
		if err := m.rope.Apply(s.k, pos, cfg.KVHeads, cfg.HeadDim); err != nil {
			return err
		}

		kv := &s.kv[layer]
		for h := 0; h < cfg.KVHeads; h++ {
			row := (h*s.capacity + pos) * cfg.HeadDim
			ops.F32SliceToF16(kv.keys[row:row+cfg.HeadDim], s.k[h*cfg.HeadDim:(h+1)*cfg.HeadDim])
			ops.F32SliceToF16(kv.values[row:row+cfg.HeadDim], s.v[h*cfg.HeadDim:(h+1)*cfg.HeadDim])
		}

		seqLen := pos + 1
		for h := 0; h < cfg.QueryHeads; h++ {
			kvh := h * cfg.KVHeads / cfg.QueryHeads
			headOff := kvh * s.capacity * cfg.HeadDim
			if err := ops.AttentionHead(
				s.attn[h*cfg.HeadDim:(h+1)*cfg.HeadDim],
				s.q[h*cfg.HeadDim:(h+1)*cfg.HeadDim],
				kv.keys[headOff:headOff+seqLen*cfg.HeadDim],
				kv.values[headOff:headOff+seqLen*cfg.HeadDim],
				s.scores,
				cfg.HeadDim, seqLen, scale,
			); err != nil {
				return err
			}
		}

		oAct := s.q8.View(cfg.QueryWidth)
		ops.QuantizeQ8K(&oAct, s.attn)
		if err := ops.MatVecQ8K(ctx, m.executor, s.proj, lw.AttnOut, &oAct); err != nil {
			return err
		}
		if err := ops.AddInplace(s.hidden, s.proj); err != nil {
			return err
		}

		if err := ops.RMSNorm(s.norm, s.hidden, lw.FFNNorm, cfg.NormEps); err != nil {
			return err
		}
		fAct := s.q8.View(cfg.Hidden)
		ops.QuantizeQ8K(&fAct, s.norm)
		if err := ops.MatVecQ8K(ctx, m.executor, s.gate, lw.Gate, &fAct); err != nil {
			return err
		}
		if err := ops.MatVecQ8K(ctx, m.executor, s.up, lw.Up, &fAct); err != nil {
			return err
		}
		if err := ops.SwiGLU(s.gate, s.gate, s.up); err != nil {
			return err
		}
		gAct := s.q8.View(cfg.FFN)
		ops.QuantizeQ8K(&gAct, s.gate)
		if err := ops.MatVecQ8K(ctx, m.executor, s.proj, lw.Down, &gAct); err != nil {
			return err
		}
		if err := ops.AddInplace(s.hidden, s.proj); err != nil {
			return err
		}
	}

	if computeLogits {
		if err := ops.RMSNorm(s.norm, s.hidden, m.weights.OutputNorm, cfg.NormEps); err != nil {
			return err
		}
		fAct := s.q8.View(cfg.Hidden)
		ops.QuantizeQ8K(&fAct, s.norm)
		if err := ops.MatVecQ8K(ctx, m.executor, s.logits, m.weights.Output, &fAct); err != nil {
			return err
		}
	}

	s.pos++
	return nil
}

// Logits returns the vocabulary logits from the most recent Forward call
// with computeLogits=true.
func (s *Session) Logits() []float32 { return s.logits }
