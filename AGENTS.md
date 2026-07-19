# ManboML Development Guide

This file is the persistent architecture record, implementation plan, and
development status for ManboML. Keep it current when a decision changes, a
milestone starts or completes, or verification results change.

## Current Status

- Phase: implementation in progress
- Updated: 2026-07-19
- Implementation authorization: granted incrementally by the user; parsing and
  operator modules are complete
- Repository contents at planning start: `README.md` only
- Latest repository commit at planning start: `468490f feat: init repo`
- Planning environment: Go 1.26.3, Darwin, arm64

Do not create the Go module, source packages, tests, CI, or implementation code
until the user explicitly asks to begin implementation.

## Product Intent

ManboML is a Go library that opens supported GGUF model files and performs text
inference inside the caller's process. Its public API is model-oriented and is
not tied to Qwen3Guard or Manboster. The first certified model happens to be the
Qwen3Guard model currently used by Manboster Hachimi.

Production binaries must build with `CGO_ENABLED=0` and run on the target
architecture without downloading or loading a shared library, FFI runtime,
native inference engine, helper process, or architecture-specific runtime
package. Inference calculations execute in code linked into the Go binary.

Pure-Go third-party modules are allowed. `unsafe` and Go assembly are not
categorically prohibited: they are acceptable when the compiled dependency
path supports every claimed target and has a working implementation or fallback
there. No optional optimization may be required for correctness. GGUF numerical
values must still be decoded with the file byte order rather than host-layout
casts so big-endian targets remain correct.

The optimization order is:

1. Numerical correctness and deterministic token behavior.
2. A complete real-model inference path.
3. Predictable memory admission and lifecycle behavior.
4. Full target-architecture runtime support.
5. Measured CPU performance.
6. Additional architectures, quantizations, and sampling features.

## Selected First Profile

The first certified profile is `qwen3guard-gen-0.6b-q4km-v1`:

- Model: Qwen3Guard-Gen-0.6B.
- Initial artifact: QuantFactory
  `Qwen3Guard-Gen-0.6B.Q4_K_M.gguf`.
- File size: 484,219,904 bytes, approximately 461.79 MiB.
- SHA-256: `a0d3385101ba362822d914ba40d9767aff634811ac99a41e4509de2d7a453b3e`.
- GGUF: single-file, little-endian v3, quantization version 2.
- Tensor inventory: 311 tensors consisting of 169 Q4_K, 29 Q6_K, and
  113 F32 tensors. Q4_K_M is a conversion recipe, not a tensor encoding.
- Architecture: Qwen3 decoder-only Transformer.
- Layers: 28.
- Hidden width: 1,024.
- FFN width: 3,072.
- Query heads: 16.
- KV heads: 8.
- Head width: 128.
- Query projection width: 2,048.
- Vocabulary: 151,936.
- Maximum model context: 32,768.
- Initial runtime contexts: 1,024, 2,048, 4,096, and 8,192.
- Default runtime context: 2,048.
- Default concurrent sessions: one.
- RoPE: NeoX layout, base 1,000,000, no scaling.
- Norms: RMSNorm plus per-head Q/K RMSNorm.
- FFN: SwiGLU.
- KV cache: F16 storage with F32 calculations.
- Tokenizer: GGUF-embedded Qwen2 byte-level BPE.
- Sampling: deterministic greedy decoding only in v0.1.
- Hachimi output bound: 128 new tokens.

The current Manboster model table assigns the Q2_K digest `aa114c...` to its
Q4_K_M entry. The downloader does not currently verify it. ManboML does not own
downloads, but callers may request strict SHA-256 verification when opening a
model.

## Explicit v0.1 Non-Goals

- Training, gradients, conversion, or quantization of model files.
- GPU, CUDA, Metal, Vulkan, external BLAS, native plugins, or FFI.
- A generic computation graph, backend registry, device abstraction, or model
  plugin API.
- Public token-level evaluation, logits, batch, session, KV-cache, tokenizer,
  sampler, or tensor APIs.
- A command-line program.
- Arbitrary Jinja execution from GGUF metadata.
- Temperature, top-k, top-p, repetition penalties, or sampler chains.
- Streaming result channels or callbacks.
- Multi-sequence batches, continuous batching, speculative decoding, prefix
  caching, or retained public chat sessions.
- Encoder models, multimodal models, MoE, LoRA, and adapters.
- GGUF model conversion, remote parsing, model downloading, or model discovery.
- Qwen3Guard-Stream, LLaMA, Gemma, or additional model architectures.
- Direct Manboster/Hachimi provider integration in this repository.
- llama.cpp-class performance before a correct baseline exists.

## Repository Architecture

```text
open.go                 model opening and architecture dispatch
model.go                public Model lifecycle and model information
generate.go             raw-prompt generation API
chat.go                 semantic-message chat API
options.go              public options and resource limits
result.go               Result and FinishReason
errors.go               stable public error categories

internal/backing/       long-lived mmap or bounded read fallback
internal/loader/        Ollama GGUF adapter and normalized tensor descriptors
internal/ops/           all portable numerical operators
internal/qwen3/         Qwen3 weights, forward path, private sessions and KV
internal/tokenizer/     Qwen2 BPE, special tokens, decoding and formatters
```

There is no public `gguf`, `tensor`, `model/qwen3`, or `ops` package in v0.1.
There is no `pkg/` or `src/` directory.

Do not add packages named `engine`, `backend`, `device`, `graph`, `runtime`,
`common`, or `utils` without a concrete second use case.

Dependency direction:

```text
manboml
  -> internal/loader
  -> internal/qwen3
  -> internal/tokenizer

internal/loader
  -> internal/backing
  -> github.com/ollama/ollama/fs/ggml

internal/qwen3
  -> internal/loader
  -> internal/ops

internal/ops
  -> standard library only unless profiling justifies a portable dependency
```

Ollama types must not appear in the public API or in numerical operator
signatures. `internal/loader` is the only package that imports Ollama.

## GGUF Parser Decision

Use `github.com/ollama/ollama/fs/ggml`, pinned to a reviewed stable tag. The
research snapshot used Ollama v0.32.1; re-run the compatibility review before
pinning the implementation dependency.

`fs/ggml` was selected instead of Ollama `fs/gguf` because it:

- Accepts an `io.ReadSeeker`, allowing parsing from the same long-lived backing
  used by inference.
- Exposes the tensor data-section offset.
- Exposes each tensor's name, kind, shape, relative offset, and encoded size.
- Exposes tokenizer and architecture metadata through `KV`.
- Compiles with `CGO_ENABLED=0` for all current Go 1.26 Linux architectures and
  the required Darwin and Windows targets.

Only the Go parser package is used. ManboML must not import Ollama runners,
servers, native libraries, GPU packages, or llama.cpp integration code.

The loading shape is:

```text
path
  -> internal/backing opens one immutable byte backing
  -> bytes.NewReader(backing.Bytes())
  -> ggml.Decode(reader, bounded tokenizer-array limit)
  -> normalize required metadata and tensor descriptors
  -> absolute tensor range = Tensors.Offset + Tensor.Offset + Tensor.Size
  -> validate the supported Qwen3 contract and every absolute range
  -> retain backing; release unneeded parser metadata
```

The tokenizer vocabulary and merges must be collected, so the Ollama decoder's
array limit must be at least the certified vocabulary and merge counts. It must
remain bounded rather than using unlimited array collection.

The loader must independently enforce the invariants needed for inference even
when the parser has already checked them:

- Supported magic and GGUF version.
- Little-endian tensor execution for the first profile.
- Supported architecture and quantization version.
- Required metadata values and tokenizer identity.
- Duplicate required tensor detection.
- Tensor rank, dimensions, element products, block divisibility, type, encoded
  byte size, alignment, absolute offset, overlap, and backing bounds.
- Exact Qwen3 required tensor set for the certified profile.
- File identity and immutability during `Open` where the platform exposes the
  required information.

Malformed or incompatible parser output must become a ManboML error rather than
an operator panic. Ollama's memory-estimation APIs such as `GraphSize` must not
be used because they describe Ollama/llama.cpp buffers, not ManboML's runtime.

## Backing and Ownership

`internal/backing` owns the bytes used both by the parser and by inference.

- Unix targets should use a whole-file read-only mmap when available.
- A mapping must remain valid until `Model.Close` has drained all requests.
- Windows and mapping failures use a bounded read-all fallback initially.
- The fallback may later become segmented if measurements show that the extra
  heap residency is unacceptable.
- The model file must not be modified or truncated while open.
- No borrowed public byte slice can outlive the model.

The root `Model` owns:

- Backing and file lifecycle.
- Normalized immutable Qwen3 weights.
- Immutable tokenizer and recognized formatter.
- Shared RoPE table.
- A bounded pool of private sessions.
- A model-owned numerical worker pool.

Each private Qwen3 session owns:

- F16 KV cache.
- Residual, projection, FFN, attention, and normalization buffers.
- Q8_K activation scratch.
- Attention scores.
- Full final logits.
- Prompt and generated token buffers.
- Request-local tokenizer workspace.

Multiple requests may share one model. A session is never used concurrently and
is reset before every independent request. `Model.Close` rejects new work,
waits for active requests and worker jobs, then releases sessions and backing.
`Close` is idempotent. Finalizers are not part of correctness.

## Public API Shape

Exact field spelling may be refined during implementation, but the capability
contract is fixed:

```go
model, err := manboml.Open(modelPath, manboml.Options{
    ContextSize:    2048,
    MaxConcurrent:  1,
    Workers:        0,
    MemoryLimit:    0,
    ExpectedSHA256: expectedDigest,
})
if err != nil {
    // handle error
}
defer model.Close()

result, err := model.Generate(ctx, manboml.GenerateRequest{
    Prompt:    prompt,
    MaxTokens: 128,
})

chatResult, err := model.Chat(ctx, manboml.ChatRequest{
    Messages: []manboml.Message{
        {Role: manboml.RoleSystem, Content: policy},
        {Role: manboml.RoleUser, Content: input},
    },
    MaxTokens: 128,
})
```

Required public concepts:

```go
type Model struct { /* unexported fields */ }

type Options struct {
    ContextSize    int
    MaxConcurrent  int
    Workers        int
    MemoryLimit    uint64
    ExpectedSHA256 string
}

type GenerateRequest struct {
    Prompt    string
    MaxTokens int
}

type ChatRequest struct {
    Messages  []Message
    MaxTokens int
}

type Result struct {
    Text            string
    GeneratedTokens int
    FinishReason    FinishReason
}
```

Public semantics:

- `ExpectedSHA256` empty means structural compatibility mode. A non-empty value
  requests exact digest verification before model publication.
- `MemoryLimit == 0` means no caller-selected total limit. Overflow, address
  space, allocation, context, and internal safety limits still apply.
- `Workers == 0` captures `runtime.GOMAXPROCS(0)` when the model opens. ManboML
  never changes process-wide `GOMAXPROCS`.
- `ContextSize == 0` uses 2,048.
- `MaxConcurrent == 0` uses one.
- `MaxTokens` must be positive. There is no unbounded generation default.
- `Generate` tokenizes an already formatted raw prompt.
- `Chat` accepts semantic messages and only executes a compiled, recognized
  formatter. An unknown embedded template returns `ErrUnsupportedTemplate`.
- v0.1 returns one complete `Result`; it has no channel or token callback.
- Finish reasons distinguish EOG and maximum-token termination. Cancellation
  and computation failures are errors, not successful finish reasons.
- Library code does not download, print, log implicitly, exit, inspect
  environment variables, or read Manboster configuration.

An `Estimate(path, Options)` capability and memory information in `ModelInfo`
are part of the planned public surface because constrained callers need
admission information before allocating sessions.

## Internal Tensor and Operator Model

There is no general runtime tensor abstraction. Qwen3 binding converts Ollama
tensor descriptors into concrete packed matrices:

```go
type Matrix struct {
    Data []byte
    Rows int
    Cols int
    Kind MatrixKind
}
```

The first matrix kinds are F32, Q4_K, and Q6_K. Packed matrices point into the
model backing. All 113 F32 norm tensors are decoded once at open into owned
`[]float32`; they contain 65,536 values and require exactly 256 KiB.

All numerical operators live in one internal package:

```text
internal/ops/
  matrix.go
  f16.go
  q4k.go
  q6k.go
  q8k.go
  matvec.go
  embedding.go
  vector.go
  norm.go
  rope.go
  softmax.go
  attention.go
  activation.go
  argmax.go
  executor.go
```

Operator responsibilities:

- Explicit F16 and little-endian F32 decoding.
- Q4_K and Q6_K block decoding and matrix-vector products.
- Q8_K activation quantization and packed dot products.
- Token embedding row extraction.
- RMSNorm and per-head Q/K RMSNorm.
- NeoX RoPE.
- Stable softmax.
- GQA attention over F16 KV storage.
- Residual vector operations.
- SiLU and SwiGLU.
- Deterministic greedy argmax; equal logits select the lowest token ID.

Keep two quantized matrix-vector paths:

1. A scalar F32-activation reference path for block correctness and debugging.
2. The production Q8_K-activation path, matching llama.cpp's intended K-quant
   execution shape more closely.

Full model matrices are never expanded to F32. Operators do not parse GGUF,
know layer names, own KV state, or depend on Ollama types.

## Qwen3 Execution

The Qwen3 binder is concrete rather than a generic graph. It validates and
stores explicit fields for embedding, final norm, output, and all per-layer
attention and FFN weights.

For each token:

```text
embedding -> hidden

for each layer:
  RMSNorm(hidden)
  quantize normalized input to Q8_K
  Q/K/V projections
  per-head Q/K RMSNorm
  NeoX RoPE
  write F16 K/V at the uncommitted position
  grouped-query attention
  attention output projection
  residual add
  RMSNorm
  gate and up projections
  SwiGLU
  down projection
  residual add

if logits are needed:
  final RMSNorm
  output projection
```

Qwen3's head width is explicit. The implementation must not assume
`hidden/queryHeads`; this profile has hidden width 1,024 but query width 2,048.
Query head `h` maps to KV head `h / 2` for the selected 16:8 GQA ratio.

Prompt prefill is token-by-token initially. Final normalization and vocabulary
projection are skipped for every prompt token except the last. Generation then
selects one greedy token, checks the model's complete EOG set, decodes its byte
piece, evaluates it when another token is needed, and stops at EOG or the
caller-provided bound.

A token evaluation commits the session position only after all layers succeed.
Cancellation may leave an uncommitted KV row partially overwritten; resetting
the logical position before pool reuse makes it unreachable.

## Session and KV Memory

Use per-layer, head-major F16 cache storage:

```text
K[layer][kvHead][position][headDim]
V[layer][kvHead][position][headDim]
```

Each K and V layer is an owned `[]uint16` containing binary16 bit patterns.
Head-major storage makes one head's historical rows sequential during
attention. Allocate K and V separately for each layer rather than one enormous
slice; this keeps individual allocations practical on 32-bit targets.

The exact KV formula is:

```text
KV bytes = layers * context * kvHeads *
           (keyHeadDim + valueHeadDim) * F16Bytes

         = 28 * context * 8 * (128 + 128) * 2
         = 114,688 * context
```

| Context | F16 KV per session |
| ---: | ---: |
| 1,024 | 112 MiB |
| 2,048 | 224 MiB |
| 4,096 | 448 MiB |
| 8,192 | 896 MiB |

Reset changes logical lengths and counters; it does not clear the entire KV
allocation. Attention may read only positions below the committed length.

Use explicit named scratch buffers in v0.1 rather than aggressive aliasing:

- Hidden and normalized hidden vectors.
- Q, K, and V projections.
- Concatenated attention result and output projection.
- FFN gate and up vectors.
- Full 151,936-value F32 logits.
- One context-length attention score vector.
- One reusable Q8_K activation buffer.

These buffers total less than 1 MiB for the selected profile, so clarity is
more valuable than saving tens of KiB beside a 224 MiB default KV cache.

All configured sessions and their KV caches are allocated during `Open`, after
the complete memory plan is accepted. This prevents a request from failing
mid-generation because a promised session could not be allocated.

## Memory Estimation

Report memory by category rather than claiming an exact RSS:

```text
peak estimate =
    shared model memory
  + MaxConcurrent * per-session memory
  + open-time transient memory
  + documented runtime headroom
```

Shared model memory includes:

- Full GGUF backing: 484,219,904 bytes for the certified file.
- Retained tokenizer strings and indexes.
- Normalized tensor descriptors and decoded F32 norms.
- Shared RoPE table.

Per-session memory includes:

- F16 KV cache.
- Numerical scratch and full logits.
- Token IDs and generated token storage.
- Bounded tokenizer and chat-formatting workspace.

A shared precomputed Qwen3 RoPE table requires:

```text
context * 64 frequency pairs * cos/sin * 4 bytes
= context * 512 bytes
```

| Context | RoPE table |
| ---: | ---: |
| 1,024 | 0.5 MiB |
| 2,048 | 1 MiB |
| 4,096 | 2 MiB |
| 8,192 | 4 MiB |

The deterministic lower bounds for one session, before tokenizer indexes and
Go runtime overhead, are:

| Context | Model backing | KV | RoPE and scratch | Lower bound |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 461.79 MiB | 112 MiB | about 1.1 MiB | about 574.9 MiB |
| 2,048 | 461.79 MiB | 224 MiB | about 1.6 MiB | about 687.4 MiB |
| 4,096 | 461.79 MiB | 448 MiB | about 2.7 MiB | about 912.4 MiB |
| 8,192 | 461.79 MiB | 896 MiB | about 4.7 MiB | about 1,362.5 MiB |

Initial conservative device-capacity guidance is 768 MiB, 1 GiB, 1.25 GiB,
and 1.75-2 GiB for those contexts respectively. Benchmark and allocation data
must replace heuristic tokenizer and runtime headroom estimates before v0.1.

With mmap, report the entire file separately as mapped bytes and include it in
the conservative capacity total even though not every page is resident at
`Open`. A read-all fallback reports it as managed heap. `MemoryLimit` applies to
the conservative ManboML plan, not an exact OS RSS guarantee.

`MemoryEstimate` should include at least model bytes, mapped bytes, shared heap,
per-session bytes, session count, open-time peak, and conservative total.

Ollama parser arrays and tokenizer indexes coexist during part of `Open`.
Tokenizer construction should reuse vocabulary strings, convert merge strings
to compact token-pair ranks, and release unneeded merge metadata before session
allocation where possible.

All products and sums use checked `uint64`. Convert individual lengths to Go
`int` only after platform checks. Contexts that cannot fit the 32-bit address
space must fail admission before `make` or mmap.

## Parallelism and Cancellation

Use one bounded worker executor per model, not package-global workers and not a
pool per session.

- `Workers == 1` runs inline without worker goroutines.
- `Workers == 0` captures `GOMAXPROCS` at open.
- Never change process-wide `GOMAXPROCS`.
- Parallelize quantized matrix-vector operations over contiguous output rows.
- One complete dot product belongs to one worker; do not combine floating-point
  partial sums from multiple workers.
- Q8_K activation buffers are immutable while row workers consume them.
- Q/K/V projections run sequentially, each with row-level parallelism.
- Q8_K quantization, norm, RoPE, elementwise operations, argmax, and attention
  remain serial initially.
- Attention-head parallelism is a measured later optimization.
- Workers cannot submit nested worker jobs.

This design keeps arithmetic order independent of worker count for each output
row. Same-toolchain worker-count tests should compare exact float bits. Across
architectures, generated token IDs must match and full logits use documented
tolerances because transcendental implementations may differ in low bits.

Cancellation checks occur:

- While waiting for a session.
- During bounded prompt formatting and tokenizer work.
- Before each input or generated token.
- Between Transformer layers and major projections.
- At bounded row/block intervals in long matvec and attention loops.

All published worker tasks must finish before a canceled session is reset or
returned to the pool. Preserve `context.Canceled` and
`context.DeadlineExceeded` for `errors.Is`.

## Error Policy

Planned stable categories include:

- Invalid request.
- Invalid or malformed model.
- Unsupported GGUF/model architecture.
- Unsupported tensor encoding or shape.
- Unsupported tokenizer or chat template.
- Context limit exceeded.
- Memory limit or platform address limit exceeded.
- Model closed.
- Numerical failure.

Errors support `errors.Is`. Detailed load errors may support `errors.As` and
identify the metadata key or tensor name, but must not include arbitrary prompt
contents, generated text, or unescaped binary metadata.

Malformed model input must return an error, not reach an operator panic. Public
methods reject work after close. EOG and maximum-token termination are normal
results, not errors.

## Portability Contract

The certified Linux matrix follows Go 1.26:

```text
linux/386
linux/amd64
linux/arm
linux/arm64
linux/loong64
linux/mips
linux/mipsle
linux/mips64
linux/mips64le
linux/ppc64
linux/ppc64le
linux/riscv64
linux/s390x
```

Manboster release targets also require Darwin amd64/arm64 and Windows
amd64/arm64.

Support means the `CGO_ENABLED=0` artifact can run inference on the target
without an external runtime. Cross-compilation alone is necessary but not
sufficient.

Verification tiers:

1. Cross-build every package for every target.
2. Run operator, endian, F16, Q4_K, Q6_K, and tokenizer fixtures on every Linux
   architecture through native CI or emulation.
3. Run a project-owned miniature Qwen3 end to end on every claimed target.
4. Run the full pinned Qwen3Guard model on representative hardware and maintain
   a documented schedule for uncommon architecture hardware or emulators.

Big-endian mips, mips64, ppc64, and s390x tests are required. Operators must use
explicit little-endian reads for GGUF F16/F32 fields and packed multibyte scales.

Every production dependency update must re-run the full cross-build matrix.
Architecture-specific assembly or unsafe fast paths are allowed only when
uncommon targets retain a direct-running implementation.

## Verification Strategy

Normal tests must not download models or require network access.

1. Operator tests compare F16, Q4_K, Q6_K, Q8_K, norm, RoPE, softmax, SwiGLU,
   attention, and argmax against pinned scalar references.
2. Worker counts from one through the configured maximum produce identical
   stored output values on the same architecture and toolchain.
3. Tokenizer tests compare exact Qwen2 token IDs and decoded bytes against
   pinned llama.cpp and Transformers behavior over multilingual, whitespace,
   special-token, and invalid-byte corpora.
4. Loader tests use tiny project-owned GGUF fixtures and compare normalized
   descriptors with pinned Ollama and llama.cpp behavior.
5. A deterministic miniature Qwen3 validates intermediate tensors, logits,
   cache reset, incremental decode, EOG, context exhaustion, and generation.
6. Optional real-model tests use a caller-supplied file, require the pinned
   SHA-256, and compare full logits and greedy token output with pinned
   llama.cpp.
7. Cancellation tests cover waiting, tokenization, prefill, matvec, and decode.
8. Race tests cover concurrent Generate/Chat, session reuse, and Close.
9. Fuzz tests cover public request validation, recognized formatters, tokenizer
   input, and the loader normalization boundary.
10. Benchmarks separate open time, prompt tokens per second, decode tokens per
    second, per-operator throughput, memory payloads, allocations, and worker
    scaling.

Generated prose is not a numerical correctness oracle. Tokenizer IDs and greedy
token IDs must be exact; logits and intermediates use documented tolerances.

## Milestones

| Milestone | Deliverable | Status |
| --- | --- | --- |
| M0 | Product contract, parser decision, runtime architecture, memory model, API, and roadmap | Documented; implementation not authorized |
| M1 | Go module, root API types, Ollama loader adapter, backing, fixtures, and cross-build baseline | Partially complete: module, backing, and loader done with real-model header verification; root API types pending |
| M2 | F16, Q4_K, Q6_K, Q8_K, matrix-vector, Transformer operators, and golden tests | Complete with internal-reference golden tests; ggml-generated fixture cross-checks remain future work |
| M3 | Qwen2 tokenizer, raw generation formatting, recognized Qwen3Guard chat formatter | Not started |
| M4 | Qwen3 binder, F16 KV sessions, tiny-model logits, greedy generation, cancellation | Not started |
| M5 | Real Qwen3Guard Q4_K_M parity, memory estimates, public Generate/Chat examples | Not started |
| M6 | Worker pool, profiling-driven portable optimization, full target runtime matrix | Not started |
| M7 | v0.1 documentation, compatibility contract, benchmarks, fuzz/race/release checks | Not started |

Each milestone updates this table with verification commands and known
limitations. A milestone is complete only when code, tests, documentation, and
required target checks are complete.

## Definition of v0.1

v0.1 is complete when:

- The pinned Qwen3Guard-Gen-0.6B Q4_K_M file produces accepted tokenizer IDs,
  logits, greedy token IDs, and decoded text through the generic API.
- Production inference does not load or execute llama.cpp, yzma, FFI, CGO,
  shared libraries, native helpers, or external processes.
- Q4_K, Q6_K, and F32 weights execute from packed backing without full-model F32
  expansion.
- Raw `Generate` and recognized-template `Chat` work with bounded output.
- Structural mode and optional SHA-256 mode behave as documented.
- Memory estimation categorizes mapped, shared, per-session, transient, and
  conservative totals and rejects infeasible allocations before session setup.
- Multiple private sessions safely share immutable weights.
- Every certified target cross-builds with `CGO_ENABLED=0` and passes the
  required runtime parity tier.
- Public APIs, lifecycle, errors, termination, compatibility, memory, examples,
  and benchmarks are documented.
- Normal tests require no network or production model download.

## Accepted Decisions

| ID | Decision | Accepted direction |
| --- | --- | --- |
| D1 | Native dependency boundary | `CGO_ENABLED=0`; no FFI, shared/native inference runtime, or helper process |
| D2 | Third-party Go code | Allowed; unsafe/assembly permitted when every target has a working path |
| D3 | First model | Qwen3Guard-Gen-0.6B |
| D4 | First encoding | Direct Q4_K_M execution |
| D5 | Product | Go library only |
| D6 | Public inference level | Generic `Generate` and `Chat`; raw token `Eval` remains internal |
| D7 | Parser | Ollama `fs/ggml`, pinned to a reviewed stable tag and isolated by `internal/loader` |
| D8 | Model compatibility | Structural validation by default; optional exact `ExpectedSHA256` |
| D9 | Output | Complete `Result`; no v0.1 stream callback or channel |
| D10 | Sampling | Greedy only; caller must provide a positive output-token bound |
| D11 | Minimum Go | Go 1.26 |
| D12 | Context and concurrency | Default context 2,048; one eager session |
| D13 | Memory limit | Zero means no caller cap; always run platform and overflow admission checks |
| D14 | Weight storage | Path-based mmap where available, bounded read fallback elsewhere |
| D15 | Operators | One private `internal/ops` package; F32 reference and Q8_K production paths |
| D16 | Scratch | Explicit named buffers first; optimize aliasing only after profiling |
| D17 | Runtime targets | Every Go 1.26 Linux GOARCH plus Manboster Darwin/Windows targets |
| D18 | Performance target | Establish a baseline before setting device-specific release thresholds |
| D19 | Hachimi integration | Deferred; ManboML first exposes generic inference only |

No accepted architecture decision authorizes implementation. The user must
explicitly start implementation in a later request.

## Known Risks

- Ollama `fs/ggml` is a package inside a large, fast-moving module and has no v1
  API promise. The internal adapter and pinned tag reduce but do not remove
  upgrade risk.
- Parser array materialization and tokenizer indexes can add tens of MiB and
  raise open-time peak memory.
- Qwen2 pre-tokenization is difficult to reproduce exactly with Go's standard
  regexp engine; a custom Unicode scanner or reviewed tokenizer code is needed.
- Pure scalar Go K-quant kernels may be too slow on low-frequency devices.
- The 2,048-token default requires about 224 MiB of KV plus a 461.79 MiB model;
  actual capacity should be at least about 1 GiB until measured more precisely.
- Whole-file mappings consume a large contiguous virtual range on 32-bit
  systems. Higher context presets may be infeasible even if individual slices
  fit `int`.
- Windows read-all fallback places the full 484 MiB model in the Go heap and may
  create significant GC and resident-memory pressure.
- Big-endian targets expose any accidental native-endian weight decoding.
- Mapped files can fault or corrupt inference if truncated or modified while
  open.
- Full-model emulation on all uncommon architectures may be prohibitively slow;
  tiny-model runtime parity is mandatory, with a separate documented full-model
  validation schedule.
- Quantized filename labels do not determine every tensor encoding.
- Numerical parity does not itself prove that Q4_K_M preserves acceptable guard
  classification quality.
- Model weights, vocabulary, and templates retain upstream licensing after
  conversion and must not be committed casually.

## Planning References

- GGUF specification:
  <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- GGML quant block definitions:
  <https://github.com/ggml-org/ggml/blob/master/src/ggml-common.h>
- GGML reference quantization kernels:
  <https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c>
- llama.cpp Qwen3 implementation and tokenizer behavior:
  <https://github.com/ggml-org/llama.cpp>
- Ollama GGML/GGUF parser:
  <https://github.com/ollama/ollama/tree/main/fs/ggml>
- Qwen3Guard model and template:
  <https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B>
- QuantFactory GGUF artifact:
  <https://huggingface.co/QuantFactory/Qwen3Guard-Gen-0.6B-GGUF>
- Current Manboster Hachimi implementation:
  <https://github.com/manboster/manboster/tree/dev/internal/hachimi/gguf>
- GPUStack parser reviewed but not selected:
  <https://github.com/gpustack/gguf-parser-go>

Pin exact upstream revisions when implementation begins. Moving branch links
above are discovery references, not reproducible implementation inputs.

## Decision Log

- 2026-07-15: Planning began from a README-only repository. The user requested
  architecture discussion before implementation and approved `AGENTS.md` as the
  persistent record.
- 2026-07-15: Qwen3Guard-Gen-0.6B and direct Q4_K_M were selected because the
  first validation path matches Manboster Hachimi.
- 2026-07-15: Inspection of the current artifact found GGUF v3 with 311 tensors:
  169 Q4_K, 29 Q6_K, and 113 F32.
- 2026-07-15: Generic `Generate`/`Chat`, a library-only product, a 2,048-token
  single-session default, and full uncommon-architecture runtime support were
  selected. Direct Hachimi integration was deferred.
- 2026-07-15: The initial design proposed a project-owned GGUF parser and public
  tensor package. Both were later removed from the selected v0.1 architecture.
- 2026-07-17: The portability contract was clarified: the hard requirement is
  `CGO_ENABLED=0` direct execution without FFI or a native runtime. `unsafe` and
  Go assembly do not automatically violate the contract when all targets retain
  a working path.
- 2026-07-17: GPUStack, Ollama, aikit, and other Go GGUF readers were compared.
  Ollama `fs/ggml` was selected because it parses an existing `io.ReadSeeker`
  and exposes tensor data offsets suitable for one long-lived mmap.
- 2026-07-17: All numerical primitives were consolidated into private
  `internal/ops`. The public surface remains model input/output only.
- 2026-07-17: The memory model was fixed around full-file packed backing, eager
  per-session F16 KV, explicit scratch, shared RoPE, categorized estimates, and
  optional caller limits.
- 2026-07-17: Structural model compatibility plus optional SHA-256, both raw and
  chat input, complete non-streaming results, zero-as-unlimited memory policy,
  and stable-tag Ollama pinning were accepted.
- 2026-07-17: The user requested that only this planning document be updated.
  Implementation remains intentionally unstarted.
- 2026-07-19: Implementation began. `internal/backing` (read-only mmap plus
  read fallback) and `internal/loader` (Ollama `fs/ggml` v0.32.1 adapter with
  independent structural validation) were completed and verified against the
  real Qwen3Guard Q4_K_M header: 311 tensors, 169 Q4_K, 29 Q6_K, 113 F32,
  151,936 vocabulary entries.
- 2026-07-19: `internal/ops` was completed: exact F16 conversion, Q4_K/Q6_K
  dequantization, Q4_K/Q6_K × Q8_K integer dot products matching ggml's
  accumulation order, Q8_K activation quantization, F32 and Q8_K matvec,
  embedding, RMSNorm, per-head RMSNorm, NeoX RoPE table, stable softmax,
  SwiGLU, GQA-ready F16 attention, and deterministic argmax. A bounded
  row-partitioned executor produces bit-identical output for worker counts
  1..64. All 13 Linux GOARCH targets cross-build with `CGO_ENABLED=0`.
