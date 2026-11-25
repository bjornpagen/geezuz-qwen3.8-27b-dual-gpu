# PLAN: geezuz rewritten as one engine for one model on one box

status: proposed, awaiting approval. no engine code has changed yet.

this plan has two parts:
- **part 1, must do**: the mandated cleanup, plus the performance work that is certain from first principles. it
  goes in regardless.
- **part 2, experiments**: everything with an uncertain payoff. each has a hypothesis, why it's uncertain, how to
  measure it, and when to kill it.

speed numbers are back-of-envelope estimates from upstream geezuz's published measurements and 3090 specs, not
measurements on this box. the code-only phase (section 1.10) builds and type-checks; real numbers start when the
hardware is up.

---

## 0. shared ground

### 0.1 decisions (locked)

| decision | value |
|---|---|
| model | qwen3.8-27b, text only (vision tower dropped), mtp head kept |
| weights | `Qwen/Qwen3.8-27B-FP8` (e4m3, 128x128 block scales) run as w8a16. the only weight format |
| gpus | exactly two rtx 3090 (sm_86, 82 SMs, 24 gb each), water cooled, nvlink bridged, tensor parallel 2 |
| host | x86_64 linux on the epyc, bare metal |
| numerics | bf16 activations, fp32 accumulation, fp32 residual stream, fp32 deltanet state, bf16 kv cache |
| parallel requests | required. fixed slot count compiled in, `S = 8` |
| memory overclock | yes, gated by the determinism harness |
| license | agpl-3.0 stays (inherited from open-jet via geezuz). refactor in place; credits kept |
| toolchain | latest stable clang/llvm and libc++, system cuda toolkit, no exact pins, dynamic linking |
| language | cpp-starter's c++26 dialect without reflection, compiled by clang; idris-mlir's engineering principles |
| async | `std::execution` (stdexec until libc++ ships it) for all orchestration around the kernels |
| build | cmake generating ninja, presets only |
| phase now | code only: build and type-check, no gpu runs until the box is up |

nothing is configurable that the hardware or the model already fixes: no runtime flags, environment switches,
compatibility options or fallbacks that choose between implementations. the build presets and the baked weight
image are the only configuration.

### 0.2 the lossless contract

full bf16 is 54 gb and does not fit in 48 gb, so the reference function is qwen's own fp8 checkpoint: 24.70b params
in e4m3 and 3.08b in bf16 (lm_head, embeddings, norms, small linear_attn params, mtp fc). the engine computes exactly
that function:

- e4m3 to bf16 conversion is exact; the block scale is applied in fp32.
- weights only are quantized. activations stay bf16, even though the checkpoint says `activation_scheme: dynamic`.
- every matmul accumulates in fp32; the residual stream and deltanet state are fp32; the kv cache is bf16.
- lm_head runs in bf16 with exact top-k and argmax merged across the two cards.
- speculative decoding verifies exactly (greedy is exact; sampling uses rejection sampling).
- sampling parameters are exact for every value the server accepts; anything else is refused with a named error.
- the only allowed difference is floating-point summation order.

outside the boundary, because they only move acceptance rates: the drafter, the draft vocabulary, suffix drafting,
tree shape, slot scheduling and the row budget.

**excluded by the contract, don't re-propose:**
- any 4-bit or sub-8-bit main-model weights;
- int8 or fp8 activations;
- fp8, int8 or 4-bit kv;
- bf16 deltanet state;
- fp16-accumulate matmuls;
- int8 or bf16 compression of the cross-card partial sums.

### 0.3 the budget (per card, one decode cycle)

| item | bytes | at 936 gb/s | at ~88% |
|---|---|---|---|
| body (fp8, 64 layers, half of each matrix) | 12.19 gb | 13.0 ms | 14.8 ms |
| lm_head (bf16, half of 248,320 vocab) | 1.27 gb | 1.36 ms | 1.5 ms |
| **main pass** | **13.46 gb** | **14.4 ms** | **16.3 ms** |
| mtp head (~5 passes: catch-up + 4 drafts) | ~1.2 gb | 1.3 ms | — |
| draft lm_head (32k vocab, 4 steps) | ~0.65 gb | 0.7 ms | — |
| kv reads (bf16, 2 kv heads per card x 16 layers) | 32 kb x context tokens, per slot | 1.0 ms per 30k tokens | — |

$$t_{\text{cycle}} \approx \max\left(t_{\text{bytes}},\ t_{\text{flops}}\right) + t_{\text{serial}}, \qquad \text{tok/s per slot} = \frac{\tau}{t_{\text{cycle}}}$$

**the compute ceiling.** on the 3090, bf16 tensor-core math with fp32 accumulation runs at ~71 tflops dense per card,
half the fp16-accumulate rate. one verify row costs ~27 gflop per card, ~0.5 ms realistically:

$$t_{\text{flops}}(R) \approx 0.5\ \text{ms} \times R$$

compute crosses the 16.3 ms weight stream at about 32 rows. the engine therefore spends a **row budget of ~24 rows per
cycle**, shared by all slots and their drafts: a compile-time constant, retuned on hardware.

**vram per card.**
- ~13.5 gb weights plus ~1 gb workspace leaves ~9 gb: about 280k tokens of bf16 kv, shared by all slots.
- deltanet state is ~75 mb per slot (fp32), ~600 mb at 8 slots, plus snapshots.
- the embedding table (2.54 gb bf16) lives in pinned host memory.

---

# part 1: must do

## 1.1 the cull

hard cutover: each item is replaced outright, with no shim, alias or "remove later". dead code, tests, docs and
comments go in the same change. the hardware baseline uses upstream's `v0.3.2` release build, so nothing old stays
in-tree for comparison.

| geezuz has | becomes |
|---|---|
| `MK_TP` 1 and 2, the one-gpu build and library, sequential one-gpu slots | tensor parallel 2 only |
| q4_k / q5_k / q6_k / q8_0 per-tensor types, `gguf.h`, the k-quant re-layouts | one e4m3 + block-scale path from the baked image |
| dp4a gemv for ≤10 rows, int8 mma above, per-type gemv instances (`mma.cuh`, `q8_tile.h`) | one bf16 tensor-core gemv for 1 to R rows |
| int8 activation quantization and folding the next norm into the quantizer | bf16 activations |
| int8 / bf16 cross-card exchange (`MK_OPT`) and the pcie p2p flag protocol | fp32 partials over nvlink |
| sm_86 / sm_89 / sm_120 builds and `ARCH=` | sm_86 only |
| `--kv f16/q8/auto`, the q8 kv path, 64/128-token blocks, `--kv-mem` | bf16 kv, one block size, pool geometry fixed at build time |
| `MK_OPT`, `MK_BATCH_ROWS`, `MK_PF_CAP`, `MK_EMBD_DEVICE`, `MK_PREFILL=mk`, `MK_DRAFT_SAMPLING`, `MK_DEVICES` | fixed choices compiled in |
| fp16-accumulate prefill gemm (`f16.f16.f16.f16` mma) | bf16 x bf16 to fp32 |
| `-d 0..4`, the rejected-draft depth drop, 16-vs-32 row caps by slot count | one drafting policy under the row budget |
| `--slots 1..8` | `S = 8` compiled in |
| libllama for tokenization, the jinja chat template from the gguf | our qwen bpe tokenizer and a hardcoded qwen3.8 chat format |
| `mk_server.py`, `capi.cpp`, python benches and tests, docker, box scripts, Makefile, `make release` | one c++ server, tools and tests on cmake + ninja |
| `tools/llama_ref`, `logit_stats.h`, `--kld` against llama.cpp | the cpu fp32 reference over the same fp8 checkpoint |
| `MK_PROFILE`, `MK_HOST_PROF`, `MK_PF_PROFILE` | a `profile` build preset |
| `--raw`, `--ids`, `--prompt-ids`, `--turn2`, `--score` | a small cli over the same pipeline, plus the harness tools |

kept as ideas and rewritten in the new style:
- the persistent megakernel and lazy-replay deltanet rollback;
- the paged kv pool with radix prefix cache, deltanet snapshots, copy-on-write and preemption;
- shared-snapshot prefill deduplication;
- the 32k draft vocabulary;
- double-buffered split-k attention;
- prefill that leaves exactly the state decode expects;
- the server's scheduling policy: fair chunked prefill, longest-cached-prefix slot assignment, turn replay.

## 1.2 toolchain and rules

### toolchain

- **latest stable clang/llvm** for host c++26 and cuda device code (clang's cuda mode, `--cuda-gpu-arch=sm_86`), with
  **libc++**. no gnu compiler or runtime. dynamic linking.
- **the system cuda toolkit**: the newest one latest stable clang supports. `PINS.md` records the pair in use.
- **no cudart.** clang compiles the kernels to cubin, the cubin is embedded in the binary, and the host loads it
  with the cuda driver api. the driver is the only nvidia runtime library.
- **cmake generating ninja**, presets only:

  | preset | purpose |
  |---|---|
  | `dev` | `-O1`, warnings as errors, trap-mode ubsan on host, device invariant checks |
  | `release` | `-O3`, lto, device checks off |
  | `asan-ubsan` | host-only runtime sanitizers for the cpu test suite |
  | `profile` | release plus the per-phase profiler |
  | `lint` | clang-tidy over the whole graph |

- the configure gate rejects anything but x86_64 linux with clang, libc++ and a cuda toolkit present.
- `-std=c++26 -fno-exceptions -fno-rtti`, warnings as errors, host assertions always on.
- `.clang-format` taken unchanged from cpp-starter.

### from cpp-starter (the 80/20)

copied and adapted:
- **AGENTS.md**: the "one concept, one primitive" table, minus the reflection rows and with clang authoritative.
  also its sections on modules, source zones, structural typing, data modeling, `struct` vs `class`, no oo or rtti,
  no exceptions and the error algebra, ownership and borrowing, no shared ownership, mutation, concurrency and the
  reactor law, casting, numeric conversions, c remnants, i/o, api shape, boolean parameters, resource classes,
  allocation, testing, suppressions, enforcement, and the review checklist.
- **build files**: the `CMakeLists.txt` skeleton with the language-profile target, `CMakePresets.json`,
  `.clang-format`, `.clang-tidy` (split into a dialect profile and a quarantine profile).
- **`src/handle.cc`**: generational `{index, generation}` handles for slots, kv blocks and connections.
- **`src/core.cc`**: the `std::expected` error shape ("errors are decision surfaces").
- **`foreign/exec.cc` + `exec.backend.cc`**: the stdexec boundary. its narrow-abi quarantine was a gcc ice workaround;
  under clang, retest importing stdexec inside the module graph and delete the workaround if it is clean.
- **`unsafe/net.cc`, `net.loop.cc`, `net.linux.cc`, `net.internal.h`**: the bounded single-threaded epoll reactor.
- **`src/http.cc`**: bounded http, extended with request bodies and sse streaming.
- **`tests/compile_fail/`** and **`PINS.md`** (the ledger of the toolchain pair and deliberate deviations).

dropped: reflection (`src/enums.cc` and the reflection tests), the gcc gate and gcc defect ledger, darwin/arm64 and
`net.darwin.cc`, the static-linking and hardening policy beyond `dev`.

### from idris-mlir (principles, written into AGENTS.md)

- **representation first**: change data, types and invariants until the special case can't be expressed, before
  adding a branch, flag or guard.
- **no static fact paid for at runtime**: every shape, offset, tile schedule, head split, buffer size and pool
  geometry is a compile-time constant checked with `static_assert`. the only runtime facts are the tokens, the
  lengths, which slots are live, and the machine's identity.
- **one fact, held once**: model dimensions live in one place (section 1.3).
- **hard cutover, hard culling. delete before you optimize. a blowup is a bug, fixed at its cause.**
- **a choice is a type or an enum case, never a `bool`.**
- **comments explain why.** new files carry `// SPDX-License-Identifier: AGPL-3.0-only` (upstream says "version 3",
  not "or later"; confirm when stamping).
- **report numbers honestly**: what was measured, on what, how many runs.
- **tests check behaviour**, never incidental output.
- **done means built and checked here**; type-checked is not built.

### device code rules (new)

- device code is a quarantine zone: it can't import modules, so it uses headers.
- allowed: templates, concepts, `constexpr`/`consteval`. forbidden: exceptions, rtti, virtual, dynamic allocation,
  `printf`.
- **no runtime shape arguments.** every dimension is a template parameter or a `constexpr` from the model
  description. kernel parameters carry only runtime facts.
- inline ptx only inside small named primitives (`cp.async`, acquire/release loads and stores, `mma.sync`, e4m3
  conversion), each documented with why.
- **every kernel's resource budget is a static fact**: threads, registers, shared memory (≤ 99 kb per block) and
  blocks per SM are `static_assert`ed. the build fails if `ptxas -v` reports spills or registers over budget.
- device invariant traps exist only in `dev`.
- one narrow typed launch function per kernel, exported upward as a module partition from `foreign/`.

## 1.3 layout and the model description

```text
AGENTS.md          normative rules (section 1.2)
CLAUDE.md          one line: read AGENTS.md
PLAN.md            this file
PINS.md            toolchain pair in use and deliberate deviations
CMakeLists.txt     gate, language profile, targets
CMakePresets.json  dev, release, asan-ubsan, profile, lint
src/               the `engine` named module and its partitions (dialect)
device/            cuda kernels and the model description header (quarantine)
foreign/           cuda driver api, stdexec boundary, model-description re-export (quarantine)
unsafe/            epoll reactor, mmap, pinned and hugepage memory (quarantine)
apps/              geezuz-server, geezuz (cli), geezuz-bake, geezuz-ref (dialect)
tests/             unit, property and compile-fail tests (dialect)
bench/             benchmark clients (dialect)
data/              draft vocabulary, tokenizer and chat-format fixtures
```

| partition | holds |
|---|---|
| `engine:model` | re-export of the model description: dims, layer kinds, head splits, tensor list |
| `engine:image` | baked-image format: header with checkpoint hash, per-tensor extents; typed errors on mismatch |
| `engine:bake` | safetensors reader, tensor-parallel split, tile layout in stream order |
| `engine:tokenizer` | qwen byte-level bpe, special tokens, utf-8-safe streaming decode |
| `engine:chat` | qwen3.8 chat format: system, user, assistant, tool turns, thinking, reasoning effort, tool-call parse |
| `engine:sampling` | sampling parameters as a closed type; exact top-k / top-p / min-p / temperature; seeds |
| `engine:kv` | paged bf16 kv pool: refcounted blocks, copy-on-write, block tables |
| `engine:prefix` | radix tree over blocks, deltanet snapshots, lru eviction |
| `engine:sched` | slots, admission, preemption, chunked-prefill share, row budget, turn replay |
| `engine:device` | typed launch surface over the driver api: cards, streams, events, the resident ring |
| `engine:pipeline` | stdexec request pipeline |
| `engine:json`, `engine:http`, `engine:openai` | bounded json, http with bodies and sse, the openai endpoint |

**one model description, held once.** `device/model.h` is the single source of every model and layout constant: plain
`constexpr`, no macros, because device code must read it. `foreign/model.cc` re-exports it as `engine:model`. tile
schedules and buffer geometries are `consteval` functions over it. nothing else defines a dimension.

**the baked image.** `geezuz-bake` reads the safetensors checkpoint once and writes one image per card:
- verifies the checkpoint hash and stamps it in the header;
- splits for tensor parallel 2 and drops the vision tower;
- lays out e4m3 tiles and scales in exact stream order, 16-byte aligned;
- converts nothing lossy.

the engine mmaps the image and copies it to vram with no layout decisions. a mismatched image is refused at the
boundary.

## 1.4 server and parallel requests

- **`S = 8` slots**, each with its own deltanet state, mtp state and position; kv comes from the shared pool.
- **every cycle runs all live slots together**, so weights are read once per cycle.
- **row budget**: ~24 rows per cycle. a fixed, simple allocation ships first: one row per live slot for its
  committed token, and the rest split evenly as draft depth, capped at 4 per slot. an adaptive allocator is an
  experiment (2.3).
- **per-slot work**: attention, deltanet, draft check and sampling run per slot with its own row count.
- **preemption**: cached blocks are evicted first, then the newest admitted request pauses with its state cached and
  resumes at the head of the queue.
- **prefill alongside decode**:
  - a prompt goes in whole when nothing is decoding;
  - otherwise in chunks for a fixed share of time, alternating shortest-remaining and longest-waiting;
  - prompts that share an uncached snapshot point are prefilled once;
  - a prefix-cache hit prefills only the tail;
  - turn replay keeps the cache hitting across turns.
- **request pipeline (std::execution)**:

  ```text
  http request (reactor) -> parse + validate (typed errors, bounded)
    -> tokenize + render chat format
    -> admit: slot + longest cached prefix            (scheduler; may wait)
    -> prefill tail chunks                            (sender per chunk, when_all over both cards)
    -> decode stream                                  (tokens -> detokenize -> sse frames)
    -> finish: stop / length / tool_calls; snapshot into the prefix cache
  cancellation: client disconnect -> stop token -> slot released at the next cycle boundary
  ```

  - one typed scheduler per card; a sender completes on a driver-api event.
  - `when_all` over both cards for weight upload and prefill chunks.
  - nvidia's `nvexec` targets nvc++/nvcc, so on clang we write a small stream/event scheduler in `foreign/`.
  - **the decode cycle is not a graph of senders**: it is one persistent kernel; senders submit cycles and collect
    results. a sender per phase would bring back ~340 launches per cycle.
  - the http reactor stays cpp-starter's owner-driven epoll state machine.
- **api**:
  - `/v1/chat/completions`, streaming and not: `messages`, `tools`, `tool_choice`, `stream`, `max_tokens`,
    `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, and `chat_template_kwargs` (`enable_thinking`,
    `reasoning_effort`, `preserve_thinking`).
  - the response carries `reasoning_content`, `tool_calls`, and `usage.prompt_tokens_details.cached_tokens`.
  - `/v1/models`, `/health`, `/status`, plus `X-Request-Id` tagging.
  - unsupported parameters are refused with a named 400. assistant turns with `content: null` (tool-call-only or
    reasoning-only turns that harnesses resend) are accepted as empty; adrien hit http 400s on exactly this.

## 1.5 kv pool and prefix cache (port)

- paged bf16 kv with refcounted blocks, copy-on-write, a radix tree over blocks, and deltanet snapshots at the system
  prompt end, last user message start, generation prompt start and prompt end.
- snapshots are capped at a fixed share of the pool; eviction is lru.
- port geezuz's pool fuzz and scenario tests against a content-checking fake device.

## 1.6 device: decode and prefill

done in two hard-cutover steps inside one lane's history, never two kernels side by side.

**step 1: straight port, the correctness baseline.** geezuz's persistent-megakernel design re-expressed in the new
rules:
- e4m3 tiles converted to bf16 in registers;
- bf16 `mma.sync` with fp32 accumulation for every row count;
- fp32 partial sums over nvlink with the same deterministic summation order on both cards;
- bf16 kv;
- fp32 deltanet state with lazy replay;
- split-k flash decoding;
- mtp drafting over the 32k draft vocabulary;
- exact rejection-sampling verify.

grid barriers stay, so it is directly comparable to upstream.

**step 2: the dataflow core. certain from first principles, so it is part 1:**
- **producer/consumer warp specialization**: producers stream weights continuously through a `cp.async`
  shared-memory ring; consumers wait on per-layer and per-tile counters instead of ~340 grid barriers. weights
  have no data dependency, so dram never has to wait. ~8 mb of shared memory plus 6 mb of l2 gives ~15 µs of
  run-ahead.
- **all-reduce fused into the gemv epilogue**: fp32 partials go over nvlink per tile as they finish.
- **serial phases under the weight stream**: deltanet core, attention, lm_head merge, sampling and the mtp steps run
  while producers prefetch. the mtp steps overlap the next cycle's layer-0 prefetch.
- **device-resident loop**: the kernel stays resident and exchanges cycle descriptors and results through a mapped,
  pinned ring. no launch or readback per cycle.
- **compile-time schedule**: generated from the model description; resource budgets are `static_assert`ed.

why it's certain: upstream's measured cycle (~20.9 ms on q4_k_m) is about twice its weight-read floor (~9.3 ms),
so about half of every cycle is not moving weights. the exact split between barriers, exchange and serial phases gets
measured in phase b, but the gap itself is arithmetic.

**prefill (forced by the contract):**
- bf16 x bf16 to fp32 tensor-core gemm on e4m3 converted in registers: ~71 tflops per card, so ~2,000 tok/s total
  is the target;
- fused `silu(gate) * up`;
- chunked (wy) deltanet on tensor cores with fp32 state;
- causal flash attention over the cache;
- all-reduces over nvlink, overlapped with the gemm.

## 1.7 platform

- **nvlink**: `nvidia-smi topo -m` shows NV4; the stock driver does p2p over nvlink with no patch. check that the
  2-slot bridge spacing matches PCIE_6 / PCIE_4.
- **power**: limit at or above stock 350 w, persistence mode, locked clocks so boost doesn't oscillate between
  cycles.
- **memory overclock**: sweep for peak decode throughput (gddr6x error replay hides errors as slowdowns), then
  accept the clock only after a long golden-hash soak at operating temperature. adrien's 5090s got +15% raw bandwidth
  but only +4% at tp2, because allreduce latency co-bound his decode. our fused nvlink exchange should let more of
  it through.
- **memory thermals**: 12 of the 24 gddr6x chips sit under the xg7's passive backplate. watch the memory junction
  temperature during the soak; add a backplate fan if it climbs.
- **host**: iommu passthrough, hugepage-backed pinned memory for the embedding table and rings, the resident loop's
  host thread pinned to one core.

## 1.8 verification (built first; gates everything)

- **cpu reference (`geezuz-ref`)**: the model computed straight from the fp8 checkpoint in fp32 on the cpu, written
  independently of the kernels. teacher-forced logits for plain text and for your own agent transcripts.
- **decode-path ruler**: compares greedy continuations token by token against the reference. a prefill-only ruler
  can't see decode kernels, and adrien's gotcha #2 is a decode kernel that read bit-identical on the prefill ruler.
- **rulers**: top-1 agreement, kl and perplexity, with the noise floor measured.
- **tokenizer and chat-format fixtures** generated once from qwen's reference tokenizer and template and committed.
  check `tokenizer.json` for a baked truncation (adrien's gotcha #7).
- **determinism harness**: a deterministic mode plus golden logit hashes per prompt and per slot mix. every clock,
  kernel or schedule change reproduces them bit-for-bit, or stays within the noise floor and re-goldens with the
  reason recorded.
- **host property tests**: pool and prefix fuzzing; a scheduler simulation over a fake engine (preemption, budgets,
  no lost tokens, no hangs); http/json/sse bounds.
- **static device gates** (code-only phase): `ptxas -v` per kernel with zero spills and registers in budget;
  shared memory and occupancy asserted; schedule geometry `static_assert`ed against the model description.
- **measurement protocol** (adrien's gotcha #9: speculative decode varies ~10% between runs through content-dependent
  acceptance): compare speed at temperature 0, within one boot, normalized per accepted token, best of n.
- **profiler** in the `profile` preset: time per phase, warp arrival, peer wait, dram utilization.

## 1.9 must-do outcome (estimates)

| config | cycle | 1 stream code (τ ≈ 4.46) | 1 stream prose (τ ≈ 2.16) | 8 streams code, aggregate |
|---|---|---|---|---|
| upstream geezuz q8_0 (pcie, 280 w) | ~25 ms | ~143 | ~85 (est) | 610 |
| step 1: straight port on fp8 + nvlink | ~23 ms | ~190 | ~95 | ~700 |
| step 2: dataflow core + resident loop | ~18 ms | ~250 | ~120 | ~1,000 |
| + memory overclock (~8%) | ~16.8 ms | ~265 | ~128 | ~1,080 |

estimates. step 2 is the row physics most clearly supports. long multi-slot contexts add kv-read time per cycle.

## 1.10 order of work

### phase a: code only (now; no gpu)

**where it runs**: a linux x86_64 environment with latest clang/llvm and the cuda toolkit, such as the cloud claude
code containers. no gpu is needed: clang compiles the device code and ptxas reports resources. macos can't host the
cuda toolkit, so the mac is for editing only.

**done means**: `dev`, `release` and `lint` build with zero warnings, the static device gates pass, and the cpu tests
pass. nothing claims runtime kernel correctness until phase c.

**lanes for parallel subagents.** lanes implement and hand over a change that builds. the coordinator merges onto
`main`, runs every check on the merged tree, and sends exact fixes back. commits go straight to `main`.

| lane | work | depends on |
|---|---|---|
| L0 scaffold | AGENTS.md and CLAUDE.md, the cpp-starter 80/20, cmake gate, presets, language profile, clang-tidy profiles, zones, compile-fail harness, PINS.md | — (first) |
| L1 cull | delete everything in 1.1 with no successor yet; move what survives into the zones | L0 |
| L2 model + image + bake | `device/model.h`, `engine:model`, `engine:image`, `engine:bake` | L0 |
| L3 tokenizer + chat | `engine:tokenizer`, `engine:chat`, fixtures | L0 |
| L4 server | `engine:json`, `engine:http`, `engine:openai`, `engine:pipeline`, `engine:sched` + fake-engine simulation | L0, L5 |
| L5 kv + prefix | `engine:kv`, `engine:prefix`, ported fuzz and scenario tests | L0, L2 |
| L6 device | e4m3/bf16 primitives, fp8 gemv/gemm, attention, deltanet, decode step 1 then step 2, prefill, driver-api loader, resident ring, static gates | L0, L2 |
| L7 reference + harness | `geezuz-ref`, rulers, decode-path ruler, determinism tooling | L0, L2, L3 |

### phase b: hardware bring-up

nvlink, power and thermals (1.7). upstream `v0.3.2` q8_0 with `MK_PROFILE=1`, with and without the bridge, gives the
baseline and the real phase breakdown.

### phase c: correctness on gpu

baked images load; step 1 matches `geezuz-ref` within the noise floor on both rulers; golden hashes recorded; the
server runs end to end with 8 parallel streams, tools and thinking.

### phase d: must-do speed

step 2 under the harness, then the memory-overclock sweep and soak. then part 2.

---

# part 2: experiments

none of these goes in by default. each one ships only if it beats its kill criterion under the measurement protocol
(1.8), with output unchanged (golden hashes, or the noise floor for intentional reorderings). ordered by expected
value.

### 2.1 tree drafts

- **hypothesis**: verifying a tree of mtp top-k branches raises τ, mostly on prose, where today's chain accepts
  ~2.16 of 5.
- **why uncertain**: the bf16 compute ceiling caps rows at ~24 per cycle, so a tree competes with slots for rows.
  per-branch deltanet state is real work: only the accepted path is replayed, but each branch needs its own state
  update in the verify pass. the gain at one stream may not survive at eight.
- **measure**: τ and tok/s at 1 and 8 streams, code and prose, against the chain.
- **kill if**: under +10% tok/s at one stream, or any loss at eight.

### 2.2 suffix / prompt-lookup branches

- **hypothesis**: agent output copies context (file rewrites, diffs, tool-call json); a suffix match can propose
  16–64 tokens for free.
- **why uncertain**: adrien measured vllm's suffix decoding as a replacement for mtp and code decode fell from ~180 to
  31 tok/s, because fresh generation has nothing to match. it can only work as extra branches added when a match
  exists, never in place of mtp.
- **measure**: share of cycles with a match, τ gain on your agent transcripts, no regression on fresh prose.
- **kill if**: no gain on your own traffic.

### 2.3 adaptive row allocator

- **hypothesis**: spending the ~24 rows where expected accepted tokens per row is highest beats the fixed split, e.g.
  a deep tree for one busy slot, shallow drafts for eight.
- **why uncertain**: the gain depends on acceptance variance across slots and needs good live estimates.
- **kill if**: under +5% aggregate on mixed loads.

### 2.4 row budget and slot count tuning

- **hypothesis**: the real compute crossover sits somewhere between 20 and 32 rows; `S` other than 8 may be better.
- **measure**: cycle time against rows on the hardware, then tok/s per stream and aggregate.
- **ship**: the measured constants. this changes constants, not code paths.

### 2.5 replicated drafter

- **hypothesis**: holding the whole mtp head on both cards (+0.48 gb vram each) removes every cross-card exchange
  from the latency-bound draft steps.
- **why uncertain**: it doubles mtp bytes per card (~+1.3 ms per cycle) to save the exchange latency of ~10 serial
  steps. which wins depends on nvlink latency, measured in phase b.
- **kill if**: the cycle isn't faster.

### 2.6 reduced-precision drafter

- **hypothesis**: drafter bytes (~1.85 gb per card per cycle as shipped) drop by about half at 8-bit-int or lower
  for the fc and draft lm_head, saving ~1 ms per cycle. this sits outside the lossless boundary, so output is
  unchanged.
- **why uncertain**: adrien saw a quantized drafter beat the original on two checkpoints, and a mismatched one lose
  26% of code decode. acceptance can move either way.
- **kill if**: τ x (1 / cycle) doesn't improve.

### 2.7 personalized drafter

- **hypothesis**: an mtp head finetuned on your own chats and coding sessions (main model frozen) raises acceptance on
  your traffic.
- **why uncertain**: needs a training run and enough of your data; the gain is unknown.
- **kill if**: τ on held-out sessions isn't up.

### 2.8 cascade (shared-prefix) attention

- **hypothesis**: kv blocks shared by several slots through the prefix cache (system prompt, repo dump, common
  transcript) are read once per cycle for all of them, with exact softmax merging. this cuts kv bandwidth when parallel
  agents share long prefixes.
- **why uncertain**: it only pays when slots actually share long prefixes, and the merge adds attention complexity.
- **measure**: cycle time at 8 slots with a shared 30k prefix vs without.
- **kill if**: your real workloads rarely share prefixes.

### 2.9 prefix-cache tiers (host ram, then nvme, restart-proof)

- **hypothesis**: agent steps resend the whole transcript, so a spilled prefix restored from ram or disk beats
  re-prefill. adrien measured 0.45 s from disk against 7.5 s cold at 32k.
- **why uncertain**: it only matters if the ~280k-token pool evicts under your workload, and it is a lot of code.
- **measure first**: the pool eviction rate on real agent sessions.
- **gate** (adrien's gotcha #3): a needle planted deep in a long context must be retrieved correctly after a restart.
  the tier namespace keys on the image's checkpoint hash (gotcha #4).

### 2.10 certified-bound lm_head pruning

- **hypothesis**:
  - score all rows with a low-precision shadow copy, bound each row's error, drop rows that provably can't make the
    top-k, then recompute the survivors in bf16;
  - exact for greedy and top-k sampling;
  - saves up to ~0.6 gb per card per cycle.
- **why uncertain**: the bounds may be too loose to prune much, and requests without top-k need the full softmax.
- **kill if**: the survivor count is too high to save time.

### 2.11 block drafter (dflash-style)

- **hypothesis**: one drafter pass proposes a whole block. adrien's single-stream code went from ~209 (mtp) to
  ~279–325 tok/s on 5090s.
- **why uncertain**: it costs extra weight bytes per cycle, did not scale with batch for adrien (c4 aggregate ≈ c1),
  and capped context. parallel requests are required here, so it likely loses.
- **priority**: lowest. revisit only if single-stream speed is the main use.

---

## 3. risks

- **clang cuda and the latest toolkit**: clang can trail the newest toolkit; use the newest pair it supports.
- **stdexec on clang + libc++**: the module boundary may still need the narrow-abi quarantine.
- **kernels are unverified until phase c**: phase a proves builds, resource budgets and host logic, not kernel
  correctness. the cpu reference exists so phase c catches mistakes on day one.
- **the bf16 compute ceiling**: the lossless contract caps verify width at ~24 rows per cycle. retune on hardware.
- **memory overclock without ecc**: errors can be silent. accept a clock only after a long golden-hash soak at
  temperature.
- **checkpoint drift**: the image header carries the checkpoint hash; a re-uploaded checkpoint is refused, never
  half-loaded.
