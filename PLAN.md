# PLAN: geezuz rewritten as one engine for one model on one box

status: proposed, awaiting approval. no engine code has changed yet.

speed numbers are back-of-envelope estimates from upstream geezuz's published measurements and 3090 specs. none
were measured on this box. the code-only phase (section 9) builds and type-checks; real numbers start when the
hardware is up.

---

## 0. decisions (locked)

| decision | value |
|---|---|
| model | qwen3.8-27b, text only (vision tower dropped), mtp head kept |
| weights | `Qwen/Qwen3.8-27B-FP8` (e4m3, 128x128 block scales) run as w8a16. the only weight format |
| gpus | exactly two rtx 3090 (sm_86, 82 SMs, 24 gb each), water cooled, nvlink bridged, tensor parallel 2 |
| host | x86_64 linux on the epyc, bare metal |
| numerics | bf16 activations, fp32 accumulation, fp32 residual stream, fp32 deltanet state, bf16 kv cache |
| parallel requests | required. fixed slot count compiled in, `S = 8` to start (one constant, revisited with measurements) |
| memory overclock | yes, gated by the determinism harness (no ecc on gddr6x, so a bad clock is silent corruption) |
| license | agpl-3.0 stays (inherited from open-jet via geezuz). refactor in place; credits kept |
| toolchain | latest stable clang/llvm and libc++, system cuda toolkit, no version pinning beyond "latest stable", dynamic linking |
| language | cpp-starter's c++26 dialect without reflection, compiled by clang; idris-mlir's engineering principles |
| async | `std::execution` (stdexec until libc++ ships it) for all orchestration around the kernels |
| build | cmake generating ninja, presets only |
| phase now | code only: build and type-check, no gpu runs until the box is up |

nothing is configurable that the hardware or the model already fixes. no runtime flags, environment switches,
compatibility options or fallbacks that choose between implementations. one path per job. the build presets and the
baked weight image are the only configuration.

---

## 1. the lossless contract

full bf16 is 54 gb and does not fit in 48 gb, so the reference function is qwen's own fp8 checkpoint: 24.70b params
in e4m3 and 3.08b in bf16 (lm_head, embeddings, norms, small linear_attn params, mtp fc). the engine computes exactly
that function:

- e4m3 to bf16 conversion is exact (every e4m3 value is a bf16 value); the block scale is applied in fp32.
- weights only are quantized. activations stay bf16: no fp8 or int8 activation quantization, even though the
  checkpoint says `activation_scheme: dynamic`, because that is a hopper speed trick and moves further from bf16.
- every matmul accumulates in fp32. the residual stream and the deltanet state are fp32. the kv cache is bf16, which
  is exactly what the reference attention sees.
- lm_head runs in bf16 with exact top-k and argmax merged across the two cards.
- speculative decoding verifies exactly. greedy output equals the model's greedy choice; sampling uses rejection
  sampling, so the output distribution is unchanged.
- sampling parameters are implemented exactly for every value the server accepts. a value outside the implemented
  set is refused at the boundary with a named error, never approximated.
- the only allowed difference is floating-point summation order.

outside the boundary, free to change because they only move acceptance rates, never output: the mtp drafter, the draft
vocabulary, suffix drafting, tree shape, slot scheduling and the row budget.

---

## 2. the byte and flop budget (per card, one decode cycle)

| item | bytes | at 936 gb/s | at ~88% |
|---|---|---|---|
| body (fp8, 64 layers, half of each matrix) | 12.19 gb | 13.0 ms | 14.8 ms |
| lm_head (bf16, half of 248,320 vocab) | 1.27 gb | 1.36 ms | 1.5 ms |
| **main pass** | **13.46 gb** | **14.4 ms** | **16.3 ms** |
| mtp head (~5 passes: catch-up + 4 drafts), as shipped | ~1.2 gb | 1.3 ms | — |
| draft lm_head (32k vocab, 4 steps), as shipped | ~0.65 gb | 0.7 ms | — |
| kv reads (bf16, 2 kv heads per card x 16 layers) | 32 kb x context tokens, per slot | 1.0 ms per 30k tokens | — |

$$\text{tok/s per slot} = \frac{\tau}{t_{\text{cycle}}}, \qquad t_{\text{cycle}} \approx \max\left(t_{\text{bytes}},\ t_{\text{flops}}\right) + t_{\text{serial}}$$

**the compute ceiling matters for lossless.** on the 3090, bf16 tensor-core math with fp32 accumulation runs at half
the fp16-accumulate rate: ~71 tflops dense per card. one verify row costs ~27 gflop per card, ~0.38 ms at peak and
~0.5 ms realistically:

$$t_{\text{flops}}(R) \approx 0.5\ \text{ms} \times R$$

| rows per cycle R | compute | against the 16.3 ms weight stream |
|---|---|---|
| 16 | ~8 ms | hidden |
| 24 | ~12 ms | hidden, little headroom |
| 32 | ~16 ms | equal: compute-bound from here |
| 40 | ~20 ms | compute-bound |

so the whole engine spends a **row budget of ~24 rows per cycle**, shared by all slots and their drafts. it is a
compile-time constant, retuned on hardware. this replaces the earlier assumption that 32 rows are free: that holds
for int8 math, which the lossless contract excludes.

**vram per card.** ~13.5 gb weights plus ~1 gb workspace leaves ~9 gb: about 280k tokens of bf16 kv, shared by all
slots through the paged pool. deltanet state is ~75 mb per slot per card (fp32), ~600 mb at 8 slots, plus prefix
snapshots. the embedding table (2.54 gb bf16) stays in pinned host memory: one row per token.

---

## 3. the cull: what gets deleted from geezuz

hard cutover: each item is replaced outright, with no shim, alias or "remove later". the dead code, tests, docs and
comments go in the same change. the hardware baseline (phase b) uses upstream's `v0.3.2` release build, so nothing
old is kept in-tree for comparison.

| geezuz has | becomes |
|---|---|
| `MK_TP` 1 and 2, the one-gpu build and library, sequential one-gpu slots | tensor parallel 2 only |
| q4_k / q5_k / q6_k / q8_0 per-tensor types, `gguf.h`, the k-quant re-layouts | one e4m3 + block-scale path from the baked image |
| dp4a gemv for ≤10 rows, int8 mma above, per-type gemv instances (`mma.cuh`, `q8_tile.h`) | one bf16 tensor-core gemv for 1 to R rows |
| int8 activation quantization (q8_1-style) and folding the next norm into the quantizer | bf16 activations |
| int8 / bf16 cross-card exchange (`MK_OPT`) and the pcie p2p flag protocol | fp32 partials over nvlink, fused into the gemv epilogue |
| sm_86 / sm_89 / sm_120 builds and `ARCH=` | sm_86 only |
| `--kv f16/q8/auto`, the q8 kv path, 64/128-token blocks, `--kv-mem` | bf16 kv, one block size, the pool sized from the vram budget at build time |
| `MK_OPT`, `MK_BATCH_ROWS`, `MK_PF_CAP`, `MK_EMBD_DEVICE`, `MK_PREFILL=mk`, `MK_DRAFT_SAMPLING`, `MK_DEVICES` | fixed choices compiled in (card order fixed by nvlink topology) |
| fp16-accumulate prefill gemm (`f16.f16.f16.f16` mma) | bf16 x bf16 to fp32 |
| `-d 0..4`, the rejected-draft depth drop, 16-vs-32 row caps by slot count | one drafting policy under the row budget (section 7c) |
| `--slots 1..8` | `S = 8` compiled in |
| libllama for tokenization, the jinja chat template from the gguf | our qwen bpe tokenizer and a hardcoded qwen3.8 chat format (thinking, tool calls) |
| `mk_server.py`, `capi.cpp`, python benches and tests, docker, box scripts, Makefile, `make release` | one c++ server, tools and tests on cmake + ninja |
| `tools/llama_ref`, `logit_stats.h`, `--kld` against llama.cpp | the cpu fp32 reference over the same fp8 checkpoint (section 8) |
| `MK_PROFILE`, `MK_HOST_PROF`, `MK_PF_PROFILE` | a `profile` build preset |
| `--raw`, `--ids`, `--prompt-ids`, `--turn2`, `--score` cli surface | a small cli over the same pipeline, plus the harness tools |

kept as ideas and rewritten in the new style:
- the persistent megakernel;
- lazy-replay deltanet rollback;
- the paged kv pool with a radix prefix cache, deltanet snapshots, copy-on-write and preemption;
- the shared-snapshot prefill deduplication;
- the draft vocabulary;
- double-buffered split-k attention;
- prefill that leaves exactly the state the decode kernel expects;
- the server's scheduling policy: fair chunked prefill, longest-cached-prefix slot assignment, and turn replay for
  the prefix cache.

---

## 4. language, toolchain and rules

### 4.1 toolchain

- **latest stable clang/llvm** for host c++26 and cuda device code (clang's cuda mode, `--cuda-gpu-arch=sm_86`), with
  **libc++**. no gnu compiler or runtime. dynamic linking.
- **the system cuda toolkit**, the newest one that latest stable clang supports (clang lags new toolkits; `PINS.md`
  records which pair is in use).
- **device code without cudart.** clang compiles the kernels to cubin, the cubin is embedded in the binary, and the
  host loads it with the cuda driver api (`cuModuleLoadData`). the only nvidia user-space library at runtime is the
  driver.
- **cmake generating ninja; presets are the only interface:**

  | preset | purpose |
  |---|---|
  | `dev` | `-O1`, warnings as errors, trap-mode ubsan on host, device invariant checks on |
  | `release` | `-O3`, lto, device checks off |
  | `asan-ubsan` | host-only runtime sanitizers for the cpu test suite |
  | `profile` | release plus the per-phase profiler |
  | `lint` | clang-tidy over the whole graph (no reflection means clang sees everything) |

- the configure gate rejects anything but x86_64 linux with a clang/libc++/cuda toolkit present. it checks presence
  and minimum feature level, not exact versions.
- `-std=c++26 -fno-exceptions -fno-rtti`, warnings as errors, host assertions always on.
- `.clang-format` taken unchanged from cpp-starter (llvm base, tabs, 140 columns).

### 4.2 what comes over from cpp-starter (the 80/20)

copied and adapted:
- **AGENTS.md**: the "one concept, one primitive" table, minus the reflection rows and with clang as the
  authoritative compiler. also its sections on modules, zones, structural typing, data modeling, `struct` vs
  `class`, no runtime oo or rtti, no exceptions and the error algebra, ownership and borrowing, no shared
  ownership, mutation, concurrency and the reactor law, casting, numeric conversions, c remnants, i/o and
  formatting, api shape, boolean parameters, resource classes, allocation, testing, suppressions, enforcement, and
  the review checklist.
- **build files**: `CMakeLists.txt` skeleton and the centralized language-profile interface target,
  `CMakePresets.json`, `.clang-format`, `.clang-tidy` (now two profiles: a dialect profile for `src/`, `tests/` and
  `tools/`, and a quarantine profile for `device/`, `foreign/` and `unsafe/`).
- **`src/handle.cc`**: generational handles. slots, kv blocks and connections are referenced by
  `{index, generation}`.
- **`src/core.cc`**: the `std::expected` error shape and the "errors are decision surfaces" pattern.
- **`foreign/exec.cc` + `exec.backend.cc`**: the stdexec boundary. its narrow-abi quarantine was a workaround for a
  gcc ice (`PIN(gcc-gmf-stdexec-ice)`); under clang, retest importing stdexec inside the module graph and delete the
  workaround if it is clean.
- **`unsafe/net.cc`, `net.loop.cc`, `net.linux.cc`, `net.internal.h`**: the bounded single-threaded epoll reactor,
  linux backend only.
- **`src/http.cc`**: the bounded http parser and writer, extended for what an openai endpoint needs (section 6).
- **`tests/compile_fail/`**: the compile-fail harness (for example, a discarded `std::expected` must not compile).
- **`PINS.md`**: kept as the ledger of deliberate deviations and the toolchain pair in use. no exact pins.

dropped:
- reflection (`src/enums.cc`, the reflection parts of `tests/conformance.test.cc`);
- the gcc gate and the `upstream/` gcc defect ledger;
- the darwin/arm64 machine boundary and `net.darwin.cc`;
- static-linking and hardening-flag policy beyond what `dev` keeps.

### 4.3 what comes over from idris-mlir (principles, written into AGENTS.md)

- **representation first.** before adding a branch, flag or guard, change the data, types and invariants until the
  case can't be expressed. a handoff names, per obligation, the state made unrepresentable, the check moved to a
  boundary, or the branch deleted.
- **no static fact paid for at runtime.** every shape, offset, tile schedule, head split, buffer size and pool
  geometry is a compile-time constant from one model description, checked with `static_assert`. the only runtime
  facts are the input tokens, the context lengths, which slots are live, and the machine's identity.
- **one fact, held once.** model dimensions live in exactly one place (section 5.2); kernels, the baker, the
  host and the tests derive from it.
- **hard cutover, hard culling.** no dual paths or compatibility options. a replaced part is deleted with its tests
  and docs.
- **delete before you optimize.** every part names the fact only it supplies.
- **a blowup is a bug**, fixed at its cause, never capped.
- **a choice is a type or an enum case, never a `bool`.**
- **comments explain why**, not what, and cite no documents or rule numbers. new files carry
  `// SPDX-License-Identifier: AGPL-3.0-only`. upstream's notices say "version 3" without "or later", so `-only`
  matches; confirm when stamping.
- **report numbers honestly**: what was measured, on what, how many runs.
- **tests check behaviour** (logits against the reference, token streams, pool invariants, exit statuses), never
  incidental output.
- **done means built and checked here.** in the code-only phase: it compiles with zero warnings, lints clean, the
  device code passes its resource gates, and the cpu tests pass. type-checked is not built.

### 4.4 device code rules (new)

the cuda kernels are a quarantine zone. device code can't import c++ modules, so the dialect's "no headers" rule
can't hold there. inside the zone:
- c++26 language features that clang supports in device code: templates, concepts, `constexpr`/`consteval`,
  `if consteval`, structured bindings.
- no exceptions, no rtti, no virtual, no dynamic allocation, no `printf`.
- **no runtime shape arguments.** every dimension is a template parameter or a `constexpr` from the model
  description. kernel parameters carry only runtime facts (pointers into the per-cycle state, slot tables, lengths).
- inline ptx only inside small named primitives (`cp.async`, `ld/st.acquire/release.sys`, `mma.sync`, e4m3
  conversion), each documented with why.
- **every kernel's resource budget is a static fact**: threads per block, registers per thread, shared memory per
  block (≤ 99 kb on sm_86) and blocks per SM, all `static_assert`ed. the build fails if `ptxas -v` reports spills
  or registers over budget.
- device invariant traps exist only in the `dev` preset.
- the launch abi upward is one narrow, typed function per kernel, exported to the dialect as a module partition
  from `foreign/`.

---

## 5. repository layout

```text
AGENTS.md          normative rules (section 4), for humans and agents
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

### 5.1 module partitions (`src/`)

| partition | holds |
|---|---|
| `engine:model` | re-export of the model description: dims, layer kinds, head splits, tensor list |
| `engine:image` | the baked-image format: header, per-tensor extents, checksum; typed errors on mismatch |
| `engine:bake` | safetensors reader, tensor-parallel split, tile layout in stream order |
| `engine:tokenizer` | qwen byte-level bpe (pretokenizer, merges, special tokens), decode with utf-8 boundary handling |
| `engine:chat` | qwen3.8 chat format: system, user, assistant, tool turns, thinking and reasoning effort, tool-call parse |
| `engine:sampling` | sampling parameters as a closed type; exact top-k / top-p / min-p / temperature; seeds |
| `engine:kv` | paged bf16 kv pool: refcounted blocks, copy-on-write, block tables per slot |
| `engine:prefix` | radix tree over blocks, deltanet snapshots, lru eviction, host-ram and nvme tiers |
| `engine:sched` | slots, admission, preemption, chunked-prefill share, row-budget allocation, turn replay |
| `engine:device` | typed launch surface over the driver api: cards, streams, events, the resident-loop ring |
| `engine:pipeline` | stdexec request pipeline (section 6) |
| `engine:json` | bounded json parser and writer for the api schema |
| `engine:http` | cpp-starter's bounded http, plus bodies, sse and chunked streaming |
| `engine:openai` | `/v1/chat/completions` (streaming or not), `/v1/models`, `/health`, `/status` |

### 5.2 one model description, held once

`device/model.h` is the single source of every model and layout constant: plain `constexpr` with no macros, because
device code must read it. `foreign/model.cc` includes it and exports it as `engine:model`. nothing else defines a
dimension. tile schedules and buffer geometries are `consteval` functions over it, so a change to the model
description re-derives everything, and `static_assert`s catch any inconsistency.

---

## 6. parallel requests and the server

### 6.1 slots and the cycle

- `S = 8` slots, compiled in. each has its own deltanet state, mtp state and position. its kv comes from the shared
  pool.
- every cycle runs all live slots together. weights are read once per cycle whatever the row count.
- **the row budget** (~24 rows per cycle, section 2) is split across slots by a row allocator. each slot gets one
  row for its committed token, plus draft rows assigned where the expected accepted tokens per row are highest
  (from live acceptance stats). one busy slot gets a deep tree; eight busy slots get one or two drafts each. this is
  scheduling, outside the lossless boundary.
- per-slot work: attention, deltanet, draft checking and sampling each run per slot with its own row count.
- **shared-prefix (cascade) attention**: kv blocks that several slots share through the prefix cache (system
  prompt, a repo dump, a common transcript) are read once per cycle for all of them, and the softmax partials merge
  exactly. lossless, and it addresses the main multi-agent cost: kv reads scale with slots × context.
- **preemption**: as in geezuz. cached blocks are evicted first, then the newest admitted request pauses with its
  state cached and resumes at the head of the queue.

### 6.2 prefill alongside decode

- with nothing decoding, a prompt is prefilled whole.
- while others decode, prompts go in chunks for a fixed share of the time, alternating between the shortest
  remaining prompt and the longest-waiting one (geezuz's fair policy).
- prompts that arrive together and share an uncached snapshot point are prefilled once.
- a prefix-cache hit prefills only the uncached tail; turn replay keeps a conversation's generated tokens
  identical, so the cache keeps hitting.

### 6.3 the request pipeline (std::execution)

```text
http request (reactor) -> parse + validate (typed errors, bounded)
  -> tokenize + render chat format
  -> admit: slot + longest cached prefix            (scheduler, may wait)
  -> prefill tail chunks                            (sender per chunk, when_all over both cards)
  -> decode stream: one sender per cycle batch      (tokens -> detokenize -> sse frames)
  -> finish: stop / length / tool_calls; snapshot into the prefix cache
cancellation: client disconnect -> stop token -> slot released at the next cycle boundary
```

- one typed scheduler per card; a sender completes on a driver-api event.
- `when_all` over the two cards for weight upload, each prefill chunk, and kv tier spills.
- **the decode cycle is not a graph of senders.** it is one persistent kernel, and senders only submit cycles to it
  and collect what it returns. a sender per phase would reintroduce ~340 launches per cycle, which is exactly the
  overhead being removed.
- nvidia's `nvexec` targets nvc++/nvcc; on clang we write a small driver-api stream/event scheduler in `foreign/`.
- the http reactor stays cpp-starter's owner-driven epoll state machine; senders carry application values only,
  never kernel readiness.

### 6.4 the api surface

- `/v1/chat/completions`, streaming (sse) and non-streaming: `messages`, `tools`, `tool_choice`, `stream`,
  `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, and `chat_template_kwargs`
  (`enable_thinking`, `reasoning_effort`, `preserve_thinking`). the response carries `reasoning_content`,
  `tool_calls`, and `usage.prompt_tokens_details.cached_tokens`.
- `/v1/models`, `/health`, `/status` (slots, pool, drafting counters).
- `X-Request-Id` tagging as in geezuz.
- unsupported parameters are refused with a named 400; nothing is silently ignored. that covers
  presence/repetition penalties and anything the exact sampler doesn't implement.
- cpp-starter's http rejects request bodies (501) and writes whole responses. it gains bounded body reads and
  chunked sse writes in the same bounded style.

### 6.5 parallel throughput estimate (8 busy slots, code prompt)

24 rows across 8 slots is ~3 rows each, τ ≈ 2.5 per slot, cycle ≈ 17.5 ms:

$$\frac{8 \times 2.5}{17.5\ \text{ms}} \approx 1{,}140\ \text{tok/s aggregate},\quad \approx 140\ \text{tok/s per stream}$$

upstream geezuz on q8_0 at 8 requests: 610. long contexts lower this: 8 slots at 30k tokens each add ~8 ms of kv
reads per cycle unless the slots share prefixes (cascade attention).

---

## 7. the lossless optimization queue

### a. platform

1. **nvlink.** `nvidia-smi topo -m` should show NV4, and the stock driver should do p2p over nvlink with no p2p
   patch. check that the 2-slot bridge spacing matches PCIE_6 / PCIE_4.
2. **power**: limit at or above stock 350 w; persistence mode; locked clocks so boost doesn't oscillate between
   cycles.
3. **gddr6x memory overclock**: sweep for peak decode throughput (error replay hides errors as slowdowns), then
   accept the clock only if the determinism harness reproduces golden hashes over a long soak.
4. **memory thermals**: 12 of the 24 gddr6x chips sit under the xg7's passive backplate, and memory-bound decode is
   their worst case. watch memory junction temperature during the soak; add a backplate fan if it climbs.
5. **host**: iommu passthrough, hugepage-backed pinned memory for the embedding table and the rings, the resident
   loop's host thread pinned to one core.

### b. decode core

6. **barrier-free dataflow megakernel**:
   - producer warps stream weights continuously through a `cp.async` ring in shared memory;
   - consumer warps wait on per-layer and per-tile counters instead of ~340 grid barriers;
   - ~8 mb of shared memory plus 6 mb of l2 gives ~15 µs of weight run-ahead at full bandwidth;
   - the schedule is generated at compile time from the model description.
7. **fp32 all-reduce over nvlink, fused into the gemv epilogue**: partials are sent per tile as they finish. both
   cards add in the same order, so the residual stays bit-identical on both.
8. **serial phases hidden under the weight stream**: deltanet core, attention, the lm_head merge, sampling and the mtp
   draft steps. the mtp steps overlap the next cycle's layer-0 prefetch.
9. **fixed tensor split**: attention q and kv heads, deltanet value heads (16 key / 48 value; 8 / 24 per card), ffn
   rows 8,704 per card, lm_head vocabulary halves.
10. **device-resident loop**: the kernel never exits. the host exchanges cycle descriptors and results through a
    mapped, pinned ring.
11. **lazy-replay deltanet rollback** with fp32 state.
12. **attention**: split-k flash decoding on bf16 kv tiles, fp32 softmax and accumulation, double buffered, plus
    cascade attention over shared blocks (section 6.1).
13. **paged bf16 kv pool, radix prefix cache, deltanet snapshots, preemption.**

### c. drafting (exact by construction)

14. **one verify path for 1 to R rows** under the row budget (section 2), with no separate small-batch kernel.
15. **tree drafts**: mtp top-k branches plus suffix branches verified together, with tree attention masks. deltanet
    state is kept per branch along the tree, and only the accepted path is replayed.
16. **suffix / prompt-lookup drafting** from the slot's context and your own past sessions. agent output (file
    rewrites, diffs, tool-call json) copies context a lot.
17. **drafter weights baked at reduced precision**: one drafter format, chosen by measured acceptance against bytes.
18. **personalized drafter**: retrain only the mtp head on your own chats and coding sessions, main model frozen.
19. **the row allocator** (section 6.1) adapts tree shape and depth to live acceptance across slots.

### d. lm_head

20. **certified-bound vocab pruning** (research):
    - score every row with a low-precision shadow copy of lm_head and bound each row's error;
    - drop rows that provably can't make the top-k, and recompute the survivors in bf16;
    - greedy and top-k sampling stay exact; requests without top-k use the full softmax;
    - saves ~0.6 gb per card per cycle if the bounds are tight enough.

### e. prefill and caching

21. **bf16 x bf16 to fp32 tensor-core gemm** on e4m3 converted in registers: ~71 tflops per card, so ~2,000 tok/s
    total is a realistic target.
22. **prefill all-reduces over nvlink**: ~42 mb per reduce for a 2,048-token chunk in fp32, overlapped with the gemm.
23. **persistent prefix-cache tiers**: kv blocks and deltanet snapshots spill to host ram, then nvme, and survive
    restarts. exact, because cached state is bit-identical.

---

## 8. verification (built first; gates everything)

24. **cpu reference (`geezuz-ref`)**: the model computed straight from the fp8 checkpoint in fp32 on the cpu. it is
    written independently of the kernels: straightforward loops, threads via stdexec. it produces teacher-forced
    logits for plain text and for your own agent transcripts.
25. **rulers**: top-1 agreement, kl and perplexity against the reference, with the noise floor measured.
26. **tokenizer and chat-format tests** against fixtures generated once from qwen's reference tokenizer and template
    and committed under `data/`.
27. **determinism harness**: a deterministic mode plus golden logit hashes per prompt and per slot mix. every clock,
    kernel or schedule change reproduces them bit-for-bit, or, for an intentional reordering, stays within the noise
    floor and re-goldens with the reason recorded.
28. **host property tests**:
    - kv pool and prefix cache fuzzing against a content-checking fake device (ported from geezuz's
      `kv_cache_fuzz` and `kv_cache_scen`);
    - a scheduler simulation over a fake engine (ported from `sched_sim`): preemption, budgets, no lost tokens,
      no hangs;
    - http/json/sse bounds tests in the cpp-starter style.
29. **static device gates** (run in the code-only phase): `ptxas -v` for every kernel with zero spills and
    registers within budget; shared memory and occupancy asserted; schedule geometry `static_assert`ed against the
    model description.
30. **per-phase profiler** in the `profile` preset: time per phase, warp arrival, peer wait, dram utilization.

---

## 9. order of work

### phase a: code only (now; no gpu)

**where it runs:** a linux x86_64 environment with latest clang/llvm and the cuda toolkit, e.g. the cloud claude code
containers. no gpu is needed: clang compiles the device code and ptxas reports resources. macos can't host the cuda
toolkit, so the mac is for editing only.

**done means**: `dev`, `release` and `lint` build with zero warnings, the static device gates pass, and the cpu test
suite passes. nothing claims runtime correctness of the kernels until phase c.

lanes for parallel subagents. lanes implement and hand over a change that builds; the coordinator merges onto
`main`, runs every check on the merged tree, and sends exact fixes back. commits go straight to `main` (no
feature branches, no prs).

| lane | work | depends on |
|---|---|---|
| L0 scaffold | AGENTS.md and CLAUDE.md, the cpp-starter 80/20 (section 4.2), cmake gate, presets, language profile, clang-tidy profiles, zones, compile-fail harness, PINS.md | — (first) |
| L1 cull | delete everything in section 3 that has no successor yet: tp1, non-sm_86, k-quants, q8 kv, env switches, python server and tools, docker, Makefile, llama_ref. move what survives into the zones | L0 |
| L2 model + image + bake | `device/model.h`, `engine:model`, `engine:image`, `engine:bake` (safetensors reader, split, stream-order layout) | L0 |
| L3 tokenizer + chat | `engine:tokenizer`, `engine:chat`, fixtures in `data/` | L0 |
| L4 server | `engine:json`, `engine:http` extensions, `engine:openai`, `engine:pipeline`, `engine:sched` with the fake-engine simulation | L0 (sched also L5) |
| L5 kv + prefix | `engine:kv`, `engine:prefix` (tiers included), ported fuzz and scenario tests | L0, L2 |
| L6 device | e4m3/bf16 primitives, the fp8 gemv/gemm, attention (plus cascade), deltanet, the decode megakernel as a straight port first, prefill kernels, driver-api loader, the resident ring, static gates | L0, L2 |
| L7 reference + harness | `geezuz-ref`, rulers, the determinism-harness tooling | L0, L2, L3 |

L6 ships the straight port first, with barriers still in place, as the correctness baseline. the dataflow rewrite
(item 6) is a separate change once the baseline exists. both are hard cutovers inside L6's history, never two
kernels kept side by side.

### phase b: hardware bring-up (when the box is up)

- nvlink and the stock-driver p2p check, power, thermals (items 1, 2, 4).
- upstream `v0.3.2` on q8_0 with `MK_PROFILE=1`, with and without the bridge: the baseline numbers.

### phase c: correctness on gpu

- baked images load; the straight-port kernel matches `geezuz-ref` within the noise floor; golden hashes recorded.
- server end to end with 8 parallel streams, tools and thinking.

### phase d: speed, each change measured under the harness

dataflow megakernel (6–8), resident loop (10), prefill (21–22), drafting (14–19), memory overclock (3), then the
research item (20).

---

## 10. expected outcome

| config | cycle | 1 stream code (τ ≈ 4.46) | 1 stream prose (τ ≈ 2.16) | 8 streams code, aggregate |
|---|---|---|---|---|
| upstream geezuz q8_0 (pcie, 280 w) | ~25 ms | ~143 | ~85 (est) | 610 |
| straight port on fp8 + nvlink (phase c) | ~23 ms (est) | ~190 | ~95 | ~700 |
| + dataflow core, resident loop, reduced-precision drafter | ~18 ms | ~250 | ~120 | ~1,050 |
| + memory overclock (~8%) | ~16.8 ms | ~265 | ~128 | ~1,120 |
| + tree, suffix and row-allocated drafting | ~17.5 ms | ~330 | ~170 | ~1,150 |

all estimates, and they compound. the dataflow row is the one physics most clearly supports. drafting gains depend on
your traffic. long multi-slot contexts cost kv bandwidth unless the slots share prefixes.

---

## 11. risks

- **clang cuda and the latest toolkit**: clang can trail the newest toolkit. use the newest pair clang supports and
  record it in PINS.md.
- **stdexec on clang + libc++**: the module boundary may still need the narrow-abi quarantine. retest, then keep or
  delete it.
- **kernels unverified until phase c**: phase a can prove builds, resource budgets and host logic, not kernel
  correctness. the cpu reference exists so phase c catches mistakes on day one.
- **the bf16 compute ceiling**: the lossless contract caps verify width at ~24 rows per cycle. measure the real
  crossover on hardware and retune the row budget constant.
- **memory overclock without ecc**: errors can be silent. accept a clock only after a long golden-hash soak at
  operating temperature.
- **checkpoint drift**: the baker checks the checkpoint hash and the image header carries it, so a re-uploaded
  checkpoint is refused, never half-loaded.
