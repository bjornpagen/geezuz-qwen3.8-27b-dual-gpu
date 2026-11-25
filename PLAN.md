# PLAN: geezuz rewritten as one engine for one model on one box

status: proposed, awaiting approval. no engine code has changed yet.

this plan has two parts:
- **part 1, must do**: the mandated cleanup, plus the performance work that is certain from first principles. it
  goes in regardless.
- **part 2, experiments**: everything with an uncertain payoff. each has a hypothesis, why it's uncertain, how to
  measure it, and when to kill it.

speed numbers are back-of-envelope estimates from upstream geezuz's published measurements and 3090 specs, not
measurements on this box; docs/HARDWARE.md holds the evidence and docs/ADVERSARIAL.md checks every claim against it.
the code-only phase (section 1.10) builds and type-checks; real numbers start when the hardware is up.

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
in e4m3 and 2.62b in bf16 (lm_head, embeddings, norms, small linear_attn params, mtp fc; 3.08b with the dropped vision
tower). the engine computes exactly that function:

- e4m3 to bf16 conversion is exact; the block scale is applied in fp32.
- weights only are quantized. activations stay bf16, even though the checkpoint says `activation_scheme: dynamic`.
- every matmul accumulates in fp32 on the tensor cores; the residual stream and deltanet state are fp32; the kv cache
  is bf16.
- lm_head runs in bf16 with exact top-k and argmax merged across the two cards.
- speculative decoding verifies exactly (greedy is exact; sampling uses rejection sampling).
- sampling parameters are exact for every value the server accepts; anything else is refused with a named error.
- the only allowed differences are floating-point summation order and the tensor core's own fp32 accumulation, which
  on sm_86 is not ieee: it sums products in blocks of 8, aligned to the largest and truncated (docs/HARDWARE.md).

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

| item | bytes | at 936 gb/s | at ~820 gb/s |
|---|---|---|---|
| body (fp8, 64 layers, half of each matrix) | 12.19 gb | 13.0 ms | 14.9 ms |
| lm_head (bf16, half of 248,320 vocab) | 1.27 gb | 1.36 ms | 1.55 ms |
| **main pass** | **13.46 gb** | **14.4 ms** | **16.4 ms** |
| mtp head (d passes for d drafts: catch-up, then d − 1 steps) | 0.24 gb per pass: 0.95 gb at d = 4 | 1.0 ms | 1.16 ms |
| draft lm_head (32k vocab split across cards, d steps) | 0.17 gb per step: 0.67 gb at d = 4 | 0.72 ms | 0.82 ms |
| deltanet state (fp32, read and written once per cycle) | 151 mb per live slot: 1.21 gb at 8 slots | 1.29 ms at 8 | 1.47 ms at 8 |
| kv reads (bf16, 2 kv heads per card x 16 layers) | 32 kb x context tokens, per slot | 1.0 ms per 30k tokens | 1.2 ms per 30k |
| mtp kv reads (its own attention layer) | 2 kb x context tokens, per slot and pass | — | — |

~820 gb/s is 90% of the 912 gb/s cuda actually gets: geforce ampere runs cuda work in p2, memory at 9501 instead of
9751 mhz (1.7). a clean read stream reaches 93-96% of peak, so 90% leaves room for kv, state and nvlink traffic sharing
the controllers; dependency stalls are not in the rate, they are $t_{\text{serial}}$. a 1-stream cycle (d = 4, 4k
context) moves ~15.4 gb per card, an 8-stream cycle (d = 2, 4k context each) ~16.7 gb.

$$t_{\text{cycle}} \approx \max\left(t_{\text{bytes}},\ t_{\text{flops}}\right) + t_{\text{serial}}, \qquad \text{tok/s per slot} = \frac{\tau}{t_{\text{cycle}}}$$

**the compute ceiling.** on the 3090, bf16 tensor-core math with fp32 accumulation runs at ~71 tflops dense per card
at the rated 1695 mhz boost (it scales with the sustained clock), half the fp16-accumulate rate. one verify row costs
~25.6 gflop per card (24.3 body, 1.3 lm_head), ~0.36 ms at peak and ~0.5 ms realistically. rows are bought in n8 tiles
(weights are the mma's m16 operand, tokens its n8 operand), so a row budget between multiples of 8 costs the next one:

$$t_{\text{flops}}(R) \approx 4\ \text{ms} \times \lceil R/8 \rceil$$

per tensor core, one 256-byte weight fragment arrives every ~152 clocks at 936 gb/s and ~173 at 820 gb/s (both at
1695 mhz), and each n8 tile costs it 32 clocks: 24 rows use ~96, 32 rows ~128, 40 rows ~160. compute crosses the
weight stream between 32 and 40 rows. the engine spends a **row budget of 24 rows per cycle** (3 tiles), shared by all
slots and their drafts: a compile-time constant. 32 (4 tiles) is the first thing measured on hardware.

**vram per card** (24 gib, ~25.3 gb usable).
- weights ~13.7 gb (main pass 13.46, mtp head 0.24; the draft lm_head is a view of lm_head, 1.3), ~1 gb workspace and
  ~0.6 gb of deltanet state leave ~9.5 gb: about 275k tokens of bf16 kv at 34 kb per token (32 kb for the 16 layers,
  2 kb for the mtp layer's own attention), shared by all slots, snapshots included.
- deltanet state is 75.5 mb per slot (fp32), ~600 mb at 8 slots, plus ~3 mb of conv state per slot, plus snapshots
  (75.5 mb each).
- the embedding table (2.54 gb bf16) lives in pinned host memory on numa node 0. each row a cycle needs crosses pcie
  once, into a vram staging buffer, fetched by the step that first uses it (an mtp draft step or the verify pass).

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

- **latest stable clang/llvm** for host c++26 and cuda device code, with **libc++** and `import std`. no gnu
  compiler or runtime, no apple clang. dynamic linking.
- **no cuda sdk at build time.** device translation units compile with clang in device-only mode straight to ptx:
  `-x cuda --cuda-gpu-arch=sm_86 --cuda-device-only -nocudainc -nocudalib -S`, with the ptx isa version set
  explicitly (the newest llvm emits for `.target sm_86`). device code includes no sdk headers: special registers,
  shuffles, barriers and memory-order operations come from nvvm builtins or inline ptx inside named primitives in
  `device/`; bf16 and e4m3 are our own types. the same ptx comes out on every platform.
- **one load path, no cudart.** the ptx is embedded in the host binary with `#embed` (in `foreign/`) and loaded with
  `cuModuleLoadData`, which jit-compiles it; the driver's compute cache keeps start-up cheap. `foreign/` declares the
  driver-api subset it uses and `dlopen`s `libcuda.so.1` at runtime: no link-time nvidia dependency. where that load
  fails (macos) engine start returns `DeviceError::DriverUnavailable`. `ptxas -v` runs as a static gate on the same
  ptx wherever a cuda toolkit exists (linux). `PINS.md` records the llvm version, the ptx isa version and the minimum
  driver that can jit it.
- **cmake generating ninja**, presets only:

  | preset | purpose |
  |---|---|
  | `dev` | `-O1`, warnings as errors, trap-mode ubsan on host, device invariant checks |
  | `release` | `-O3`, lto, device checks off |
  | `asan-ubsan` | host-only runtime sanitizers for the cpu test suite |
  | `profile` | release plus the per-phase profiler |
  | `lint` | clang-tidy over the whole graph |

- the configure gate accepts exactly two machines: darwin arm64 (development: everything but running kernels) and
  x86_64 linux (production), both with latest clang and libc++. the reactor's wait backend is chosen in the build
  graph, one per platform (kqueue on darwin, epoll on linux), never by preprocessor tests in dialect code.
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

dropped: reflection (`src/enums.cc` and the reflection tests), the gcc gate and gcc defect ledger, the
static-linking and hardening policy beyond `dev`. kept: cpp-starter's darwin kqueue backend (`net.darwin.cc`) for
development, next to the linux epoll backend.

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
- inline ptx only inside small named primitives in `device/prims/`, each documented with why, and each stating the
  ptx isa version and sm level it needs, checked at compile time.
- **every kernel's resource budget is a static fact**: threads, registers, shared memory (≤ 99 kb per block) and
  blocks per SM are `static_assert`ed. the build fails if `ptxas -v` reports spills or registers over budget.
- device invariant traps exist only in `dev`.
- one narrow typed launch function per kernel, exported upward as a module from `foreign/`.
- **rounding is pinned in the ptx**, so neither llvm nor the driver's jit can change a result:
  - every fp32 operation is ieee round-to-nearest (`add.rn`, `mul.rn`, `fma.rn`, `div.rn`, `sqrt.rn`) or a fixed
    routine built from them;
  - no `.approx` and no `.ftz` anywhere;
  - contraction is explicit (`-ffp-contract=off`, fmas spelled out);
  - exp, log, rsqrt and the silu, sigmoid and softplus built on them are fixed fp32 algorithms, specified once.
    `geezuz-ref` follows the same specification, and each is tested over every fp32 input.

### device style: the newest idiom sm_86 has (hard cutover)

sm_86 is the only binary target: anything built for sm_89 or later does not load on these cards. "newest style"
means the newest ptx isa llvm emits for `.target sm_86`, every async, synchronization and cache-control feature
ampere has, and hopper-era kernel architecture emulated where the silicon lacks it. no legacy idioms. the step-1
baseline keeps geezuz's algorithm, but its building blocks are these primitives too.

**required idioms**, each a named primitive in `device/prims/` with a microbenchmark in `bench/ubench/`:
- **warp specialization**: producer warps own the weight stream, consumer warps own the math. roles are fixed per
  kernel and come from the compile-time schedule. sm_86 has no `setmaxnreg`, so a producer warp holds as many
  registers as a consumer: as few producer warps as keep ~11 kb per SM in flight (one `cp.async.cg` warp instruction
  moves 512 b; an SM needs one every ~76 clocks), counted by a microbenchmark.
- **async copy**: `cp.async.cg` 16-byte transfers with the `.L2::256B` hint (the dram chunk), whole contiguous 16 kb
  tiles, completion signalled with `cp.async.mbarrier.arrive.noinc` into mbarriers; `cp.async.commit_group` /
  `wait_group` only where an mbarrier does not fit.
- **mbarrier inside a block**: `mbarrier.init` / `arrive` and phase-parity waits, multi-stage rings. sm_86 has no
  `try_wait` and no `expect_tx` (both sm_90), so a wait is a `test_wait.parity` spin with `nanosleep` backoff and
  `cp.async` completion is counted in arrivals, not bytes. no `__syncthreads` in steady state; named barriers
  (`bar.sync id, n`) only for fixed sub-groups. a ring of ≤ 7 slots could signal "slot free" with named-barrier pairs
  instead; a microbenchmark picks the cheaper.
- **l2 control**: `createpolicy.fractional` with evict_first on every single-use stream (weights, kv, deltanet state)
  and evict_last only on what is reused within microseconds (activations, partials, flags);
  `prefetch.global.L2::evict_last` for run-ahead past the shared-memory ring, `applypriority` back to evict_normal
  after use; `discard.global.L2` on dead scratch.
- **fragments**: weights need no `ldmatrix`: the bake stores each lane's a-fragment bytes for two k16 steps as one
  contiguous 16 bytes, so a warp's 512 bytes are one conflict-free `ld.shared.v4`. `ldmatrix` (and `.trans`) from
  xor-swizzled shared memory is for the bf16 activation b-fragments. `mma.sync.m16n8k16` bf16 x bf16 to fp32; register
  double-buffering across k-steps (the wgmma pipeline, by hand).
- **reductions**: `redux.sync` for integers, butterfly `shfl.sync` for floats in a fixed order (determinism).
- **cross-card**: push, never pull: weak stores into the peer's receive slot, then a single-writer
  `st.release.sys` flag carrying a monotonic epoch; the peer polls its own memory with `ld.acquire.sys` and `nanosleep`
  backoff. one release per tile, never per element. no peer atomics (native atomics over the bridge are not
  established). sm_86 has one memory-sync domain, so a `.sys` release also waits for in-flight host-ring writes: those
  happen once per cycle, away from release points.

**emulated, because the silicon lacks it**:
- **tma**: producers issue the baked image's precomputed 16-byte transaction stream. the image is laid out as exactly
  that stream, so a producer is a pointer bump plus `cp.async`, nothing else.
- **clusters / distributed shared memory**: none. cross-SM handoff goes through l2-resident flags and buffers with an
  explicit evict_last policy.
- **fp8 cvt**: an exact e4m3-to-bf16 sequence on a bake-permuted encoding, straight into mma a-fragments in
  registers. 5 integer ops (`and`, shifts, one `lop3`) place each pair's payload bits, then `fma.rn.bf16x2` by 2¹²⁰
  with a -0 addend fixes the exponent: 7 instructions per 4 weights. the 2¹²⁰ is never folded into the block scale,
  which is exact only while partial sums stay normal. bf16 scales widen to fp32 with an integer shift. tested
  exhaustively over all 256 codes, through the mma.
- **wgmma**: consumer warpgroups of 4 warps sharing one stage, register-blocked mma with the next stage's `ldmatrix`
  overlapped.

**banned**:
- dp4a and any int8 or fp16-accumulate math path;
- cooperative-groups `grid.sync` and atomic-counter grid barriers in the dataflow core (allowed only in the step-1
  baseline, deleted at step 2);
- per-phase kernel launches;
- weight-stream loads without a cache policy;
- unswizzled shared tiles with bank conflicts;
- texture / surface paths, `printf`, dynamic shared allocation decided at runtime.

## 1.3 layout and the model description

```text
AGENTS.md          normative rules (section 1.2)
CLAUDE.md          one line: read AGENTS.md
PLAN.md            this file
PINS.md            toolchain pair in use and deliberate deviations
CMakeLists.txt     gate, language profile, targets
CMakePresets.json  dev, release, asan-ubsan, profile, lint
src/               one named module per component, `engine.<name>`, each its own target and directory (dialect)
device/            cuda kernels and the model description header (quarantine)
foreign/           cuda driver api, stdexec boundary, model-description re-export (quarantine)
unsafe/            epoll reactor, mmap, pinned and hugepage memory (quarantine)
apps/              geezuz-server, geezuz (cli), geezuz-bake, geezuz-ref (dialect)
tests/             unit, property and compile-fail tests (dialect)
bench/             benchmark clients (dialect)
data/              draft vocabulary, tokenizer and chat-format fixtures
```

| module | holds |
|---|---|
| `engine.model` | re-export of the model description: dims, layer kinds, head splits, tensor list |
| `engine.image` | baked-image format: header with checkpoint hash, per-tensor extents; typed errors on mismatch |
| `engine.bake` | safetensors reader, tensor-parallel split, tile layout in stream order |
| `engine.tokenizer` | qwen byte-level bpe, special tokens, utf-8-safe streaming decode |
| `engine.chat` | qwen3.8 chat format: system, user, assistant, tool turns, thinking, reasoning effort, tool-call parse |
| `engine.sampling` | sampling parameters as a closed type; exact top-k / top-p / min-p / temperature; seeds |
| `engine.kv` | paged bf16 kv pool: refcounted blocks, copy-on-write, block tables |
| `engine.prefix` | radix tree over blocks, deltanet snapshots, lru eviction |
| `engine.sched` | slots, admission, preemption, chunked-prefill share, row budget, turn replay |
| `engine.device` | typed launch surface over the driver api: cards, streams, events, the resident ring |
| `engine.pipeline` | stdexec request pipeline |
| `engine.json`, `engine.http`, `engine.openai` | bounded json, http with bodies and sse, the openai endpoint |

**one model description, held once.** `device/model.h` is the single source of every model and layout constant: plain
`constexpr`, no macros, because device code must read it. `foreign/model.cc` re-exports it as `engine.model`. tile
schedules and buffer geometries are `consteval` functions over it. nothing else defines a dimension.

**the baked image.** `geezuz-bake` reads the safetensors checkpoint once and writes one image per card:
- verifies the checkpoint hash and stamps it in the header;
- splits for tensor parallel 2 and drops the vision tower;
- lays out the image as the exact transaction stream producers issue:
  - each 128x128 e4m3 tile is one contiguous 16 kb run (whole 256-byte dram chunks, for row locality);
  - inside it, each lane's 16 bytes are in m16n8k16 a-fragment order for two k16 steps, bit-permuted for the decode;
  - scales sit in a small separate array;
- rejects the e4m3 nan codes (0x7f, 0xff) with a typed error, so the device has no nan path;
- permutes each card's lm_head rows so its half of the draft vocabulary is a contiguous prefix: the draft lm_head is a
  view, not a copy;
- converts nothing lossy.

the engine mmaps the image and copies it to vram with no layout decisions. a mismatched image is refused at the
boundary.

## 1.4 server and parallel requests

- **`S = 8` slots**, each with its own deltanet state, mtp state and position; kv comes from the shared pool.
- **every cycle runs all live slots together**, so weights are read once per cycle.
- **row budget**: rows come in n8 tiles, so 24 or 32 per cycle. step 2 ships 24 and measures 32 first; step 1 runs 32,
  as upstream did (its stream leaves the tensor cores half idle). a fixed, simple allocation ships first: one row per
  live slot for its committed token, and the rest split evenly as draft depth, capped at 4 per slot. an adaptive
  allocator is an experiment (2.3).
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

a straight port must not carry upstream's numerics outside the contract. step 1 uses the fixes from the start:
- the fp16 state operand of upstream's chunked prefill deltanet: replaced by the recurrence (prefill, below);
- P rounded to 16 bits before PV, in decode and prefill attention: settled by the owner's decision on P;
- fp16 activations and fp16-accumulate prefill gemms: culled (1.1);
- fast-math `expf` / `logf` / `rsqrtf`: replaced by the pinned routines (1.2).

**step 2: the dataflow core. certain from first principles, so it is part 1:**
- **producer/consumer warp specialization**: producers stream weights continuously through a `cp.async`
  shared-memory ring; consumers wait on per-layer and per-tile counters instead of ~340 grid barriers.
  - weights have no data dependency, so producers wait only for ring space.
  - the ring is 64-80 kb per SM, 5.4-6.7 mb per card: ~6-8 µs at the stream rate, ≤ ~11 µs with
    `prefetch.global.L2::evict_last` run-ahead.
  - little's law needs only ~1 mb in flight per card, so the ring is a jitter buffer: it rides over dependency hops
    (~1-3 µs each), not over serial phases.
- **all-reduce fused into the gemv epilogue**: fp32 partials go over nvlink per output tile as they finish. 128
  exchanges per main pass, each exposing only its last tile and one handoff (~1-2.5 µs).
- **one arithmetic per output element**: the k-split is static and independent of the row count, and attention
  splits the context into chunks at fixed absolute positions, merged in order; balance comes only from assigning
  chunks to ctas. each output tile's epilogue also writes its per-row partial sum of squares, which the next phase's
  rms norm adds in a fixed order, with no extra pass.
- **serial phases made bandwidth-bound, not hidden**: at 8 slots the deltanet core alone moves ~25 mb of state per
  layer (~27 µs at full rate; upstream's took ~80 µs), far past the run-ahead. so:
  - the core runs as one wave over all SMs at state bandwidth. value columns are independent chains; a head's 128
    columns stay in one cta for the output norm;
  - state reads are a dependency-free producer stream;
  - split-k attention runs from a balanced per-cycle plan that one warp computes on the device;
  - dependency-free weights stream while the core and attention run: `in_proj_z` is consumed only after the core,
    the gate half of `q_proj` only after attention;
  - the mtp steps stream 0.41 gb each and are bandwidth-bound already.
- **device-resident loop**: the kernel stays resident, builds each next cycle itself from the accepted tokens, and
  exchanges results through a mapped, pinned ring. no launch or readback per cycle.
  - the host's mailbox (admissions, cancellations) is read with a load issued at cycle start and consumed at cycle
    end, so its pcie round trip is never on the critical path.
  - for a prefill chunk the loop yields at a cycle boundary: one relaunch (~10 µs) against ~1 s chunks.
- **compile-time schedule**: generated from the model description; resource budgets are `static_assert`ed. the
  producer stream depends only on the draft depth, so a cycle's shape is an enum case (d = 1 to 4), each a
  compile-time stream program.

why it's certain: upstream's measured cycle (~20.9 ms on q4_k_m) is about twice its weight-read floor (~9.3 ms),
so about half of every cycle is not moving weights. the exact split between barriers, exchange and serial phases gets
measured in phase b, but the gap itself is arithmetic.

**prefill (forced by the contract):**
- bf16 x bf16 to fp32 tensor-core gemm on e4m3 converted in registers.
  - each card does 24.3 gflop per token, so 71 tflops is a ~2,900 tok/s ceiling.
  - fp32-accumulate gemms sustain ~50-60 tflops on the 3090 in isolation and less in situ, so expect ~1,300-2,000
    tok/s total.
  - ~1,600 is the planning number, 2,000 the stretch.
- fused `silu(gate) * up`;
- deltanet as the decode recurrence itself, in fp32 on the cuda cores.
  - value columns are independent chains: 3,072 per layer per card, ~0.2-0.4 ms per layer per 2,048-token chunk,
    ~1-2% of prefill.
  - the chunked (wy) form re-associates gate products, and its tensor-core operands round the state (upstream: fp16),
    so it is outside the contract.
  - the recurrence leaves bit for bit the state decode would have;
- causal flash attention over the cache; P enters the PV mma as the owner's decision on P says (docs/ADVERSARIAL.md);
- all-reduces over nvlink, overlapped with the gemm.

## 1.7 platform

- **nvlink**: the stock driver does p2p over nvlink with no patch. the owner confirmed the 2-slot bridge fits the
  water-blocked cards in PCIE_6 / PCIE_4. bring-up checks:
  - persistence mode on (links stay inactive without it);
  - `nvidia-smi topo -m` shows NV4 and `nvlink -s` shows 4 x 14.062 gb/s;
  - peer access works both ways.
- **power**: limit at the fe maximum of 400 w. power cut by a cap comes out of the core clock, and the row budget
  needs it ($t_{\text{flops}} \propto 1/f_{\text{core}}$).
  - persistence mode on.
  - clocks locked one step below what 400 w sustains under the heaviest phase mix. a lock above that is undone by
    throttling, so accept it only if the `sw_power_cap` event counter stays flat over a 10-minute soak.
  - all settings come from a systemd unit after nvidia-persistenced; nothing persists on its own.
- **memory clock**: cuda runs geforce ampere in p2, memory at 9501 mhz (912 gb/s), not 9751 (936). bring-up tries
  `-lgc F,F -lmc 9751,9751` in one call, then `--cuda-clocks=1`. otherwise the budget stays on 912, and an overclock
  counts from 9501.
- **memory overclock**: sweep a pure-read kernel per offset step at operating temperature; the edr knee is where gb/s
  stops tracking the clock.
  - the in-rating ceiling is ~21 gb/s (micron's -21 bin): +10% over p2's 19.0, worth ~+8% of a stream-bound cycle.
    adrien's +15% raw gave only +4% at tp2 because allreduce latency and the drafter co-bound his decode.
  - edr's crc covers link data only, not the array or the command bus. acceptance therefore needs a long soak at
    temperature: an on-device hash of the whole weight image every few minutes, a pattern test over the free kv
    region, and the golden hashes.
  - the engine refuses to start if the memory clock differs from the accepted one, and a low-rate weight-hash scrub
    runs in production.
- **memory thermals**: 12 of the 24 gddr6x chips sit under the xg7's passive backplate. the 110 c junction throttle is
  the binding limit, so a backplate fan is the default. nvml does not report the junction temperature on the 3090;
  reading it needs the `gddr6` tool and `iomem=relaxed`, which is an owner decision.
- **host**: iommu passthrough. both slots hang off cpu0, so every pinned buffer (embedding table, rings) is
  hugepage-backed on numa node 0, and the resident loop's host thread is pinned to a cpu0 core.

## 1.8 verification (built first; gates everything)

- **cpu reference (`geezuz-ref`)**: the model computed straight from the fp8 checkpoint in fp32 on the cpu, written
  independently of the kernels. teacher-forced logits for plain text and for your own agent transcripts.
  - it rounds to bf16 exactly where the engine does: the activation points, listed once in the model description.
  - it carries a bit-exact model of the sm_86 mma: 8-product blocks aligned to the largest and truncated, as measured
    in docs/HARDWARE.md. that model tests device primitives exactly and sizes the noise floor the hardware alone
    causes.
- **decode-path ruler**: compares greedy continuations token by token against the reference. a prefill-only ruler
  can't see decode kernels, and adrien's gotcha #2 is a decode kernel that read bit-identical on the prefill ruler.
- **rulers**: top-1 agreement, kl and perplexity, with the noise floor measured.
- **tokenizer and chat-format fixtures** generated once from qwen's reference tokenizer and template and committed.
  check `tokenizer.json` for a baked truncation (adrien's gotcha #7).
- **determinism harness**: a deterministic mode plus golden logit hashes per prompt and per slot mix. every clock,
  kernel or schedule change reproduces them bit-for-bit, or stays within the noise floor and re-goldens with the
  reason recorded.
- **batch invariance**: a slot's logits are bit-identical whatever the other slots, the row count and the draft depth
  (one arithmetic per output element, 1.6). so speculative greedy output equals plain greedy output token for token,
  and the golden hashes per slot mix check it.
- **host property tests**: pool and prefix fuzzing; a scheduler simulation over a fake engine (preemption, budgets,
  no lost tokens, no hangs); http/json/sse bounds.
- **static device gates** (code-only phase): `ptxas -v` per kernel with zero spills and registers in budget;
  shared memory and occupancy asserted; schedule geometry `static_assert`ed against the model description; the ptx
  has no `.approx`, no `.ftz` and no fp32 `add` / `mul` without `.rn`.
- **measurement protocol** (adrien's gotcha #9: speculative decode varies ~10% between runs through content-dependent
  acceptance): compare speed at temperature 0, within one boot, normalized per accepted token, best of n.
- **profiler** in the `profile` preset: time per phase, warp arrival, peer wait, dram utilization.

## 1.9 must-do outcome (estimates)

| config | 1-stream cycle | 1 stream code (τ ≈ 4.46) | 1 stream prose (τ ≈ 2.16) | 8-stream cycle | 8 streams code, aggregate |
|---|---|---|---|---|---|
| upstream geezuz q8_0 (pcie, 280 w), measured | 24.9 ms | ~179 (est; 143 measured at τ ≈ 3.6) | ~87 (est) | 43.8 ms, 32 rows | 610 |
| step 1: straight port on fp8 + nvlink | ~25 ms | ~180 | ~87 | ~42 ms, 32 rows | ~640 |
| step 2: dataflow core + resident loop | ~19 ms | ~230 | ~112 | ~21-24 ms, 24 rows | ~900-1,000 |
| + memory overclock (~21 gb/s) | ~17.5 ms | ~255 | ~123 | ~19-22 ms | ~980-1,100 |

estimates. the assumptions:
- every byte moves at ~820 gb/s (90% of p2), at 4k tokens of context per slot;
- d = 4 at 1 stream; d = 2 at 8 streams with 24 rows, d = 3 with 32;
- 8-stream τ ≈ 2.66 per slot (3.34 with 32 rows), from the acceptance upstream's 8-slot run implies;
- exposed serial time is 0.3-0.6 ms per cycle at 1 stream and 0.5-3.3 ms at 8.

the 8-stream range is set by how much per-slot serial time step 2 leaves exposed (upstream's barrier design leaves
~1.8 ms per slot). other cases:
- step 1's ~25 ms assumes upstream's own efficiency (~595 gb/s).
- at 30k tokens per slot, kv reads add ~8 ms to the 8-stream cycle: ~700 aggregate.
- 32 rows at 8 streams, if they fit under the stream, give ~1,100-1,250.
- at the ~80% cycle efficiency prior art reaches, step 2 lands at ~21.5 ms: ~205 code, ~100 prose.

docs/ADVERSARIAL.md holds the arithmetic.

## 1.10 order of work

### phase a: code only (now; no gpu)

**where it runs**: the development mac (darwin arm64, latest llvm and libc++, no cuda sdk). clang compiles every
kernel to ptx for sm_86; host code builds, lints and runs its tests; the baker and the cpu reference run on the real
checkpoint. the `ptxas -v` resource gates, the linux epoll backend and everything touching a real driver run first on
the linux box.

**done means**: `dev`, `release` and `lint` build with zero warnings, the static device gates pass, and the cpu tests
pass. nothing claims runtime kernel correctness until phase c.

**lanes for parallel subagents.** lanes implement and hand over a change that builds. the coordinator merges onto
`main`, runs every check on the merged tree, and sends exact fixes back. commits go straight to `main`.

| lane | work | depends on |
|---|---|---|
| L0 scaffold | AGENTS.md and CLAUDE.md, the cpp-starter 80/20, cmake gate, presets, language profile, clang-tidy profiles, zones, compile-fail harness, PINS.md | — (first) |
| L1 cull | delete everything in 1.1 with no successor yet; move what survives into the zones | L0 |
| L2 model + image + bake | `device/model.h`, `engine.model`, `engine.image`, `engine.bake` | L0 |
| L3 tokenizer + chat | `engine.tokenizer`, `engine.chat`, fixtures | L0 |
| L4 server | `engine.json`, `engine.http`, `engine.openai`, `engine.pipeline`, `engine.sched` + fake-engine simulation | L0, L5 |
| L5 kv + prefix | `engine.kv`, `engine.prefix`, ported fuzz and scenario tests | L0, L2 |
| L6 device | e4m3/bf16 primitives, fp8 gemv/gemm, attention, deltanet, decode step 1 then step 2, prefill, driver-api loader, resident ring, static gates | L0, L2 |
| L7 reference + harness | `geezuz-ref`, rulers, decode-path ruler, determinism tooling | L0, L2, L3 |

### phase b: hardware bring-up

nvlink, power, clocks and thermals (1.7). then the microbenchmarks in `bench/ubench/` (docs/ADVERSARIAL.md lists each
with its expected value and the decision it settles). the first is a plain read kernel per card that must reach ≥ 93%
of the measured memory clock's peak before any kernel tuning. upstream `v0.3.2` q8_0 with `MK_PROFILE=1`, with and
without the bridge, gives the baseline and the real phase breakdown.

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
- **why uncertain**: the bf16 compute ceiling caps rows at 24-32 per cycle, so at 8 streams a tree competes with slots
  for rows (at 1 stream, 5 of 32 rows are in use).
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

- **hypothesis**: spending the 24 or 32 rows where expected accepted tokens per row is highest beats the fixed split,
  e.g. a deep tree for one busy slot, shallow drafts for eight.
- **why uncertain**: the gain depends on acceptance variance across slots and needs good live estimates.
- **kill if**: under +5% aggregate on mixed loads.

### 2.4 row budget and slot count tuning

- **hypothesis**: rows come in n8 tiles, so the row budget is 24 or 32 (40 only if the sustained clock and stream
  allow). 32 gives each of 8 slots a third draft (code τ ≈ 2.66 to 3.34) for one more mtp pass (~0.5 ms): ~+20%
  aggregate if 4 tiles stay under the stream (~128 of ~173 tensor clocks per weight fragment at 90% of p2).
- **`S`**: other than 8 is a separate question. at 32 rows, 16 slots with one draft each give about the same aggregate
  as 8 with three, at half the per-stream speed, and the kv pool (~275k tokens) caps S x context.
- **measure**: cycle time against rows {8, 16, 24, 32, 40} on the hardware, then tok/s per stream and aggregate.
- **ship**: the measured constants. this changes constants, not code paths.

### 2.5 replicated drafter (killed by arithmetic)

holding the whole mtp head on both cards adds the other card's half to every pass: +0.24 gb x d per card per cycle,
~1.2 ms at d = 4. it saves ~2-3 nvlink exchanges per pass at ~1-2 µs each, ~20-50 µs per cycle. break-even needs ~100
µs per exchange. adrien measured a tensor-parallel-1 drafter as null. not built; the nvlink handoff microbenchmark
reopens it only if the handoff exceeds ~100 µs.

### 2.6 reduced-precision drafter

- **hypothesis**: drafter bytes (~1.6 gb per card per cycle at 4 drafts, ~0.8 gb at 2) drop by about half at
  8-bit-int or lower for the fc and draft lm_head, saving ~1 ms per cycle. this sits outside the lossless boundary, so
  output is unchanged.
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
- **why uncertain**: it only matters if the ~275k-token pool evicts under your workload, and it is a lot of code.
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

### 2.12 deferred deltanet write-back

- **hypothesis**: lazy replay already makes this exact. a slot writes its committed state every k cycles and replays k
  cycles of accepted tokens from the ring, through the same recurrence. k = 4 saves 3/4 x 75.5 mb per slot per cycle:
  0.45 gb at 8 slots, ~0.55 ms (~2.5%).
- **why uncertain**: replay adds ~k·τ x 0.2 µs per layer to the core (≲ 0.1 ms per cycle, mostly under the state
  stream), the ring grows, and snapshot points and preemption must force a write.
- **kill if**: the 8-slot cycle isn't ≥ 1.5% faster.

### 2.13 windowed mtp attention

- **hypothesis**: the drafter is outside the lossless boundary, so its attention may see only the last w tokens.
  at 8 slots, d = 2 and 30k context, w = 4k cuts mtp kv reads from 0.98 to 0.13 gb per card, ~1 ms (~3%).
- **why uncertain**: acceptance may fall on long-range copies.
- **kill if**: τ x (1 / cycle) doesn't improve at long contexts.

### 2.14 prefill and decode bit-identical

- **hypothesis**: the prefill gemms walk k in decode's static split order with the same per-k128 scale step, and
  attention uses decode's fixed chunking. then every logit is bit-identical whichever path computed it: prefix hits,
  chunk boundaries and preemption never change output, and the decode-path ruler becomes an exact gpu-vs-gpu check.
- **why uncertain**: fixed split-k partials cost ~2-3% of prefill.
- **kill if**: the cost exceeds 3%, or 2.16 replaces the prefill path (which makes it free).

### 2.15 draft-vocabulary embedding rows in vram

- **hypothesis**: keeping the 32k draft rows on the cards (335.5 mb each) removes a pcie fetch (~2-5 µs) from every
  serial draft step.
- **why uncertain**: that is ≤ ~15 µs per cycle (< 0.1%), against ~10k tokens of kv pool.
- **kill if**: the measured host fetch per draft step is under ~10 µs.

### 2.16 prompt rows in decode cycles

- **hypothesis**: a prefill chunk is a verify whose rows are all accepted, so spare rows of every cycle can carry
  prompt tokens of a prefilling slot, riding a weight stream that is read anyway.
  - one stream decoding at 32 rows leaves 27 spare: ~1,400 tok/s of prefill with the decoding stream unslowed. the
    time-sliced prefill instead stops decode for its share.
  - all 32 rows as prompt rows prefill at ~1,950 tok/s, comparable to the dedicated gemm path. if that holds, the
    dedicated prefill kernels go, and prefill equals decode bit for bit.
- **why uncertain**: the allocation policy (a prompt row is worth one token, a draft row p^k), and attention cost on
  long prompts.
- **measure**: prefill tok/s and decode tok/s with one prompt arriving during 1, 4 and 7 decoding streams, against
  the time-sliced default.
- **kill if**: decode or time-to-first-token gets worse on mixed loads.

---

## 3. risks

- **clang cuda and the latest toolkit**: clang can trail the newest toolkit; use the newest pair it supports.
- **stdexec on clang + libc++**: the module boundary may still need the narrow-abi quarantine.
- **kernels are unverified until phase c**: phase a proves builds, resource budgets and host logic, not kernel
  correctness. the cpu reference exists so phase c catches mistakes on day one.
- **the bf16 compute ceiling**: the lossless contract caps verify width at 24-32 rows per cycle. retune on hardware.
- **per-slot serial time**: the 8-stream estimate assumes step 2 leaves ≤ ~0.35 ms of serial time exposed per slot
  (upstream: ~1.8 ms). the deltanet core and split-k attention at state and kv bandwidth carry it.
- **the p2 memory clock**: cuda may hold the memory at 9501 mhz whatever bring-up tries; the budget already assumes it.
- **memory overclock without ecc**: errors can be silent. accept a clock only after a long golden-hash soak at
  temperature.
- **checkpoint drift**: the image header carries the checkpoint hash; a re-uploaded checkpoint is refused, never
  half-loaded.
