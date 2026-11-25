# HARDWARE: the two rtx 3090s this engine is built for

what the silicon, memory, link and power delivery of this box can do, with a source and a confidence for every
number: **datasheet** (nvidia whitepaper or spec), **documented** (ptx isa, cuda guide, tuning guide), **measured by
others** (microbenchmark papers, credible repos and reports), or **inferred** (derived, derivation shown). nothing
here was measured on this box yet; every open question has a microbenchmark in `bench/ubench/` to run at bring-up.

the box: two geforce rtx 3090 founders edition (ga102, sm_86, 82 SMs, 24 gb gddr6x on a 384-bit bus, 12 of the 24
modules on the back of the board) under corsair xg7 water blocks, a 2-slot nvlink bridge, in PCIE_6 and PCIE_4 of a
gigabyte mz72-hb0 (dual epyc 7v13), linux, bare metal.

## facts reported by the owner

- **nvlink bridge**: the owner bought a 2-slot nvlink bridge and reports that it fits the water-blocked 3090 FEs in
  PCIE_6 / PCIE_4 (confirmed by the owner's pc-build work, 2026-11-25). treat physical clearance as settled; bring-up
  still verifies the links read NV4 in `nvidia-smi topo -m`.

## the SM (hw-sm)

scope: one GA102 SM as found in the rtx 3090 founders edition (sm_86, 82 SMs enabled of 84), what it can issue per clock,
what it can hold, what the PTX ISA lets us say to it, and the cheapest exact e4m3 -> bf16 path into `mma.sync`.

confidence labels: **datasheet** (NVIDIA whitepaper / product spec), **documented** (CUDA guide, PTX ISA, release
notes), **measured by others** (paper or credible public measurement), **inferred** (derivation shown).

source key used below:
- [WP] NVIDIA Ampere GA102 GPU Architecture whitepaper v2,
  https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf (sections "GA10x SM
  Architecture", "2x FP32 Throughput", "Larger and Faster Unified Shared Memory and L1 Data Cache", Appendix A table 9
  "RTX 3090 FE")
- [CG-CC] CUDA Programming Guide (current, CUDA 13.5), appendix "Compute Capabilities", tables 33-35,
  https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- [CG12.6] CUDA C++ Programming Guide 12.6 (archived; last edition that carries the per-CC arithmetic throughput
  table and the 8.x shared-memory text), https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html
  §5.4.1 table 4 "Throughput of Native Arithmetic Instructions", §5.4.3, §16.4 (5.x shared memory banks), §16.7 (8.x)
- [CG-L2] CUDA Programming Guide "L2 Cache Control",
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- [PTX] PTX ISA 9.5, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (per-instruction "PTX ISA
  Notes" / "Target ISA Notes"; section numbers given per row)
- [RN] CUDA Toolkit 13.5 release notes table 2 "CUDA Toolkit and Corresponding Driver Branch",
  https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html ; [MVC] CUDA minor version compatibility,
  https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
- [SUN] Sun et al., "Dissecting Tensor Cores via Microbenchmarks: Latency, Throughput and Numeric Behaviors", IEEE TPDS
  2022, https://arxiv.org/abs/2206.02874 , tables 3 (A100) and 4 (RTX 3070 Ti, GA104 = same SM as GA102)
- [GM] "Generalized Methodology for Determining Numerical Features of Hardware Floating-Point Matrix Multipliers: Part
  I", https://arxiv.org/abs/2510.15884 (RTX 3060, GA106 = sm_86) ; [FASI] Fasi, Higham, Mikaitis, Pranesh, "Numerical
  behavior of NVIDIA tensor cores", PeerJ CS 2021, https://peerj.com/articles/cs-330
- [CNC] Chips and Cheese, "Measuring GPU Memory Latency" (2021, rtx 3090),
  https://old.chipsandcheese.com/2021/04/16/measuring-gpu-memory-latency/
- [TH] Tom's Hardware RTX 3090 FE review spec table, https://www.tomshardware.com/reviews/nvidia-geforce-rtx-3090-review
- [LLVM] LLVM 23.1.0 release notes (NVPTX), https://releases.llvm.org/23.1.0/docs/ReleaseNotes.html ; plus the
  coordinator's measurement of MacPorts clang/llvm 23.1.5 on this project (stated as such below)
- [SIM] my own bit-exact simulation of every conversion sequence below over all 254 finite e4m3 codes and 200k random
  4-code words (python, exact rationals; a one-off script, not kept). the exhaustive host test in `device/prims`
  supersedes it.

### 1. the SM: units, schedulers, limits

| item | value | confidence / source |
|---|---|---|
| SMs per 3090 | 82 (GA102 full die: 84) | datasheet [WP table 9] |
| partitions (SMSPs) per SM | 4; each: 64 KB register file, L0 i-cache, 1 warp scheduler, 1 dispatch unit, 1 tensor core | datasheet [WP "GA10x SM Architecture"] |
| issue | each scheduler issues one instruction per clock for one ready warp; warps statically assigned to schedulers | documented [CG12.6 §16.7.1] |
| FP32 lanes | 2 x 16 per SMSP (one FP32-only datapath, one FP32-or-INT32) -> 128 FP32 FMA/clk/SM, or 64 FP32 + 64 INT32 | datasheet [WP "2x FP32 Throughput"] |
| INT32 lanes | 16 per SMSP (shared with the second FP32 datapath) = 64/clk/SM | datasheet [WP]; documented [CG12.6 table 4] |
| FP64 | 2 units per SM (1/64 FP32) | datasheet [WP]; documented [CG-CC table 33] |
| tensor cores | 4 per SM (3rd gen), 328 per card | datasheet [WP table 9] |
| register file | 256 KB/SM = 64 K x 32-bit; 64 K per block; **255 per thread** | datasheet [WP]; documented [CG-CC table 34] |
| max resident blocks / warps / threads per SM | **16 / 48 / 1536** | documented [CG-CC table 33] |
| max threads per block | 1024 | documented [CG-CC table 33] |
| register allocation granularity | 256 registers per warp (8 per thread) -> e.g. 288 threads/CTA caps at 224 regs/thread, 160 threads at 255 | inferred (occupancy-calculator convention; not restated in the current guide; confirm with `ptxas -v` + `cudaOccupancy*`) |
| unified L1 / shared | 128 KB per SM | datasheet [WP]; documented [CG-CC table 35] |
| shared-memory carveouts | 0, 8, 16, 32, 64, **100 KB** (L1 gets the rest: 128/120/112/96/64/**28 KB**) | datasheet [WP]; documented [CG-CC table 35, CG12.6 §16.7.3] |
| max shared per SM / per block | **100 KB / 99 KB** (1 KB per resident block reserved for system use; > 48 KB needs dynamic smem + `cudaFuncAttributeMaxDynamicSharedMemorySize` opt-in) | documented [CG-CC table 34, CG12.6 §16.7.3] |
| carveout API | `cudaFuncAttributePreferredSharedMemoryCarveout` is a hint, rounded up to the next supported size | documented [CG12.6 §16.7.3] |
| shared-memory banks | 32 banks, successive 32-bit words -> successive banks, 32 bits per bank per clock | documented [CG12.6 §16.4 (5.x rules, inherited by 8.x per §16.7)] |
| shared-memory bandwidth | **128 B/clk/SM** (2x Turing) | datasheet [WP "Larger and Faster ... L1"] |
| L2 | **6144 KB** on the 3090 (12 x 512 KB, one slice per 32-bit memory controller, 384-bit bus) | datasheet [WP "memory subsystem", table 9] |
| L1 -> L2 latency | "over 100 ns" (pointer chase) | measured by others [CNC] |
| DRAM latency (unloaded) | roughly the same as RDNA2's ~226 ns figure in the same test; no exact ampere number printed | measured by others [CNC] (low precision) |
| clocks, 3090 FE | base **1395 MHz**, boost **1695 MHz** (rated; GPU Boost runs above this when power/thermals allow, which water cooling helps) | datasheet [WP table 9 for boost]; documented [TH spec table for base]; actual sustained clock: not established, measure |
| memory | 24 GB GDDR6X, 19.5 Gbps, 384-bit, 936 GB/s | datasheet [WP table 9] |
| sm_86 vs sm_80 ISA delta | the only addition: `.xorsign.abs` on `min`/`max` | documented [PTX §5 target table: "sm_86 Adds support for .xorsign modifier on min and max"] |

### 2. issue rates per SM per clock (non-tensor)

all from [CG12.6 table 4], compute capability 8.6 column, "results per clock per multiprocessor" (documented). the
right-hand column converts to warp-instructions per clock per SMSP (inferred: results / 32 / 4).

| class | results/clk/SM | warp-instr/clk/SMSP |
|---|---|---|
| fp32 add/mul/fma | 128 | 1 |
| fp16 add/mul/fma (half2 counts 2) | 256 | 1 (as half2) |
| **bf16 add/mul/fma** | **128** (footnote: "128 for __nv_bfloat16") | **0.5 as bf16x2** |
| fp64 | 2 | 1/64 |
| rcp/rsqrt/lg2/ex2/sin/cos (MUFU) | 16 | 1/8 |
| int32 add/sub (IADD3) | 64 | 0.5 |
| int32 mul/mad (IMAD) | 64 (32 for extended precision) | 0.5 |
| shift (SHF) | 64 | 0.5 |
| and/or/xor (LOP3) | 64 | 0.5 |
| compare/min/max | 64 | 0.5 |
| bfe/bfi | **multiple instructions** on 8.6 (native only from 8.9) | avoid; use `shf`/`lop3`/`prmt` |
| popc, clz, brev | 16 | 1/8 |
| warp shuffle | 32 | 1/4 |
| warp reduce (`redux.sync`) | 16 | 1/8 |
| warp vote | 64 | 1/2 |
| type conversions (F2F, I2F, `cvt.rn.bf16x2.f32`, ...) | 16 | 1/8 |
| 8/16-bit int -> 32-bit int | 64 | 1/2 |
| `__syncthreads()` | 16 ops/clk/SM | documented [CG12.6 §5.4.3] |
| `prmt` | not listed | inferred 64 (integer ALU class); measure |

notes:
- [WP] states peak fp16 non-tensor = peak fp32 = 35.6 TFLOPS on the 3090, i.e. 128 fp16 FMA/clk/SM, which disagrees
  with the guide's 256 for fp16; for bf16 both sources agree (128/clk/SM, 35.6 TFLOPS). we only need bf16. (documented,
  conflict noted)
- [WP] says "standard FP16 operations are handled by the Tensor Cores in GA10x GPUs". if `fma.rn.bf16x2` (HFMA2.BF16)
  issues to the tensor pipe it competes with `mma.sync` for the same unit. not established; measure in phase b (it
  matters only at the tensor-bound end, see §6).
- IMAD issuing to the FMA pipe (so `shl` lowered as `IMAD.SHL` balances against LOP3 on the ALU pipe) is common sass
  behaviour on sm_70+; inferred, not documented.

### 3. tensor cores: rates and numerics

**rates (dense, per card, at rated boost 1695 MHz)** — datasheet [WP table 9, "Peak rates are based on GPU Boost
Clock"]:

| mma flavour | TFLOPS | flop/clk/SM (inferred: TFLOPS / (82 x 1.695 GHz)) |
|---|---|---|
| fp16 in, fp16 accumulate | 142 (284 sparse) | 1024 |
| fp16 in, **fp32 accumulate** | **71** (142 sparse) | 512 |
| **bf16 in, fp32 accumulate** | **71** (142 sparse) | 512 |
| tf32 | 35.6 | 256 |
| int8 | 284 TOPS | 2048 |

confirmed: GeForce GA102 halves the fp32-accumulate tensor rate (the A100 runs fp16/fp32 at the same rate). bf16 has
no fp16-accumulate form, so **71 TFLOPS is the ceiling for every lossless matmul on this card** (58.6 at base 1395 MHz,
~79.8 if it sustains 1.9 GHz — inferred, linear in clock).

**per instruction** (inferred from the above): `mma.sync.m16n8k16` = 16·8·16·2 = 4096 flop; at 512 flop/clk/SM =
128/clk/SMSP, one m16n8k16 bf16/fp32 occupies its SMSP's tensor core for **32 clocks** (8 clocks SM-wide).
`m16n8k8` = 16 clocks.

**measured by others** — [SUN table 4], RTX 3070 Ti (GA104, same SM design):

| A/B, C/D, shape | completion latency (clk) | peak throughput (FMA/clk/SM) |
|---|---|---|
| fp16, fp32, m16n8k16 | 33 | 252.7 (= 505 flop/clk/SM, matches 512) |
| fp16, fp32, m16n8k8 | 18.8 | 253.2 |
| fp16, fp16, m16n8k16 | 24 | 509.4 |
| tf32, fp32, m16n8k8 | 33.3 | 126.8 |
| int8, int32, m16n8k32 | 24.3 | 1023 |

on the A100 the same fp16/fp32 m16n8k16 runs at ~1000 FMA/clk/SM [SUN table 3]: the GeForce halving is measured, not
just a datasheet line. bf16 was not in the 3070 Ti table; bf16/fp32 shares the fp16/fp32 rate per [WP]. peak needs
>= 2 warps per SMSP or ILP >= 2 per warp [SUN §4]: with 33-clk latency and 32-clk issue interval, one warp with one
dependent chain cannot keep a tensor core busy.

**numerics (contract-relevant)** — measured by others [GM §IV, table I] on an RTX 3060 (sm_86), reported identical to
A100 [FASI]: with fp32 output the tensor core sums products in **blocks of 8 (fp16/bf16 inputs) with truncation (RZ),
one extra alignment bit (25-bit significand datapath) and >= 3 carry bits**; subnormal inputs and outputs are supported.
so "fp32 accumulation" on sm_86 is not IEEE fp32 RN accumulation: each 8-term block is aligned to its largest term and
truncated. this is a property of every tensor-core path on this card, so the contract's "only summation order differs"
must be read as "summation order and tensor-core block rounding" — see implications.

**what does not exist on sm_86** (documented [PTX target notes, §9.7.16.5.14 and others]): fp8 (`e4m3`/`e5m2`) mma
(sm_89+), `cvt` from/to `e4m3x2` (sm_89+; the bf16x2 forms are later still), `wgmma` (sm_90a), TMA /
`cp.async.bulk*` (sm_90), clusters and distributed shared memory (sm_90), `stmatrix` (sm_90), `setmaxnreg` (sm_90a),
`elect.sync` (sm_90), `griddepcontrol` / programmatic dependent launch (sm_90), `mbarrier.try_wait` and `expect_tx`
(sm_90), `fence.proxy.async` (sm_90), float `redux.sync` (sm_100a), `mul.bf16x2`/`add.bf16x2` (sm_90; `fma.bf16x2` is
sm_80), `red`/`atom` `.add.bf16x2` (sm_90).

### 4. memory-side knobs: L1, L2, policies

| knob | sm_86 status | source |
|---|---|---|
| L1 bypass for streamed weights | `cp.async.cg` (caches at L2 only), `ld.global.nc.L1::no_allocate` | documented [PTX §9.7.10.28.3.1, §9.7.10.9] |
| per-access L2 eviction priority | `createpolicy.fractional.L2::{evict_first,evict_last,evict_normal,evict_unchanged}.b64 pol, frac` or `createpolicy.range...`, then `.L2::cache_hint` on `ld`/`st`/`cp.async`/`atom`/`red`/`prefetch` | documented [PTX §9.7.10.21] |
| L2 prefetch with priority | `prefetch.global.L2::evict_last [a]` / `::evict_normal` (eviction-priority form sm_80) | documented [PTX §9.7.10.16] |
| L2 sector prefetch on loads | `.L2::64B/.L2::128B/.L2::256B` on `ld`, `cp.async` (256B needs sm_80) | documented [PTX §9.7.10.8] |
| reset a line's priority | `applypriority.global.L2::evict_normal [a], 128` | documented [PTX §9.7.10.17] |
| drop dirty lines without writeback | `discard.global.L2 [a], 128` (128-B aligned, size must be 128) | documented [PTX §9.7.10.20] |
| persisting set-aside (host API) | `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, ...)` capped by `prop.persistingL2CacheMaxSize`; stream/graph `accessPolicyWindow` with `hitRatio`, `hitProp`/`missProp` (persisting/streaming/normal); disabled under MIG | documented [CG-L2 §4.15.1-2]; applies to cc 8.6 per NVIDIA forum reply (documented-ish) |
| 3090 `persistingL2CacheMaxSize` | not found published. inferred **~4.5 MB** (= 75% of 6 MB, the ratio seen on A100 40 MB -> 30 MB and on a 4 MB part -> 3 MB); `accessPolicyMaxWindowSize` likely ~128 MB. query `cudaDevAttrMaxPersistingL2CacheSize` on the box | inferred |
| L2 topology | 12 slices x 512 KB, one per 32-bit controller; one crossbar (no A100-style two-partition split described for GA102) | datasheet [WP]; "single partition" inferred |

### 5. instruction availability: the brief's confirm/refute list

| claim | verdict | source |
|---|---|---|
| `cp.async` on sm_86 | **confirmed**: `.ca` sizes 4/8/16, `.cg` size 16 only; optional `src-size` (zero-fill) or `ignore-src`; `.L2::cache_hint` + policy operand; `.L2::64B/128B/256B` prefetch hint; sm_80, PTX 7.0 (hints 7.4) | documented [PTX §9.7.10.28.3.1] |
| `cp.async.commit_group` / `wait_group N` / `wait_all` | **confirmed**, sm_80, PTX 7.0 | documented [PTX §9.7.10.28.3.2] |
| `cp.async.mbarrier.arrive{.noinc}` | **confirmed**, sm_80, PTX 7.0 | documented [PTX §9.7.15.16.18] |
| `mbarrier` init/inval/arrive/arrive_drop/test_wait/pending_count | **confirmed**, sm_80, PTX 7.0; `.parity` variant of `test_wait` PTX 7.1 | documented [PTX §9.7.15.16.12-20] |
| `mbarrier.try_wait` | **refuted for sm_86**: "mbarrier.try_wait requires sm_90 or higher" | documented [PTX §9.7.15.16.19] |
| `mbarrier.arrive.expect_tx` / `mbarrier.expect_tx` | **refuted for sm_86**: sm_90 | documented [PTX §9.7.15.16.14, .16] |
| `mbarrier.arrive` with a `count` (without `.noComplete`) | **refuted for sm_86**: sm_90; on sm_86 only `arrive.noComplete` takes a count | documented [PTX §9.7.15.16.16] |
| `ldmatrix` | **confirmed**: `.m8n8 .x1/.x2/.x4 {.trans} .b16`, sm_75, PTX 6.5; `.b8`, `.m16n16`, `.m8n16` forms are sm_100a/120a only | documented [PTX §9.7.16.5.15] |
| `mma.sync` bf16 | **confirmed**: `m16n8k8` and `m16n8k16` `.f32.bf16.bf16.f32`, sm_80, PTX 7.0; f16 `m16n8k16` sm_80, `m16n8k8` sm_75 | documented [PTX §9.7.16.5.14] |
| fp32-accumulate rate halved on GeForce GA102 | **confirmed**: 71 vs 142 TFLOPS; measured 253 vs 509 FMA/clk/SM on GA104 | datasheet [WP]; measured [SUN] |
| no TMA, no wgmma, no clusters | **confirmed** (all sm_90+) | documented [PTX target notes] |
| no hardware e4m3 conversion | **confirmed**: `cvt ... e4m3x2` requires sm_89 | documented [PTX §9.7.10.24] |
| fp32 FMA 128/clk/SM | **confirmed** | datasheet [WP]; documented [CG12.6 table 4] |
| shared memory 128 B/clk/SM | **confirmed** | datasheet [WP] |
| 255 registers/thread | **confirmed** | documented [CG-CC table 34] |

### 6. the cheapest exact e4m3 -> bf16 for `mma.sync`

**the identity.** e4m3fn: sign s, 4-bit exponent e (bias 7), 3-bit mantissa m; e=0 is subnormal (m/8 · 2^-6); only
0x7f/0xff are NaN; no inf. bf16: sign, 8-bit exponent (bias 127), 7-bit mantissa. place the 7 payload bits `eeeemmm`
at bf16 bits 10:4 and the sign at bit 15:

    pre(b) = (b & 0x80) << 8  |  (b & 0x7f) << 4

for normals the bf16 exponent field is e, so pre = 2^(e-127)(1+m/8) = 2^-120 · value. for e=0 the bf16 field is also 0,
so pre is a bf16 *subnormal* 2^-126 · (16m/128) = 2^-120 · (m/8 · 2^-6): the same factor. so **pre(b) = 2^-120 · x(b)
exactly for all 254 finite codes, subnormals and ±0 included** (verified [SIM]). every e4m3 value is exactly
representable in bf16 (exponents -9..8 fit; 3 mantissa bits fit in 7), so

    x = fma.rn.bf16x2(pre, 2^120, -0.0)      // 2^120 = bf16 0x7B80; addend -0.0 keeps -0 as -0

is exact: the product is exact and representable, so RN does nothing; `fma.bf16x2` supports subnormal inputs by default
(documented [PTX §9.7.4.4]: "By default, subnormal numbers are supported"; `.ftz` does not exist for bf16). there is no
`mul.bf16x2` on sm_86 (sm_90), hence the fma with -0.

NaN codes map to ±480 (finite) under this sequence [SIM]. a weight NaN is not part of any real checkpoint; **the bake
must reject 0x7f/0xff** with a typed error so the device never needs a NaN path.

**rejected shortcut: folding 2^120 into the fp32 block scale.** feeding `pre` (2^-120-scaled weights) straight into the
mma and multiplying 2^120 into the block scale is exact only while every tensor-core output stays in the normal fp32
range. given the measured tensor-core model (align to the block's largest term, truncate to a 25-bit window, subnormal
outputs supported [GM]) the scaled computation is bit-identical *except* when a partial sum falls below 2^-126, i.e.
when |true partial| < 2^-6 in (activation x raw e4m3) units, which happens with small activations or cancellation.
there it loses bits. that is a numerics change beyond summation order, so it is out under the contract. keep the fma:
it costs 0.5 instr/element (below).

**sequence A: natural checkpoint byte order** (4 codes per 32-bit word w = [b3 b2 b1 b0]):

    // hi = (b3:b1) pre-images, lo = (b2:b0)
    shr.u32  x, w, 4;            and.b32 x, x, 0x07F007F0;
    lop3.b32 hi, w, 0x80008000, x, 0xEA;          // (w & K) | x
    shl.b32  y, w, 4;            and.b32 y, y, 0x07F007F0;
    shl.b32  z, w, 8;
    lop3.b32 lo, z, 0x80008000, y, 0xEA;
    fma.rn.bf16x2 hi, hi, c2p120, cneg0;  fma.rn.bf16x2 lo, lo, c2p120, cneg0;

7 integer + 2 bf16 fma = **9 instructions per 4 elements (2.25/element)**. a single `lop3` cannot do the mask-and-merge
for one output, because each output needs three bit classes (from w, from the shifted copy, zero) and `lop3`'s constant
operand only splits positions into two (proof sketch: at bit 11 the shifted copy carries a sign bit that must become 0,
at bit 15 the same two inputs must pass the sign through; both are class-0 positions under any one mask).

**sequence B: bake-permuted bits (recommended).** the bake may store any bijective re-encoding of the 32 bits of a
4-code word (lossless; only the device decode changes). choose: pair A at its final positions, pair B's 16 bits in the
16 free positions so that a left shift by 4 places 10 of them and a right shift by 4 places the other 6:

    MS = 0x87F087F0   M1 = 0x080F080F   K = 0x07000700
    and.b32  a, w, MS;                 // pair A: already in place
    and.b32  p, w, M1;   shl.b32 p, p, 4;   // 10 bits of pair B, now sparse at their targets
    shr.u32  q, w, 4;                  // the other 6 bits land at 0x07000700
    lop3.b32 b, q, p, K, 0xE4;         // b = K ? q : p   (p is zero outside its targets)
    fma.rn.bf16x2 a, a, c2p120, cneg0; fma.rn.bf16x2 b, b, c2p120, cneg0;

the bake encoder: bits of pre_A at MS stay; bit p of pre_B goes to p+4 for p in {8..10, 24..26}, else to p-4. covers all
32 bits, bijective, verified over 200k random words and all codes [SIM]. **5 integer + 2 bf16 fma = 7 instructions per 4
elements (1.75/element, 3.5 per bf16x2 pair).** i did not find a 4-integer-op form; the argument above shows every
output needs a sparse operand plus a merge, and pair B needs two different shifts (S's 7-bit runs cannot be one rigid
shift of the free positions), so 5 looks minimal for `lop3`/`shf`/`prmt`/`mad` sequences, but that's not proven.

**fragment layout.** for `mma.m16n8k16.row.col` A (weights, 16 output rows x 16 k), lane l holds rows g=l/4 and g+8,
k pairs 2t,2t+1 and 2t+8,2t+9 (t=l%4) as 4 bf16x2 registers [PTX §9.7.16.5.8]. one lane's fp8 for one k16 step is 8
bytes = 2 words: word 0 -> (a0a1 row g, a2a3 row g+8), word 1 -> (a4a5, a6a7). the bake writes each lane's 16 bytes
for two consecutive k16 steps contiguously, already bit-permuted. a warp then reads 512 contiguous bytes with one
`ld.shared.v4.b32` (or `cp.async` 16 B / `ld.global.nc.v4`): 4 conflict-free 128-B wavefronts, **no `ldmatrix` for
weights**. `ldmatrix.x4(.trans)` remains the right load for bf16 activation B fragments unless their producer writes
them in fragment order too.

**cost against the mma it feeds** (inferred; per SMSP at rated boost 1.695 GHz; 936 GB/s split over 82 SMs = 6.73
B/clk/SM = 1.68 B/clk/SMSP, so one 256-B A fragment arrives every **152 clocks**; 125 at base clock, 170 at 1.9 GHz):

| per 256-B weight fragment (per warp) | instructions | pipe clocks (SMSP) |
|---|---|---|
| integer ops, sequence B (2 words x 5) | 10 | 20 (ALU, 16 lanes) |
| integer ops, sequence A (2 words x 7) | 14 | 28 |
| `fma.rn.bf16x2` (4 regs) | 4 | 8 (0.5/clk/SMSP; FMA or tensor pipe, unknown) |
| `mma.m16n8k16`, R rows padded to 8 | ceil(R/8) | 32 per n8 tile |
| shared/global load of the fragment | 0.5-1 | — |
| issue slots used (R=24, sequence B) | ~19 | of 152 available (~13%) |

so conversion is **not** the bottleneck in decode at any R: the tensor core is (ceil(R/8)·32 clk vs 152). R=24 -> 96 clk
(63% tensor busy); R=32 -> 128 clk (84%, plus 8 clk if HFMA2 shares the tensor pipe = 89%); R=40 -> 160 > 152, so the
DRAM/tensor crossover is **32 rows at 1695 MHz** and stays 32 at 1.9 GHz (4 tiles = 128 <= 170). in prefill, with an A
fragment reused across >= 16 n8 tiles (128 tokens), conversion is <= 5% of tensor time.

### ptx feature list for sm_86

rows give: instruction (and the forms usable here), minimum PTX ISA, minimum sm, semantics in one line, cost where
known. "—" in the cost column means no number found; all ISA/sm columns are documented [PTX], section in the first
column. sm_86 rows that say sm_90 are listed because the brief asked; they are **not** available.

| instruction | PTX ISA | sm | semantics | cost (source, confidence) |
|---|---|---|---|---|
| `cp.async.ca.shared.global [d],[s],{4,8,16}{,src-size\|ignore-src}` §9.7.10.28.3.1 | 7.0 (`src-size` 7.0, `ignore-src` 7.5) | 80 | async global->shared copy, L1+L2 allocate; zero-fill tail | — |
| `cp.async.cg.shared.global [d],[s],16` | 7.0 | 80 | as above, L2 only (bypass L1); only 16 B | one warp-instr moves 512 B (inferred) |
| `cp.async... .L2::cache_hint [..], pol` | 7.4 | 80 | per-access L2 eviction policy from `createpolicy` | hint only |
| `cp.async... .L2::64B/.L2::128B/.L2::256B` | 7.4 | 80 | L2 prefetch-size hint | hint only |
| `cp.async.commit_group` / `cp.async.wait_group N` / `cp.async.wait_all` §9.7.10.28.3.2 | 7.0 | 80 | per-thread groups; wait until <= N groups pending | — |
| `cp.async.mbarrier.arrive{.noinc}.shared.b64 [mb]` §9.7.15.16.18 | 7.0 | 80 | arrive-on the mbarrier when this thread's prior cp.asyncs complete; `.noinc` = arrival is pre-counted in the init count | — |
| `mbarrier.init.shared.b64 [mb], count` §9.7.15.16.12 | 7.0 | 80 | 8-byte object in shared memory; `.layout` qualifier sm_90 | — |
| `mbarrier.inval` §.13 | 7.0 | 80 | invalidate before reuse of the memory | — |
| `mbarrier.arrive{.release.cta}.shared.b64 st, [mb]` §.16 | 7.0 (`.sem/.scope` spelled out 8.0; `_` sink 7.1) | 80 | arrive, returns phase state; default `.release.cta` | — |
| `mbarrier.arrive.noComplete.shared.b64 st,[mb],count` | 7.0 | 80 | arrive with count, must not complete the phase | — |
| `mbarrier.arrive` with plain `count`, `.expect_tx`, `.relaxed`, `.cluster`, `::cluster` | 7.8 / 8.0 / 8.6 | **90** | not on sm_86 | — |
| `mbarrier.arrive_drop` §.17 | 7.0 | 80 | arrive and drop this thread from future phases | — |
| `mbarrier.test_wait{.parity}.shared.b64 p, [mb], st\|par` §.19 | 7.0 (`.parity` 7.1) | 80 | **non-blocking** test of phase completion; default `.acquire.cta`. the sm_86 wait loop is a `test_wait.parity` spin, optionally with `nanosleep` backoff | per-test cost not measured on sm_86; measure |
| `mbarrier.try_wait{.parity}` | 7.8 | **90** | potentially blocking wait with time hint — **not on sm_86** | — |
| `mbarrier.expect_tx` / `complete_tx` | 8.0 | **90** | tx-count (TMA) — **not on sm_86** | — |
| `mbarrier.pending_count` §.20 | 7.0 | 80 | pending arrivals from a state value | — |
| `bar.sync a{,b}` = `barrier.sync.aligned` §9.7.15.1 | 1.0 / thread count 2.0 / `barrier` 6.0 / `.cta` 7.8 | 20 (`barrier` 30) | named barrier 0..15, b = participating threads (multiple of 32); waits | `__syncthreads` 16 ops/clk/SM (documented [CG12.6 §5.4.3]) |
| `bar.arrive a, b` = `barrier.arrive.aligned` | 2.0 | 20 | arrive without waiting — producer/consumer handoff on a named barrier | as above |
| `barrier.sync` (non-`.aligned`) | 6.0 | 30 (sm_70+ semantics) | divergence-tolerant form; on sm_6x it equals `.aligned` | — |
| `bar.red.{popc,and,or}` | 2.0 | 20 | barrier with predicate reduction | — |
| `bar.warp.sync mask` | 6.0 | 30 | warp-level convergence barrier (`__syncwarp`) | — |
| `createpolicy.fractional.L2::{evict_last,evict_normal,evict_first,evict_unchanged}{.L2::evict_first\|unchanged}.b64 pol{, f}` §9.7.10.21 | 7.4 | 80 | opaque 64-bit policy: fraction f of accesses get the primary priority | 1 instr, hoist out of loops |
| `createpolicy.range.global.L2::...b64 pol, [a], primary, total` | 7.4 | 80 | priority by address range (total <= 4 GB) | — |
| `createpolicy.cvt.L2.b64 pol, prop` | 7.4 | 80 | convert a CUDA access-property | — |
| `ld{.global}{.L1::evict_*\|.L1::no_allocate}{.L2::cache_hint}{.L2::64B\|128B\|256B}.v4.b32` §9.7.10.8 | 7.4 | 80 (L1 priority 70, 128B 75, 256B 80) | weak load with cache hints; `.level2::eviction_priority` form is sm_100 | — |
| `ld.global.nc...` §9.7.10.9 | 3.1 (hints 7.4) | 32 (hints as above) | non-coherent read-only path; right for weights | — |
| `st{...}.L2::cache_hint` §9.7.10.11 | 7.4 | 80 | store with L2 policy | — |
| `prefetch.global.L2 [a]` / `prefetch.global.L2::evict_last [a]` / `::evict_normal` §9.7.10.16 | 2.0 / 7.4 | 20 / 80 | bring a line into L2, optionally with priority; `.L1::32B`/`.valid_addr` are sm_90 | — |
| `discard.global.L2 [a], 128` §9.7.10.20 | 7.4 | 80 | drop a 128-B line without writeback (data becomes indeterminate) | — |
| `applypriority.global.L2::evict_normal [a], 128` §9.7.10.17 | 7.4 | 80 | demote evict_last lines back to normal | — |
| `ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}{.shared}.b16` §9.7.16.5.15 | 6.5 | 75 | 1/2/4 8x8 b16 matrices from shared into mma fragment layout; `.b8`, `.m16n16`, `.m8n16` sm_100a/120a | — |
| `stmatrix` §9.7.16.5.16 | 7.8 | **90** | **not on sm_86** | — |
| `movmatrix.sync.aligned.m8n8.trans.b16` §9.7.16.5.17 | 7.8 | 75 | in-register 8x8 transpose of a fragment | — |
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` §9.7.16.5.14 | 7.0 | 80 | D = A·B + C, A 16x16 bf16 (4 regs), B 16x8 (2 regs), C/D 4 fp32 | 32 clk/SMSP issue interval (inferred from [WP]); latency 33 clk (measured [SUN], fp16 in) |
| `mma...m16n8k8.row.col.f32.bf16.bf16.f32` | 7.0 | 80 | half-K form | 16 clk; latency 18.8 [SUN] |
| `mma...m16n8k16.row.col.f32.f16.f16.f32` | 7.0 | 80 | fp16 inputs, fp32 accumulate (same rate as bf16) | as bf16 |
| `mma...m16n8k16 .f16.f16.f16.f16` | 7.0 | 80 | fp16 accumulate, 2x rate — **excluded by the contract** | 16 clk, latency 24 [SUN] |
| `mma... .e4m3/.e5m2` | 8.4 | **89** | not on sm_86 | — |
| `redux.sync.{add,min,max}.{u32,s32}` / `.{and,or,xor}.b32` §9.7.15.13 | 7.0 | 80 | warp reduction, **integer only** (`.f32` needs sm_100a) | 16/clk/SM [CG12.6] |
| `shfl.sync.{up,down,bfly,idx}.b32 d{\|p}, a, b, c, mask` §9.7.10.6 | 6.0 | 30 | warp shuffle | 32/clk/SM [CG12.6] |
| `vote.sync.{all,any,uni,ballot}` §9.7.15.10 | 6.0 | 30 | warp vote | 64/clk/SM [CG12.6] |
| `match.sync.{any,all}.b32/.b64` §9.7.15.11 | 6.0 | 70 | lanes with equal values | — |
| `ld.acquire.{cta,gpu,sys}` / `st.release.{cta,gpu,sys}` / `ld.relaxed`/`st.relaxed` | 6.0 | 70 | scoped acquire/release; `.sys` reaches peer (nvlink-mapped) memory; `.cluster` is sm_90 | — |
| `red.release.gpu.global.add.u32` / `atom.{acq_rel,...}.{gpu,sys}` §9.7.15.5-6 | 6.0 (sem), 5.0 (scope) | 70 | scoped atomics; `.L2::cache_hint` sm_80; bf16/vector forms sm_90 | — |
| `fence.{sc,acq_rel}.{cta,gpu,sys}` / `membar.{cta,gl,sys}` §9.7.15.4 | 6.0 / 1.4 | 70 / all | memory fences; `fence.acquire`/`.release` (8.6) are sm_90 | — |
| `fence.proxy.{alias,...}` | 7.5 | 70 | generic-proxy forms only | — |
| `fence.proxy.async` | 8.0 | **90** | not needed on sm_86: `cp.async` is a generic-proxy weak op (PTX §9.7.10.28.3.1) | — |
| `nanosleep.u32 t` §9.7.21.2 | 6.3 | 70 | sleep ~t ns, actual in [0, 2t], max 1 ms; a warp's sleepers wake together | — |
| `prmt.b32 d, a, b, c` §9.7.10.7 | 2.0 | 20 | byte permute of 8 source bytes (+ sign-replicate modes) | not in the throughput table; inferred 64/clk/SM |
| `lop3.b32 d, a, b, c, lut` §9.7.9.6 | 4.3 | 50 | any 3-input bitwise function | 64/clk/SM [CG12.6 "bitwise"] |
| `shf.{l,r}.{wrap,clamp}.b32` §9.7.9.7 | 3.1 | 32 | funnel shift (rotate when a=b) | 64/clk/SM [CG12.6 "shift"] |
| `bfe`/`bfi` §9.7.1.20-21 | 2.0 | 20 | bit-field extract/insert | **multiple instructions on 8.6** [CG12.6]; avoid |
| `fma.rn.bf16x2 d, a, b, c` §9.7.4.4 | 7.0 | 80 | packed bf16 fma, subnormals supported, no `.ftz` | 128 results/clk/SM = 0.5 warp-instr/clk/SMSP [CG12.6] |
| `mul.bf16x2` / `add.bf16x2` | 7.8 | **90** | not on sm_86 (use fma with -0 / 1.0) | — |
| `cvt.rn.bf16x2.f32 d, hi, lo` §9.7.10.24 | 7.0 | 80 | pack two fp32 to bf16x2 RN (the activation write-out) | 16/clk/SM [CG12.6 "conversions"] |
| `cvt.f32.bf16` | 7.1 | 80 | widen (or just `shl 16` / `prmt`) | 16/clk/SM vs 64 for the integer form |
| `cvt.rn.f16x2.e4m3x2` | 7.8 / 8.1 | **89** | not on sm_86 | — |
| `min/max.xorsign.abs` | 7.2 | **86** | the one sm_86-only addition | — |
| `elect.sync` | 8.0 | **90** | not on sm_86 (use `laneid == 0`) | — |
| `setmaxnreg` | 8.0 | **90a** | not on sm_86: producer warps cannot donate registers | — |
| `griddepcontrol` | 7.8 | **90** | not on sm_86: no programmatic dependent launch | — |

**toolchain: newest PTX for sm_86**
- LLVM 23.1.0 release notes list one NVPTX change: default sm moved from sm_30 to sm_75 [LLVM] (documented). they say
  nothing about PTX versions.
- measured by the coordinator on this project: MacPorts clang/llvm **23.1.5** emits **`.version 9.3`** for `.target
  sm_86` with `+ptx93`; llc's feature list runs ptx32..ptx93. clang's front end gates the `__nvvm_*_sync` builtins on
  an enumerated feature list that ends at ptx92, so **those builtins are rejected under +ptx93**. inline PTX works.
  consequence: either pin `+ptx92` for builtins or (the plan's choice) write the sync primitives as inline PTX, which
  the device rules already require.
- `.target sm_86` with PTX 9.3: allowed. the target string has existed since PTX 7.1 [PTX "sm_86 introduced in PTX ISA
  version 7.1"] and CUDA 13.5 deprecates no 8.x architectures [RN §2.6.2] (documented).
- **minimum driver to JIT PTX 9.3: the R610 branch** (CUDA 13.3) or newer. [RN table 2] maps CUDA 13.3 -> R610, 13.4
  -> R615 and 13.5 -> R620, and CUDA 13.5 ships PTX ISA 9.5 [RN §2.3.3] (documented). that PTX 9.3 came with CUDA 13.3
  is inferred from the one-to-one cadence (PTX 9.0 / CUDA 13.0 ... PTX 9.5 / CUDA 13.5). minor-version compatibility
  does **not** cover PTX: "Applications that compile device code to PTX will not work on older drivers" [MVC]
  (documented). safest: ship sass (`ptxas -arch=sm_86` at build time) plus the PTX. then an older driver can still run
  the cubin, and PTX JIT needs >= R610.

### 7. synchronisation on sm_86: what exists, cheapest first

- **intra-warp**: `shfl.sync` (32/clk/SM), `vote.sync`, `redux.sync` (integer only, 16/clk/SM), `bar.warp.sync`. fp32
  reductions use a `shfl.bfly` tree (5 steps).
- **intra-CTA, fixed partners**: named barriers `bar.arrive id, n` (producer) / `bar.sync id, n` (consumer). they are
  hardware barriers, 16 ids per CTA, with thread counts in multiples of 32 (documented). a full/empty pair per ring slot
  caps the ring at about 7 slots with 2 ids spare. the documented throughput is `__syncthreads` at 16/clk/SM; arrive
  latency is not published for sm_86.
- **intra-CTA, many slots / async completion**: `mbarrier` in shared memory. the sm_86 pattern: producer `cp.async` ->
  `cp.async.mbarrier.arrive.noinc [full_s]`, so the copy's own completion arrives and the producer never runs
  `wait_group`. consumers spin on `mbarrier.test_wait.parity [full_s]` (no `try_wait` on sm_86), with `nanosleep` only
  if the spin steals issue slots. after use they run `mbarrier.arrive [empty_s]`, and the producer `test_wait.parity`s
  `empty_s` before refilling. any number of slots works. the per-operation cost on sm_86 is not published; measure in
  phase b next to named barriers.
- **cross-CTA / cross-SM**: counters in global memory. `red.release.gpu.global.add.u32` to publish, `ld.acquire.gpu`
  polling (optionally with `nanosleep`) to consume. both hit L2 (>100 ns away [CNC]), so each hop costs several hundred
  ns.
- **cross-card (nvlink peer memory)**: the same with `.sys` scope (`st.release.sys`/`red.release.sys`,
  `ld.acquire.sys`).
- **not available**: clusters/DSMEM barriers, `try_wait`, tx-count barriers, `griddepcontrol`, `elect.sync`,
  `fence.proxy.async` (and not needed: no async proxy on sm_86).

### implications for the plan

1. **compute ceiling and row budget.** 71 TFLOPS dense bf16/fp32-accumulate at the rated 1695 MHz is confirmed twice
   (datasheet; measured 253 FMA/clk/SM on GA104). it scales with the real sustained clock, so measure that on the water
   loop. per SMSP, an `m16n8k16` costs 32 clocks and a 256-B weight fragment arrives every ~152 clocks, so **rows are
   bought in units of 8** (the n8 tile; R not a multiple of 8 wastes mma work) and **the DRAM/tensor crossover is 32
   rows**. the plan's "~32 rows" crossover holds, and its budget of 24 = 3 n-tiles is a sound default (63% tensor busy).
   tune only through {24, 32}.
2. **lossless contract wording.** on sm_86, fp32-output tensor-core accumulation truncates (RZ) within 8-term blocks
   with a 25-bit alignment window (measured [GM], same as A100). that is more than "summation order". section 0.2 and
   the verification harness (1.8) should define the reference as "exact products, fp32 accumulation with tensor-core
   block truncation", or compare against a bit-exact mma model, not against IEEE fp32 RN sums. the 2^-120
   "fold into the scale" shortcut is **out**: it loses bits when partial sums are tiny.
3. **conversion placement: in registers, from a bake-permuted encoding.** decode with sequence B (5 int + 2
   `fma.rn.bf16x2` per 4 codes) straight into the A fragment. add a `geezuz-bake` step that (a) orders each lane's 16
   bytes in m16n8k16 A-fragment order for two k16 steps, (b) applies the bit permutation (MS/M1/K masks above), (c)
   rejects e4m3 NaN codes. this replaces the plan's generic "e4m3 tiles converted to bf16 in registers" with a concrete
   layout. it removes `ldmatrix` from the weight path (one 16-B load per lane, conflict-free) and uses ~13% of issue
   slots at R=24. the natural byte order costs 9 instead of 7 instructions per word, also affordable, so the
   permutation is elegance plus ~1/3 fewer ALU ops, not a necessity. one open item: whether HFMA2.BF16 runs on the
   tensor pipe (whitepaper hint). if it does, it adds ~8 clk per fragment to tensor time (R=32: 136 of 152).
4. **gemv tile shape.** weights are the mma A operand (m16 = 16 output rows), tokens are n8 tiles, k steps are k16,
   grouped in k128 chunks aligned to the 128x128 block scales. per warp: 2 m16 tiles (32 rows, one scale block row) x
   R/8 n-tiles x k128. each k128 chunk accumulates into temporary fp32 fragments, then one FFMA per accumulator element
   applies the block scale into the running sum (4·(R/8)·2 FFMAs per 8·2·(R/8) mmas: negligible). at R=24 that is
   24 temporary + 24 running accumulator registers, well inside 255. activations (k128 x R bf16 = 6 KB at R=24) are
   staged once per CTA per chunk and shared by all consumer warps. one persistent CTA per SM.
5. **shared-memory ring size.** the hard cap is 99 KB per block (100 KB per SM, 1 KB reserved) with only 28 KB left as
   L1. the plan's "~8 MB of shared memory" equals 82 x 100 KB, i.e. all of shared memory as ring, which is impossible
   once activation staging, partial-sum exchange and mbarriers need space. realistic: **a 64-80 KB ring per SM = 5.4-6.7
   MB per card = 5.7-7.0 µs of run-ahead at 936 GB/s**. L2 adds at most what is free of 6 MB. the persisting
   set-aside is probably capped near 4.5 MB (query it), and KV, activations and nvlink partials compete for it, so
   count ~2-4 MB (2-4 µs). **total run-ahead ~8-11 µs, not 15 µs.** the serial phases that must hide under it (deltanet
   core, attention, sampling, mtp) need that budget; re-derive it in phase b.
6. **sync primitives.** inside a CTA, use `cp.async` + `cp.async.mbarrier.arrive.noinc` for "slot full" and consumer
   `mbarrier.arrive` for "slot empty", waiting with `test_wait.parity` spins (`try_wait`/`expect_tx` do not exist on
   sm_86). if phase b measures named barriers as clearly cheaper and the ring has <= 7 slots, switch to `bar.arrive`/
   `bar.sync` pairs. across SMs, use `red.release.gpu` / `ld.acquire.gpu` counters; across cards, the same at `.sys`
   scope. no proxy fences are needed. keep exactly one producer warp per CTA: one `cp.async.cg` warp-instruction moves
   512 B, and the SM needs one every ~76 clocks. with no `setmaxnreg`, every extra producer warp holds as many registers
   as a consumer.
7. **cache policy.** stream weights with `cp.async.cg` (L1 bypass) under a
   `createpolicy.fractional.L2::evict_first` policy, so a single-use 13 GB stream does not evict KV, activations or
   the run-ahead prefetched with `prefetch.global.L2::evict_last`. `applypriority ... evict_normal` reclaims those lines
   after use, and `discard.global.L2` drops scratch partials without writeback. all are sm_80 instructions, legal on
   sm_86.
8. **toolchain.** LLVM 23.1.5 emits PTX 9.3 for sm_86. the `__nvvm_*_sync` builtins are rejected under +ptx93, which
   the plan's inline-PTX-only rule for sync primitives already avoids. ship sass built by `ptxas` alongside the PTX;
   PTX JIT needs an R610+ driver.

**not established (measure in phase b):** the sustained boost clock under the water loop; `persistingL2CacheMaxSize`
and `accessPolicyMaxWindowSize` on the 3090; mbarrier arrive / test_wait and named-barrier latencies on sm_86; which pipe
runs HFMA2.BF16 (FMA vs tensor) and whether it contends with `mma.sync`; `prmt` throughput; loaded DRAM latency (for
bytes in flight: ~5.7 KB per SM at an assumed 500 ns); whether a 4-integer-op e4m3 decode exists (none found).

## dram (hw-dram)

Scope: the GDDR6X subsystem of one RTX 3090 FE (GA102, sm_86, 82 SMs), and what it implies for streaming ~13.5 GB of
weights per card per decode cycle. Confidence labels: **datasheet**, **documented** (vendor docs or whitepaper),
**measured by others**, **inferred** (with the derivation shown). Everything here was read in this session unless it is
marked "not re-read".

### 1. the bus, the chips and the access granularity

| quantity | value | label | source |
|---|---|---|---|
| bus | 384-bit, twelve 32-bit memory controllers, 512 KB of L2 per controller (6 MB total) | documented | GA102 whitepaper v1, "GA102 memory subsystem" (p. ~12) and spec tables. [pdf](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) |
| data rate | 19.5 Gbps per pin, 936 GB/s peak (384 × 19.5 / 8 = 936.0) | documented | whitepaper, RTX 3090 spec table (Appendix A) and the "GDDR6X Memory" section (p. 30-31) |
| signalling | PAM4, 2 bits per symbol, so 19.5 Gbps is 9.75 GBaud. MTA coding avoids full-swing transitions | documented | whitepaper p. 30-31 (fig. 16, 17) |
| clock relationships | datasheet example in PAM4 mode: CK 2.5 GHz, CA 5 Gb/s per pin, WCK 5 GHz, DQ 20 Gb/s = 10 GBaud. At 19.5 Gbps this scales to about CK 2.44 GHz and WCK 4.88 GHz | datasheet; the scaling is inferred (linear) | Micron MT61K256M32 GDDR6X brief, Table 1 and Fig. 3. [pdf](https://assets.sourcengine.com/datasheets/810969df-46bf-40b2-9bc7-215fdae2d760.pdf) |
| device | 8 Gb, 2 independent 16-bit channels (x16 mode) or 2 × x8 (x8 mode). 16 banks in 4 bank groups. 16n prefetch. BL8 in PAM4 mode. 2 KB page per channel. Refresh 16k/32 ms. Tc 0 to 95/105 °C. Speed bins -19/-19G (19 Gb/s) and -21 (21 Gb/s) | datasheet | Micron brief, features list and Table 2 |
| bytes per access | 256-bit prefetch per channel in x16 mode = **32 bytes per burst** (x8 mode is 128 bit = 16 B per chip, so a clamshell pair still gives 32 B) | datasheet | Micron brief, Table 2 |
| board topology | 24 chips of 1 GB on a 384-bit bus means clamshell: 12 on the front and 12 on the back, each chip in x8 mode, a front/back pair forming each 16-bit channel, so **24 channels** (2 per controller), about 39 GB/s each, and 24 × 16 = 384 banks | inferred (24 GB / 8 Gb parts on 384 bits; the datasheet's x8 mode) | — |
| exact part on the FE | the marking reported is D8BGW = MT61K256M32JE-19G. The board runs it at 19.5 Gbps, above the datasheet's 19 Gb/s bin. Not verified against a Micron source | measured by others (seller listing), unverified | search result: eBay listing "D8BGW MT61K256M32JE-19G" |
| L2 / L1 granularity | 128-byte L2 lines made of four 32-byte sectors. DRAM moves 32-byte sectors | documented (not re-read here) | Nsight Compute kernel profiling guide, memory tables |
| address interleave | on GA102 (RTX A6000, GDDR6), **each contiguous 256-byte chunk maps to one bank and row, and successive 256 B chunks go to different banks**. The gap between chunks of the same row varies from row to row, so the mapping is hashed. A6000 rows are 2 KB with 64K rows per bank | measured by others | GPUHammer (Lin, Qu, Saileshwar, USENIX Sec 2025), §4 "Observation 1-3". [arxiv 2507.08166](https://arxiv.org/pdf/2507.08166) |
| shared memory | sm_86: 100 KB per SM (carveouts 0/8/16/32/64/100), 99 KB per block (opt-in), 128 KB L1+smem combined, 48 warps (1536 threads) and 16 blocks per SM | documented | CUDA Ampere tuning guide §1.4.1.1, §1.4.2.3. [link](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) |
| cp.async | sizes of 4, 8 or 16 bytes. `.cg` (bypasses L1) needs 16 bytes. Optional `.L2::64B/128B/256B` prefetch-size hint | documented | PTX ISA, cp.async. [link](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) |

**Partition camping.** The 256 B hashed interleave across 12 controllers and 24 channels (above) means any long contiguous
stream spreads evenly over every channel. The old GT200 failure mode, many SMs hammering one partition with power-of-two
strides (described in NVIDIA's 2009 "Optimizing Matrix Transpose" note, not re-read here), does not apply to contiguous
streaming. What does still matter is **DRAM row locality**: a 256 B chunk is one bank and row. A layout that reads 128 B
fragments with large strides (row-major fp8 rows of a 128-column tile) touches half a chunk, and the other half comes
later, possibly after the row has closed. That adds activates (tRC ≈ 45 ns on GDDR6, GPUHammer §5) and costs bandwidth.
**Inferred rule:** bake every 128×128 e4m3 tile as one contiguous 16 KB run (64 whole 256 B chunks), and have producers
issue whole tiles.

### 2. peak vs achievable streaming bandwidth

| measurement | result | % of peak | label | source |
|---|---|---|---|---|
| Chips and Cheese, RTX 3090, VRAM bandwidth scaling, 2 GB region, local size 256 (read-only, Vulkan/OpenCL) | plateau **≈ 890-900 GB/s**, reached by about 30 workgroups. Read off the chart | **≈ 95-96%** | measured by others | "Microbenchmarking Nvidia's RTX 4090", figure "GPU VRAM Bandwidth Scaling". [link](https://chipsandcheese.com/p/microbenchmarking-nvidias-rtx-4090) |
| same article, "Cache and Memory Bandwidth, High Occupancy", RTX 3090 (Vulkan) | 865.6 GB/s labelled at 64 MB. About 800-870 GB/s out to 4 GB | 92.5% | measured by others | same article |
| same article, one workgroup (one SM) on a 3090 | **≈ 47.6 GB/s from VRAM**, 62.8 GB/s from L2 | — | measured by others | same article, "Single Workgroup Bandwidth" |
| te42kyfo gpu-benches `gpu-stream`, NVIDIA A40 (same GA102 die, 84 SMs, GDDR6, 696 GB/s spec), pure-read kernel | **680 GB/s** at ≥ 54% occupancy | **97.7%** | measured by others (the A40 spec of 696 GB/s is from NVIDIA's A40 datasheet, not re-read) | [gpu-stream/a40.txt](https://github.com/te42kyfo/gpu-benches) (raw file fetched) |
| same, A40 triad / scale | 670 / 653-658 GB/s | 96% / 94% | measured by others | same |
| RTX 3090 `bandwidthTest` device-to-device | ≈ 700 GB/s (a copy, counted as read + write) | ~75% (a copy, not a read stream) | measured by others, one forum post | [NVIDIA forum 345210](https://forums.developer.nvidia.com/t/pcie-bandwidth-issue-h2d-very-slow-gen1-but-d2h-reaches-gen4/345210) (via search summary) |
| Ethash on a 3090 (each hash reads 64 random 128 B lines = 8 KiB): stock-ish ~107-110 MH/s | 877-901 GB/s of random 128 B reads | 94-96% | inferred from measured-by-others hashrates (Kryptex user profiles, via search summaries; low quality) | — |
| LLM decode, end to end, on a 4090 (calm, fp16/int8) | ~90% of the weight-read floor; llama.cpp 82% (fp16), 71% (q8) | — | measured by others (different card, end-to-end) | [zeux.io, LLM inference speed of light](https://zeux.io/2024/03/15/llm-inference-sol/) |
| NVIDIA rule of thumb | "best achievable ... typically 75-85%" | — | documented (forum moderator), dated and conservative | search summary of an NVIDIA developer forum answer |

**Conclusion:** a well-formed pure-read stream on a 3090 reaches **~93-96% of 936 GB/s (≈ 870-900 GB/s)**. GA102's
controller can reach ~98% on GDDR6 (A40). I found no clean nvbandwidth or BabelStream number for a 3090 (searched
several times).

**Refresh overhead.** GDDR6 tREFI is 1.9 µs by datasheet (16K/32 ms), but GPUHammer measured **1407 ns on GA102**
(16K tREFIs in ~22 ms). tRFCab for a Samsung 8 Gb GDDR6 is **120 ns** (K4Z80325BC datasheet Rev 1.3, Table 91, the IDD
parameter set; used here as a proxy, since no GDDR6X tRFC value is public). All-bank refresh with nothing hidden would
cost 120 / 1407 = **8.5%** of channel time (inferred). GDDR6 and GDDR6X both support per-bank refresh (REFpb), which
lets reads continue to other banks (GDDR6X brief, command table; Samsung §7.18). The A40's 97.7% shows that the GA102
controller hides almost all of it. **Inferred:** refresh, read-to-read turnaround and page overheads together cost ≤ ~2-5%
on a clean read stream. This is already inside the 93-96% figure.

### 3. latency and Little's law

**Measured unloaded latency (pointer chase):**
- RTX 3090: L1 ≈ 20 ns, L2 ≈ 138 ns, **VRAM ≈ 268-271 ns** (1 GB region). Measured by others: Chips and Cheese 4090
  article, figure "GPU Memory Latency" (dashed 3090 line).
- RTX A6000 (GA102, GDDR6): ≈ 320 ns per access, with 360-380 ns on a row-buffer conflict. Measured by others:
  GPUHammer §4.
- A40 (GA102, GDDR6): **314 ns = 547 cycles at 1740 MHz**, L2 144 ns / 250 cycles. Measured by others:
  gpu-benches `gpu-latency/a40.txt`.

**Loaded latency.** No published 3090 loaded-latency curve was found. I derived one from the A40 occupancy sweep. The
read kernel does one 8 B load per thread, on 84 SMs.
- Light load: 128 threads/SM → 160 GB/s. 128 × 8 × 84 = 86 KB in flight, so the round trip is 86 KB / 160 GB/s ≈
  **540 ns**. That includes thread launch and exit overhead on top of the 314 ns raw latency.
- Near saturation (50%, 768 threads/SM → 677 GB/s; 54%, 832 threads/SM → 680 GB/s): 516-559 KB in flight, so
  **≈ 760-820 ns** (an upper bound, since not every thread is in flight at once).
- So queueing adds **~200-300 ns** at 97% utilisation (inferred). For a 3090: about 270 ns unloaded, **~500-600 ns
  loaded** raw, and ~800 ns effective issue-to-use with per-request overhead (inferred).
- **Design for 1 µs.**

**Bytes in flight needed** (B = 0.95 × 936 ≈ 890 GB/s target; inferred, arithmetic only):

| latency assumed | card-wide in flight | per SM (÷82) | 16 B cp.async per SM | warp-wide LDGSTS.128 per SM |
|---|---|---|---|---|
| 270 ns (unloaded) | 240 KB | 2.9 KB | 185 | 6 |
| 600 ns (loaded estimate) | 534 KB | 6.5 KB | 407 | 13 |
| 800 ns | 712 KB | 8.7 KB | 543 | 17 |
| 1 µs (design) | 890 KB | **10.9 KB** | 680 | 21 |

Cross-check (inferred from measured): the A40 saturates its 696 GB/s with ≤ 6.7 KB/SM in flight (832 threads × 8 B).
Scaling to the 3090's bandwidth at the same latency gives **~8.9 KB/SM**, which agrees with the table.

**Can 82 SMs issuing 16 B cp.async hold this?** Yes, with a wide margin.
- **Capacity:** the target of 16-32 KB/SM in flight (1.5-3× the 1 µs requirement) is 16-32% of the 100 KB smem per SM.
  Card-wide that is 1.3-2.6 MB of a total smem of 82 × 100 KB = 8.2 MB.
- **Issue rate:** 890 GB/s / 82 = 10.9 GB/s per SM ≈ 6 B per SM cycle at ~1.8 GHz, which is one warp-wide LDGSTS.128
  (512 B) every ~85 cycles.
  - Huerta et al. measured on an RTX A6000 (GA102) that the SM's shared memory pipeline accepts one memory instruction
    every 2 cycles. Each sub-core buffers ~5 memory instructions (queue ≈ 4). LDGSTS latency is fixed at 13/39 cycles
    (WAR/RAW) for 32, 64 and 128 bit. Measured by others: "Analyzing Modern NVIDIA GPU cores", §5.4, Table 2,
    [arxiv 2503.20481](https://arxiv.org/pdf/2503.20481).
  - So the stream needs ~2% of the SM's memory-instruction issue slots (inferred).
- **Per-SM ceiling:** one 3090 SM running one 256-thread workgroup pulls **~47.6 GB/s from VRAM**, 4.4× what each SM
  needs. One workgroup per SM reaches the 890-900 GB/s plateau by ~30 SMs (Chips and Cheese figures above). By Little's
  law, 47.6 GB/s × 270 ns ≈ 12.9 KB in flight from one SM is demonstrably sustainable (inferred).
- **MSHR-like limits:** NVIDIA documents no per-SM outstanding-request or MSHR count for GA102, and none was found in
  the microbenchmark literature. The empirical bounds above (≥ 13 KB/SM in flight, ≥ 47 GB/s per SM) are the best
  available. **Not established:** the hard per-SM cap on outstanding L1TEX/LDGSTS sectors.
- **Practical shape (inferred):**
  - 4-8 producer warps per SM, each keeping 4-8 LDGSTS.128 groups outstanding (`cp.async.commit_group` /
    `wait_group N`), which gives 8-32 KB per SM.
  - Use `.cg` with the `.L2::256B` prefetch hint to match the 256 B DRAM chunk.
  - Each warp instruction covers 512 contiguous bytes. Read whole contiguous 16 KB tiles.

### 4. GDDR6X EDR under overclock: what is and isn't covered

- **What EDR is (documented).** The memory interface computes a link CRC. When it shows a transmission error, the
  transfer "is retried or replayed until a successful memory transaction occurs". As the overclock rises, retries eat
  bandwidth and performance plateaus. NVIDIA also says crashes will still occur if pushed further, and that "EDR may not
  prevent all crashes before performance plateaus". Source: GA102 whitepaper, Appendix B (p. 49, fig. 25).
- **What the CRC is (datasheet).** GDDR6X: "Read/write data transmission integrity secured by cyclic redundancy check",
  with programmable CRC read and write latency (Micron brief, features list). The GDDR6 EDC, as the closest public
  definition:
  - It is an 8-bit CRC (X⁸+X²+X+1) per byte lane over 72 bits (8 DQ + DBI × BL8), in full-rate or half-rate mode.
  - "The device itself does not perform any error correction".
  - The controller "can decide ... [to] retry the READ or WRITE".
  - Detection: 100% for 1, 2 and 3-bit errors and bursts ≤ 8 at full rate. 99.9% for 4-bit errors. Half rate is 99.6%
    for 3- and 4-bit errors.
  - Source: Samsung K4Z80325BC GDDR6 datasheet, §7.15 and Table 58. GDDR6X's CRC parameters and NVIDIA's rate mode are
    not public.
- **So "errors become retries" is probabilistic, not guaranteed.** Even on the data link, some multi-bit patterns pass
  the CRC (inferred from the table: roughly 10⁻³ of 4-bit link errors at full rate, worse at half rate). The escape rate
  rises with the raw bit error rate, which is exactly what an overclock raises.
- **Not covered by EDR at all (inferred from what the CRC protects):**
  1. **DRAM array errors.** These are retention failures (worse with heat), weak cells and Rowhammer-style disturbance.
     The CRC is computed over data as read out, so a flipped stored bit arrives with a valid CRC. GPUHammer produced real
     bit flips on GA102 GDDR6 (A6000), though none on the RTX 3080 (GDDR6X) it tested. GeForce has no ECC: the 3090
     nvidia-smi dump below shows `ECC Object: N/A`.
  2. **Command/address bus errors.** The CRC covers DQ/DBI data only, and the GDDR6X feature list shows CA bus inversion
     and training but no CA parity. A mis-latched address under overclock reads or writes the wrong location, with a
     valid CRC.
  3. **Errors inside the GPU** (crossbar, L2, SM): EDR protects only the GPU-to-DRAM pins.
  4. **Write path.** A detected write error is replayed, so it is safe, but an escaped one is persistent and silent.
     The KV cache and activations are written every cycle. The weights are written once at load, then only read.
- **Is there a counter?** None was found for GeForce. NVML exposes ECC counters, which are N/A on GeForce. The
  `Replays Since Reset` field in `nvidia-smi -q` is **PCIe** replay, not memory EDR (it shows up in the 3090 dump below).
  The only observable is bandwidth flattening.

### 5. overclock headroom vs back-side temperature

**Units.**
- nvidia-smi reports the 3090's max memory clock as **9751 MHz**, and the data rate is 2× that (19.5 Gbps).
- An Afterburner-style offset of +N MHz therefore adds 2N Mbps per pin: +1000 gives 21.5 Gbps and 1032 GB/s (+10.3%).
- Linux `nvidia-settings GPUMemoryTransferRateOffset` and NVML's MemClk VF offset are in **transfer-rate units, twice
  the Windows number**: Linux +2000 equals Windows +1000.
- Sources: a forum post on NVML "secretly uses the 2x DDR clock" (NVIDIA forum via search), Kryptex "2000 for Linux,
  1000 for Windows" (via search; the page itself returned proxy 502), and the dump below (measured by others).

**Typical stable mining offsets on 3090s** (measured by others, community profiles, low quality):
- Windows-unit offsets of +1000 to +1550, i.e. 21.5-22.6 Gbps.
- Ethash went from ~107-110 to ~120-127 MH/s, about +9% to +18%.
- Hashrate is a direct random-read bandwidth meter: 126.6 MH/s × 8 KiB = 1037 GB/s at about 22.6 Gbps (1085 GB/s
  peak), i.e. ~96% (inferred). So bandwidth scales roughly linearly with clock up to the EDR knee.
- Source: Kryptex profile list via search summary.

**Primary-ish dump from a Linux 3090 rig** (Hiveon forum, 2022, `nvidia-smi -q` pasted):
- `+3000` (Linux units) gave **Memory 10925 MHz** in **P2** under the miner.
- `Max Clocks Memory: 9751 MHz`.
- `Memory Current Temp: N/A`.
- `Power Draw 319 W` with the core locked to 1140 MHz.
- `Default Power Limit 390 W` (Strix).
- Source: [hiveon thread 69913](https://hiveon.com/forum/t/stable-3000-memory-overclock-still-too-dangerous/69913)
  (JSON fetched).
- Note: 10925 is not 9751 + 1500. Either the P2 base is lower than 9751 or the offset applied differently; see the open
  items.

**A further reference point:** the 3090 Ti (2 GB single-sided modules, 21 Gbps stock) has been run at 24 Gbps,
1181 GB/s, 132 MH/s. Sources: Tom's Hardware / HotHardware, via search. The plain 3090's two-sided module placement is
the usual explanation for its lower headroom (search summary).

**Thermals.**
- The memory junction limit is ~110 °C. Igor's Lab calls it "the 110 °C maximum temperature, as specified by Nvidia and
  Micron". Above it the memory "is now automatically throttled down". The sensor is only accurate to ±2 °C. Source:
  [Igor's Lab, 3090 FE + Alphacool GPX-N, p.1](https://www.igorslab.de/en/hot-headed-unicorn-under-water-alphacool-eisblock-aurora-acryl-gpx-n-for-nvidia-rtx-3090-founders-edition-installation-and-test/)
  (measured by others / secondary). Tom's Hardware also reports throttling only at 110 °C.
- The datasheet's case limit is Tc ≤ 95/105 °C (Micron brief).
- On the stock FE air cooler: mining reached 110 °C within seconds, Port Royal (+1000 mem) ~100 °C. The passive
  backplate "without airflow remains a dead end". The back-side modules make the 3090 worse than the 3080.
- Under water (same 3090 FE, Alphacool block with its passive backplate, Port Royal loop, +1000 mem): memory junction
  settled at **+44 K over water**, versus 76 K over ambient on air. It levelled off after about 400 s.
  - Source: [Igor's Lab p.5](https://www.igorslab.de/en/hot-headed-unicorn-under-water-alphacool-eisblock-aurora-acryl-gpx-n-for-nvidia-rtx-3090-founders-edition-installation-and-test/5/),
    read in the built-in browser (measured by others).
  - Port Royal is a mixed load. A decode weight stream runs at ~90%+ DRAM utilisation continuously, closer to mining.
  - **Inferred:** with water at 30-35 °C, back-side junction would be ~75-90 °C on the XG7 with its passive backplate,
    and higher with an overclock (more I/O power). The headroom to 110 °C is real but not large. A fan on the backplate
    is likely the cheapest margin.

**Power coupling (inferred).** Memory-only mining at 1140 MHz core already drew ~310-320 W on a 3090. A memory
overclock raises memory and I/O power inside the same board limit (FE: 350 W stock, up to ~400 W per Igor's Lab). Under
a combined decode load the overclock may take power from the SM clock, so gains must be measured at the power limit
actually used.

### 6. Linux tools: setting and reading memory clocks and temperature

| task | tool | notes | label |
|---|---|---|---|
| lock memory clock (within the stock range, not an overclock) | `nvidia-smi -lmc MIN,MAX` / `nvmlDeviceSetMemoryLockedClocks` (root; `-rmc` resets) | NVML: "For Ampere or newer **fully supported** devices". GeForce support is reported as mixed in forums. `-lmi` shows the lock style supported | documented ([nvidia-smi docs](https://docs.nvidia.com/deploy/nvidia-smi/index.html), [NVML device commands](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceCommands.html)) |
| lock SM clock | `nvidia-smi -lgc` / `nvmlDeviceSetGpuLockedClocks` | reported to work on GeForce RTX | documented + measured by others |
| override CUDA's default (P2) clocks | `nvidia-smi -cc 1` (`--cuda-clocks=OVERRIDE`, root) | may be the way to get P0 memory clocks under CUDA. Unverified on a 3090 | documented (flag), effect unverified |
| memory clock offset, headless | `nvmlDeviceSetMemClkVfOffset(dev, offset)` (driver ≥ 520.56.06, "requires elevated privileges"; deprecated in favour of `nvmlDeviceSetClockOffsets`, "works on Maxwell onwards"). Units are 2× (transfer rate) | no X needed. Python: `pynvml` / `nvidia-ml-py` | documented (NVML docs) + measured by others (forum) |
| memory clock offset via X | `nvidia-settings -a '[gpu:N]/GPUMemoryTransferRateOffset[<top perf level>]=<2×MHz>'` with `Coolbits` (bit 8) in xorg.conf | needs a running X server on each GPU (a headless dummy screen works). Several drivers had regressions | measured by others (forum threads) |
| read memory junction temperature | **not via NVML/nvidia-smi on GeForce**. `Memory Current Temp: N/A` on a 3090 (dump above). The docs say "only available on supported devices". | the long-running NVIDIA forum request ([168346](https://forums.developer.nvidia.com/t/request-gpu-memory-junction-temperature-via-nvidia-smi-or-nvml-api/168346)) | measured by others |
| read memory junction temperature (workaround) | `gddr6` (olealgoritme/gddr6; fork AImmd/gddr6). Reads a register in PCI BAR0 via `/dev/mem`, needs root and `iomem=relaxed` on the kernel command line, and lists GA102 / RTX 3090 as supported | reverse-engineered and unsupported. **GitHub was unreachable from here** (repeated timeouts, browser navigation refused), so the README and source were not read; details are from the search summary | measured by others, unverified |
| throttle evidence | `nvidia-smi -q -d PERFORMANCE` / `--query-gpu=clocks_event_reasons*` (SW/HW thermal slowdown, power cap) | memory-thermal throttling may not appear as a named reason; watch the memory clock itself | documented |
| bandwidth check | own read kernel, or nvbandwidth (`device_local_copy` reports a copy, not a read) | — | documented |

### 7. could not establish

- A clean, published pure-read stream number for a **3090** from nvbandwidth or BabelStream. The ~95% figure rests on
  Chips and Cheese charts and the A40 (GDDR6) data.
- A published **loaded** latency curve for GA102. The 500-800 ns figure is derived from the A40 sweep.
- GA102's per-SM outstanding-request / MSHR limits (undocumented). Only empirical lower bounds exist.
- GDDR6X CRC polynomial, coverage and NVIDIA's EDC mode. Whether any EDR replay counter is exposed on GeForce (none was
  found).
- A tRFC value for GDDR6X (a GDDR6 proxy was used), and whether NVIDIA uses per-bank refresh.
- Whether CUDA work on a 3090 under Linux runs memory at **P2 clocks below 9751 MHz**. The Hiveon dump shows P2, but
  the stock P2 memory clock was not observed.
- Measured 3090 overclock bandwidth sweeps. The FPS Review 3090 FE overclocking article was blocked by Cloudflare in
  both curl and the built-in browser. The Kryptex pages returned proxy 502, so only search snippets were used.
- The `gddr6` tool's exact register offset and how well it is maintained on current drivers.

### implications for the plan

1. **The ~88% assumption (§0.3) is conservative for the stream itself.**
   - Clean read streams on a 3090 hit ~93-96% (≈ 870-900 GB/s), and GA102 can do ~98% on GDDR6.
   - So the weight stream can be budgeted at **~92-94% (≈ 15.3-15.6 ms for 13.46 GB)**, provided the layout is
     tile-contiguous and DRAM never idles.
   - The remaining gap to 88% is margin for KV reads, NVLink traffic sharing L2, and producer stalls.
   - Keep 88% as the planning number until phase b measures it, but treat >90% as the step-2 target, not a stretch.
   - Add a phase-b gate: a plain read microbenchmark on each card must show ≥ 870 GB/s at stock before any kernel tuning
     (it also catches the P2 question).
2. **In-flight sizing is easy and is not the binding constraint.**
   - Little's law needs ~0.9 MB card-wide (~11 KB/SM) at a 1 µs design latency.
   - Budget **16-32 KB/SM of cp.async in flight**: 4-8 producer warps × 4-8 outstanding LDGSTS.128 groups, `.cg` +
     `.L2::256B`, whole 16 KB tiles.
   - That is ≤ 1/3 of the 100 KB per SM (99 KB per block), leaving the rest for mma staging and activations.
   - The plan's "~8 MB smem + 6 MB L2 ≈ 15 µs run-ahead" is arithmetic for a jitter buffer, not for latency hiding.
     - Only ~1 MB has to be in flight. The ring beyond that only decides how long consumers may stall before DRAM idles:
       ring bytes / 936 GB/s, e.g. 4 MB ≈ 4.3 µs, 8 MB ≈ 8.5 µs.
     - L2 is not a safe ring. Prefetched lines are evicted by KV, activations and NVLink traffic, and each 512 KB slice
       holds only its own addresses.
     - So the "serial phases under the weight stream" (§1.6) must run on warps other than the stream consumers, or
       finish inside that few-µs window. The reliable run-ahead is smem only, realistically 3-6 MB card-wide.
3. **The overclock expectation (~8%) is plausible as a bandwidth gain but should be stated as ~5-7% of cycle time.**
   - +8% bandwidth is about +780 MHz in Afterburner units (+1560 in Linux units), i.e. ~21 Gbps, which is the
     datasheet's -21 bin and well inside typical mining-stable offsets (+1000 to +1500).
   - Cycle-time gain is smaller: only the weight-stream part scales, and the overclock adds memory power that can cost
     SM clock at the power limit. Adrien's +15% raw bandwidth giving +4% at TP2 shows the same effect.
   - The binding limit on these cards is **back-side junction temperature** (passive backplate, 110 °C throttle), not
     EDR.
   - Plan for a backplate fan as a default, not a fallback.
   - Also check whether CUDA runs the memory in P2 below 9751 MHz. If it does, either fix it (`-cc 1`, or a clock lock
     where supported) or count the offset from the real P2 base.
4. **The soak procedure must not rely on EDR or golden hashes alone**, because EDR covers only link-data errors and its
   CRC is not exhaustive. Proposed procedure, run on both cards together (shared heat and power):
   - **a. Sweep.** Per offset step (Linux +250, i.e. +125 MHz clock), run a 60-120 s pure-read kernel at steady
     operating temperature. Log GB/s, `gddr6` junction temperature, power, SM clock and clock-event reasons. The EDR knee
     is where GB/s stops tracking the clock ratio.
   - **b. Candidate.** Take the knee minus two steps, and keep junction ≤ ~95 °C at worst-case ambient (≥ 15 K under the
     110 °C throttle and inside the Tc 95/105 °C datasheet range).
   - **c. Integrity soak, several hours at temperature.**
     - An on-device hash of the full weight image every few minutes (≈ 15 ms per pass), compared with the load-time
       hash. This catches array, CA and write-path corruption that EDR cannot see.
     - A pattern write/read test over the free KV region (~9 GB, including the back-side chips). Use several patterns
       including inversions.
     - The golden-logit-hash harness from §1.8.
   - **d. Accept only on:** zero mismatches, zero pattern errors, bit-exact golden hashes, no thermal or power
     clock-events, and stable bandwidth.
   - **e. Operations.**
     - Offsets reset on reboot or driver reload. Apply them from a boot-time unit through NVML (no X needed), and have
       the engine log the observed memory clock and refuse to start if it differs from the accepted value.
     - Repeat the soak after driver changes and seasonally (ambient).
     - Keep a low-rate weight-hash scrub in production, since a silent weight flip persists until reload.

## nvlink and pcie (hw-link)

Confidence labels: **datasheet** (vendor spec), **documented** (vendor programming docs), **measured by others**
(a named post, paper or repo), **inferred** (derivation shown). Upstream geezuz's own changelog counts as
"measured by others": it was measured on a different box, over PCIe 4.0 x8 with the patched driver.

### 1. the link itself

| item | value | confidence | source |
|---|---|---|---|
| generation | third-generation NVLink (same as A100 and A6000) | datasheet | NVIDIA GA102 whitepaper v1, "Third-Generation NVLink" (p. 8) |
| links | "four x4 links" on GA102 | datasheet | same |
| per link | 14.0625 GB/s each direction | datasheet | same; `nvidia-smi nvlink -s` prints `14.062 GB/s` per link (NVIDIA forum threads 232673 and 226922, both on 3090s) |
| per direction | 56.25 GB/s | datasheet | same |
| total (both directions) | 112.5 GB/s | datasheet | same |
| topology | 2 cards only ("3-Way and 4-Way SLI configurations are not supported"); `nvidia-smi topo -m` shows `NV4` | datasheet / measured by others | whitepaper p. 8-9; forum 232673 (`GPU0 NV4 GPU1`) |
| measured, unidirectional | 52.61 / 52.75 GB/s (93.5% of 56.25) | measured by others | forum 232673, p2pBandwidthLatencyTest, P2P=Enabled, Windows 526.86 |
| measured, bidirectional | 99.46 / 99.38 GB/s (88% of 112.5) | measured by others | same |
| measured, bidirectional, Linux | "~100 GB/s ... regardless of the underlying PCIe configuration" (x4, x8, x16) | measured by others | platform-fools.com "When is NVLink Worth It?" (2026-04-27): EPYC 7532, H12SSL-i, Ubuntu 24.04, driver 595 |
| measured, 3090 Ti pair | 52.81 GB/s peer vs 13.28 GB/s host-staged | measured by others, not read in full | club-3090 issue #2000 (search snippet only: GitHub is unreachable from this sandbox) |

Those bandwidth numbers come from **copy engines** (`cudaMemcpyPeer`). I found no published number for
**SM-issued stores** over a 3090 bridge. **Inferred**: SM stores should get close to the same rate if enough
128-byte lines are in flight, but this needs a phase-b measurement (see section 6).

### 2. p2p with the stock linux driver

- **GeForce p2p over NVLink works with the stock driver.** On EPYC with driver 595, NCCL picks `P2P/direct pointer`
  over the bridge at ~100 GB/s. With the bridge unplugged, the same box falls back to `SHM` (host memory) until
  the community patch is installed (**measured by others**, platform-fools.com). Puget's 2×3090 Threadripper Pro
  run without a bridge shows `simpleP2P: NO P2P` and 16.5 GB/s host-staged (**measured by others**, Puget
  Systems, "Problems With RTX4090 MultiGPU...", 2023-02/04, driver 525.85.05).
- **GeForce p2p over PCIe is off in the stock driver.** From 525.105.17, "NVIDIA's intention is for P2P to be
  unavailable on GeForce"; 4090s then report `CANNOT Access Peer` (**measured by others**, same Puget page,
  "Appendix P2P Update").
- **The tinygrad / aikitoria patch is irrelevant here.** It enables PCIe BAR1 p2p. Upstream geezuz needed it
  because it ran "PCIe 4.0 x8 per card ... no NVLink", using the "595.84 open kernel module with the community
  P2P patch ... Resizable BAR on and `iommu=pt`" (repo README.md, "Two GPUs" and "NVLink?"). Use the stock
  driver. **Inferred**: don't install the patched module on this box at all. It adds a second p2p path, and
  nothing here needs it.
- **Bring-up gotchas** (all **measured by others**, NVIDIA developer forums):
  - links show `<inactive>` until persistence mode is on (`nvidia-smi -pm 1`, then reboot): EPYC 7352 + 2×3090
    + 4-slot bridge, Ubuntu 22.04, driver 530 (thread 160561, #15);
  - a bridge that isn't fully seated ("I heard a click sound") leaves all links inactive: 2×3090 FE + A6000 3-slot
    bridge, Ubuntu 20.04 (160561, #13);
  - a card that isn't fully seated does the same (226922, #4);
  - a Windows display on one card took link 0 (226922).

  Acceptance check: `nvidia-smi nvlink -s` shows 4 × 14.062 GB/s on both cards, `topo -m` shows `NV4`, and
  `cudaDeviceCanAccessPeer` is 1 both ways.
- **Open vs proprietary kernel module: not established.** Every working 3090-NVLink report I found either predates
  the open module becoming the default, or doesn't say which module it used. If links stay inactive on the open
  module, try the proprietary module of the same version first.
- **IOMMU.** The CUDA programming guide says "CUDA and the display driver do not support IOMMU-enabled bare-metal
  PCIe peer-to-peer memory transfer ... The IOMMU must be disabled" (**documented**, CUDA Programming Guide
  §3.4.2.5). That rule is about PCIe p2p. **Inferred**: NVLink peer traffic never touches the IOMMU, and
  upstream ran `iommu=pt` without trouble. So the plan's `iommu=pt` is fine, and `iommu=off` is the fallback if
  bring-up misbehaves.

### 3. latency: what is and isn't known

**No in-kernel number exists for a 3090 pair.** I found no published in-kernel remote-store or flag-handoff
latency for a 3090 NVLink pair. The numbers that do exist measure other things:

| number | what it really measures | confidence | source |
|---|---|---|---|
| 3090 NVLink, P2P=Enabled "latency (P2P writes)": 6.54-6.73 µs off-diagonal; 8.5-9.1 µs on the diagonal (same card) | per-op time of back-to-back tiny peer copies, on Windows (WDDM). The diagonal being slower than the peer shows that software overhead dominates | measured by others | forum 232673 (bowdenrj) |
| 2080 Ti NV-SLI (Turing NVLink bridge): local ~5 µs, over NV-SLI ~8 µs, over PCIe ~13 µs | `cudaMemcpy` + `cudaEvent` "start-up latency" (Tartan, built on p2pBandwidthLatencyTest) | measured by others | Li et al., "Evaluating Modern GPU Interconnect", arXiv 1903.04611, §"NV-SLI Latency" |
| DGX-1 NVLink-V1/V2, direct neighbours ~9 µs; "NVLink-V2 exhibits higher latency than NVLink-V1" | same method | measured by others | same paper |
| GB200: L2 round trip 0.306 µs, **remote store 0.792 µs**, all-reduce speed-of-light 1.404 µs | in-kernel: a single `__threadfence()` for L2, and a one-value ping-pong for the remote store, so remote store = (ping-pong − 2·L2)/2 | measured by others | Shen et al. (ETH/NVIDIA), "Every µs Matters", arXiv 2607.16100, §VI |
| B200 NVSHMEM scalar `p` at 512 B: 2.3 µs | in-kernel put, including completion | measured by others | arXiv 2606.05951, §B |
| upstream geezuz, 2×3090 over PCIe 4.0 x8: waiting for the other card 1-2 µs; "written to sent" 24 µs per residual call against a 14 µs transfer floor (184 KB at 13.2 GB/s) | in-kernel, inside the megakernel, with a PCIe p2p flag protocol | measured by others | repo CHANGELOG.md, "What moved the numbers" (w) and "Next" |

**Inferred estimate for this box, to be replaced by the phase-b microbenchmark:**
- one-way remote store until visible in the peer's L2: **~0.6-1.0 µs**;
- full flag handoff (data stores drain, `st.release.sys` flag, the peer's `ld.acquire.sys` poll sees it):
  **~1-2 µs**;
- remote load round trip: **~1.5-2.5 µs**.

Derivation: start from GB200's 0.79 µs remote store, which crosses an NVSwitch hop that a direct bridge doesn't
have, against older and slower NVLink 3 SerDes and core clocks that cut the other way. Add one local L2 round
trip (~0.3 µs) on each side. Treat the result as ±2×.

### 4. memory model: scope, fences, push vs pull, atomics

- **Peer memory needs `.sys` scope.** PTX defines `.gpu` as "all threads in the current program executing on the
  same compute device", and `.sys` as "all threads in the current program, including all kernel grids invoked by
  the host program on all compute devices, and all threads constituting the host program itself" (**documented**,
  PTX ISA §8.5, Table 21). Two operations synchronize only if each one's scope "includes the thread executing the
  other operation" (§8.7, morally strong). So a card-0 writer and a card-1 reader both need `.sys`: `.gpu` is not
  enough on either side. The CUDA guide agrees: "Threads synchronizing across devices operate at the
  thread_scope_system synchronization scope" (**documented**, §3.4.2.3). `__threadfence_system()` is the fence
  whose ordering is "observed by ... all threads in peer devices" (**documented**, C++ language extensions,
  memory fence functions).
- **`membar.sys` is the expensive fence.** On sm_70 and later, `membar` is a synonym for `fence.sc`, and levels
  `cta`/`gl`/`sys` map to scopes `cta`/`gpu`/`sys` (**documented**, PTX §9.7.15.4). So `membar.sys` is
  `fence.sc.sys`, the sequentially consistent fence that PTX calls "slower". `fence.acq_rel` is "light-weight ...
  sufficient ... in most programs", and a release store (`st.release.sys`) folds that fence into the flag write.
  **Measured by others**: upstream replacing `fence.sc.sys` + flag with a bare `st.release.sys` cut "written to
  sent" from 28 to 24 µs per residual call over PCIe (CHANGELOG build w). The protocol to keep:
  - data with plain (weak) stores;
  - flag with `st.release.sys`;
  - consumer polls `ld.acquire.sys` on its own memory, with `__nanosleep` backoff;
  - consumer then reads the data with ordinary or `.cg` loads.
- **A system fence waits on more than it needs to.** The CUDA guide calls this fence interference: "because the
  GPU is waiting on memory operations it is not required to at the source level, the fence/flush may take longer
  than necessary". Its example is a fence that waits "on slower nvlink or PCIe writes from the communication
  kernel". Memory sync domains fix this only on cc ≥ 9.0, and "CUDA will report a count of 1 on devices prior to
  compute capability 9.0" (**documented**, CUDA Programming Guide §4.16). **Inferred** consequences on sm_86:
  - a `.sys` release in one tile epilogue may also wait for other warps' in-flight NVLink stores, and for any
    in-flight **PCIe writes to the host ring**;
  - keep host-ring writes to once per cycle, at a point where no peer flags are about to be released;
  - measure release cost with background traffic (section 6).
- **Push, don't pull.** "Push is generally faster, requiring only half a GPU-to-GPU RTT versus a full RTT for
  remote loads" (**documented**/measured, arXiv 2607.16100 §III-A). Remote stores are posted: the issuing warp
  does not wait for them. A remote load stalls the warp for a full link round trip. Polling your *own* memory for
  the peer's flag costs only a local L2 hit, because the peer's stores land in your L2, "the point of coherency
  (PoC) across GPUs" (same paper, Fig. 10). Upstream already pushed (`qwen38_mk.cu`, `xr_send` / `flag_release_sys`
  / `flag_acquire_sys`); keep that shape.
- **Write-combining and granularity.** Measured copy-engine efficiency is 93.5% of datasheet (section 1).
  **Inferred**: SM stores reach that only as full 128-byte lines (a warp of fp32, or 16-byte vectors per lane),
  contiguous per destination, with many lines in flight across many warps. Scattered 4-byte stores waste link
  packets on headers. I found no public NVLink 3 packet-overhead figure for GA102.
- **Flag inside the data (LL / LL128).** NCCL's LL protocol packs the flag with the data in 8-byte stores, at half
  the payload bandwidth (arXiv 2607.16100 §IV). Upstream tried it over PCIe and dropped it: "slower from 2
  requests up" (CHANGELOG). LL128 (one flag per 128-byte line) is enabled by NCCL only "on platforms which support
  LL128"; "enabling LL128 on platforms that don't support it can lead to data corruption" (**documented**, NCCL
  env docs, `NCCL_PROTO`). Whether a GeForce NVLink pair qualifies is **not established**; `NCCL_DEBUG=INFO` on
  the box would show whether NCCL picks LL128. **Inferred**: LL might pay off only for the small, latency-bound
  mtp exchanges (R ≤ 8). It should not be the default.
- **Atomics over the link.** A peer atomic is atomic at system scope only if
  `cudaDevP2PAttrNativeAtomicSupported` is 1 between the two devices (**documented**, CUDA C++ memory model §5.8.3).
  Its value on a 3090 pair is **not established**; query it at bring-up. The exchange doesn't need peer atomics:
  each card writes only its own receive slots in the peer's memory, and each flag has one writer and holds a
  monotonic epoch. A deterministic summation order (own partial + peer partial, in a fixed order) rules out
  reduce-by-atomic anyway.

### 5. what crosses pcie anyway, and where the slots go

**Both cards hang off CPU0, not one per socket.**
- **Datasheet**, both board revisions:
  - "Slot_6: PCIe x16 (Gen4 x16 bus) slot (from CPU_0)"
  - "Slot_4: PCIe x16 (Gen4 x16 bus) slot (from CPU_0)"
  - "Slot_3 ... (from CPU_1)", "Slot_2 ... x8 (from CPU_0)", "Slot_1 ... x8 (from CPU_1)"

  Sources: Gigabyte `MZ72-HB0_datasheet_v1.0.pdf` (EPYC 7002) and `MZ72-HB0_rev30_datasheet_v1.0.pdf` (EPYC
  7003, PCB rev 3.0), from download.gigabyte.com.
- **Documented**: the rev 1.x manual's block diagram (p. 8) draws three PCIe 4.0 x16 lanes from CPU0 to Slot6,
  Slot4 and Slot2. CPU1 feeds only Slot3 and Slot1. The sockets are joined by "4x16 GOP" xGMI. (Manual read via
  the mcgrp.ru mirror; gigabyte.com refused automated access.)

So host traffic for both cards goes through CPU0's root complex: no xGMI hop if host buffers sit on CPU0's memory.
**Inferred**:
- bind the resident-loop host thread to a CPU0 core;
- allocate the embedding table, the rings and every pinned buffer on NUMA node 0 (`mbind` / `numactl --membind=0`,
  hugepages);
- under NPS4, pick the node that owns the slot-4/slot-6 root ports;
- CPU1 should stay idle or serve only unrelated work.

A GPU reads remote-socket memory only "marginally" slower than same-socket memory (**measured by others**,
arXiv 2312.03113, Fig. 9), so a wrong binding costs a little, not a lot.

**Slot spacing: a 2-slot bridge fits.** The manual's layout and parts list (pp. 6-7) number the slots PCIE_1-4,
then "NVME_0 / NVME_1 Slimline SAS 4i" in the slot-5 position, then PCIE_6.
- **Inferred** from the drawing:
  - the slot centres measured on the drawing are ~75.7 px apart; with the board's 305 mm width spanning
    ~1,138 px, that is 3.73 px/mm, so the pitch is ≈ 20.3 mm, the standard 0.8";
  - PCIE_4 to PCIE_6 is 152 px ≈ 40.7 mm, or 2 pitches;
  - that is exactly the spacing of the RTX A6000/A5000 2-slot bridge.
- **Measured by others**: Ampere bridges are "functionally identical" across the GeForce and A-series brandings
  ("2- and 3-slot for RTX A-series and only 4-slot for GeForce"). Puget tested a 2-slot bridge on 3090s, but only
  on 2-slot-wide Gigabyte Turbo cards (Puget, "NVIDIA NVLink 2021 Update and Compatibility Chart", 2021-02-24).
- The 3090 FE's air cooler is ~3 slots wide, so 2-slot spacing works only with the water blocks.
- **Inferred risks**, check on the bench:
  - the bridge must clear the water-block terminals and fittings;
  - the card in PCIE_4 overhangs the NVME_0/1 connectors in the slot-5 position, so cabled U.2 there may collide;
  - no 3090 **FE** + 2-slot bridge combination has been reported working, though a 3090 FE pair + A6000
    **3-slot** bridge has (forum 160561 #13).

**PCIe 4.0 x16 to the host.**

| item | value | confidence | source |
|---|---|---|---|
| raw rate | 16 GT/s; "x16 PCIe 4.0 slot providing up to 64 GB/sec of peak bandwidth" (both directions) | datasheet | GA102 whitepaper, "PCIe Gen 4" |
| per direction, after encoding | 31.5 GB/s | inferred | 16 GT/s × 16 lanes × 128/130 ÷ 8 |
| effective, measured | ~24 GB/s | measured by others | EMOGI, arXiv 2006.06890, "Performance Scaling with PCIe 4.0" (DGX A100, Rome); arXiv 2312.03113 uses 24,000 MB/s effective |
| GPU load from host DRAM (zero-copy) | "around 1+ usec" | measured by others | arXiv 2312.03113, Fig. 9 |
| PCIe round trip | 1.0-1.6 µs (GPU to FPGA, Gen3) | measured by others | EMOGI §4 |
| kernel polling a mapped pinned flag, CPU store seen after | "∼1–2 µs" | measured by others | arXiv 2512.16056, §"Dummy Task" sync |
| upstream, Gen4 x8 | 13.2 GB/s | measured by others | repo CHANGELOG "Next" |

What that means for the plan's host traffic (**inferred**):
- **Resident ring.**
  - Each GPU poll of a host-memory descriptor costs one PCIe round trip (~1-2 µs) plus a read TLP; poll with
    `__nanosleep` backoff, from one thread per card.
  - GPU-to-host result writes are posted (no wait at the issuer), but they count against the next `.sys` release
    (section 4).
  - At one handoff per ~17 ms cycle, the ring costs ~2-4 µs per cycle. That is negligible, as long as nothing in
    the hot path waits on it.
- **Embedding rows for the main pass.**
  - Each card fetches R rows of 10 KB (bf16 × 5120): at R = 24 that is 240 KB per card, ~10 µs at 24 GB/s plus
    ~1.5 µs latency.
  - Both cards fetch at once through separate x16 ports. CPU0's 8 channels of DDR4-3200 (204.8 GB/s) are not the
    limit.
  - It can overlap the start of layer-0 weight streaming.
- **Embedding rows inside the mtp chain: a new serial cost.**
  - Each draft step's mtp input embeds the token just drafted (`fc(concat(embed(t), hidden))`), so every serial
    draft step would pay a host round trip: ~1.5 µs plus slots × 10 KB, ~5 µs at 8 slots, times 4 drafts per
    cycle.
  - Drafts come only from the 32k draft vocabulary, so the fix is to keep those 32,768 rows in vram on each card:
    32,768 × 5120 × 2 B = 335.5 MB. That also covers the draft rows of the verify pass.
  - Only the one corrected or bonus token per slot per cycle then needs a full-vocab host fetch.

### 6. phase-b microbenchmarks that settle the unknowns (≈1 day)

1. `nvidia-smi nvlink -s/-c`, `topo -m`, and three p2p attributes (`cudaDevP2PAttrNativeAtomicSupported`,
   `PerformanceRank`, `AccessSupported`). Run on the stock driver with the open module, then the proprietary
   module if needed.
2. Flag ping-pong, one warp per card, timed with `%globaltimer`, 10⁵ iterations, in three variants:
   `st.release.sys`/`ld.acquire.sys`; relaxed `.sys` with no fence; `fence.sc.sys` + store. Half the round trip is
   the **handoff latency** used in section 7.
3. Push R × 20 KB of fp32 with 16-byte vector stores, then release a flag; the peer acquires and reads. Sweep R
   (1, 8, 24) and the number of sending CTAs. This gives SM-store bandwidth and the latency-versus-size curve.
4. The same exchange done as remote loads (pull).
5. Cost of a `.sys` release with no other traffic, with background peer stores, and with background host-ring
   writes. This is the fence-interference size on sm_86.
6. Kernel polls a mapped pinned flag: host-write-to-seen latency. Then a 10 KB × R embedding fetch, one binding on
   NUMA node 0 and one on node 1.

### 7. implications for the plan

- **The fused per-tile fp32 all-reduce over NVLink.**
  - Size: at R = 24 rows, one exchange is 24 × 20 KB = 480 KiB per direction, ~9.3 µs of link time at the
    measured 52.7 GB/s. Over a cycle that is 128 × 480 KiB ≈ 63 MB per direction (~1.19 ms of link time); at 190
    exchanges, ~93 MB (~1.77 ms) (**inferred**).
  - Fused per tile, the transfer hides under the weight stream. One layer streams ~190 MB per card (12.19 GB / 64),
    ~203 µs at 936 GB/s. Two exchanges occupy the link for ≤ ~19 µs of that, under 10%.
  - What stays exposed per exchange is the tail: the last tile's partial (~12 KB, ~0.25 µs) plus one handoff
    (~1-2 µs). So **~1.2-2.5 µs × 128-190 exchanges ≈ 0.15-0.48 ms per cycle** (**inferred**, ±2× until
    microbenchmark 2 runs).
  - Unfused (compute, then send in bulk): ~(9.3 + 1.5) µs × 128-190 ≈ **1.4-2.1 ms per cycle**.
  - Compare upstream over PCIe x8: ~24 µs "written to sent" per call. The fused NVLink exchange should remove
    roughly an order of magnitude of exchange stall. That is consistent with the plan's step-2 estimate, but the
    estimate rests on an unmeasured handoff latency.
- **Push, not pull.**
  - Each card stores its partial into a per-source receive slot in the peer's memory, then releases a per-tile (or
    per-group) flag with `st.release.sys`.
  - Consumers poll local memory with `ld.acquire.sys` and add own + peer in a fixed order, so both cards stay
    bit-identical.
  - No remote loads, no peer atomics, and no `membar.sys` / `fence.sc.sys` on the hot path.
- **Flag scope is `.sys` on both sides; `.gpu` is wrong for peer memory, not just slow.**
  - The flag is single-writer with a monotonic epoch, as upstream already did.
  - Watch fence interference: sm_86 has one sync domain, so keep PCIe host-ring writes away from release points.
  - Flag granularity is a trade-off to measure: one flag per tile adds releases, one per CTA group adds tail
    (upstream went to one per block over PCIe).
- **mtp draft steps' serial exchanges.**
  - Each draft pass has ~2-3 exchanges with R ≤ 8 rows (≤ 160 KB, ~3 µs of transfer), with nothing to hide
    behind. ~10-15 serial exchanges × ~2-5 µs ≈ **20-75 µs per cycle** (**inferred**).
  - Experiment 2.5 (replicated drafter) spends ~1.3 ms per cycle to remove that. It breaks even only at ~90-130 µs
    per exchange, which is implausible over NVLink. **Expect 2.5 to lose**; microbenchmark 2 decides it in under
    an hour, before any drafter code is written.
  - The bigger serial hazard in the mtp chain is host-resident embeddings. **Add 32k draft-vocab embedding rows in
    vram (335.5 MB per card) to part 1.**
- **Platform corrections to 1.7.**
  - PCIE_6 and PCIE_4 are both on **CPU0** (datasheet + manual), so put every host buffer and the loop thread on
    NUMA node 0.
  - The 2-slot bridge matches PCIE_4/PCIE_6 spacing (~40.6 mm, inferred from the manual layout), but only with
    water blocks. Check clearance against the NVME_0/1 connectors and the block fittings.
  - Stock driver, persistence mode on, `iommu=pt`; no p2p patch.

## power and clocks (hw-power)

scope: rtx 3090 founders edition (pci subsystem 10de:147d), water-cooled (xg7 front block, passive backplate over the 12
back-side gddr6x), linux bare metal, cuda compute only (no x server). confidence labels: **datasheet**, **documented**
(nvidia docs / nvml / nvidia-smi manual), **measured by others**, **inferred** (derivation shown).

### 1. power limits and the real ceiling

| item | value | source | confidence |
|---|---|---|---|
| default power target (fe vbios 94.02.27.00.0a, subsystem 10de 147d) | 350.0 w | techpowerup vgabios db, https://www.techpowerup.com/vgabios/225524/nvidia-rtx3090-24576-200901 ("board power limit: target 350.0 w, limit 400.0 w, adj. range -71%, +14%") | documented (vbios dump) |
| max settable power limit (`nvidia-smi -pl`) | 400.0 w (= +14%) | same vbios entry; igor's lab fe review p.16 ("a maximum of 400 is allowed") https://www.igorslab.de/en/nvidia-geforce-rtx-3090-founders-edition-review-between-even-between-evaluate-and-common-decadence-if-price-is-not-all/16/ | documented |
| min settable power limit | ~100 w (350 x (1-0.71)) | same vbios entry | inferred (from -71%) |
| vbios thermal target / max | rated 83 c, max 90 c | same vbios entry ("thermal limits: rated 83.0 c, max 90.0 c") | documented |
| 400 w is the binding limit even under water | gpu boosts ~100 mhz (~5%) higher under water, "but unfortunately the power limit of 400 w is the hard limit here" | igor's lab, alphacool 3090 fe block test p.5 https://www.igorslab.de/en/hot-headed-unicorn-under-water-alphacool-eisblock-aurora-acryl-gpx-n-for-nvidia-rtx-3090-founders-edition-installation-and-test/5/ | measured by others |
| shunt mods | "since the firmware can also evaluate the mosfet dcr of the plc, shunt mods are no longer so easily possible" | igor's lab fe review p.2 (board analysis) https://www.igorslab.de/en/nvidia-geforce-rtx-3090-founders-edition-review-between-even-between-evaluate-and-common-decadence-if-price-is-not-all/2/ | documented (teardown analysis) |
| vrm | nvvdd 10 phases (mp2888a + mp86957 smart power stages), msvdd 6 phases (mp2886), memory 4 phases (mp2884b); 0.7-1.2 v range | igor's lab fe review p.2 | measured by others (teardown) |
| connector | modified molex micro-fit 3.0 12-pin, fed by bundled 2x 8-pin adapter | igor's lab connector story p.2 https://www.igorslab.de/en/nvidias-connector-story-eps-vs-12vhpwr-connector-unfortunately-good-doesnt-always-win-but-evil-does-more-and-more-often-background-information/2/ | measured by others |
| slot load at max pl | pcie slot 12 v rail reaches 5.5 a at stock full load and 5.6-5.7 a at the 400 w limit (cem spec 5.5 a = 66 w) | igor's lab fe review p.16 | measured by others |
| transients | 1-10 ms peaks above average; igor budgets 460-500 w of psu per card at a 400 w limit | igor's lab fe review p.16 | measured by others |

**the real ceiling is firmware, not copper.** the 12-pin is fed by 2x 8-pin (2 x 150 w per cem) plus the slot (66 w
on 12 v), i.e. ~366 w "in spec"; the 400 w vbios limit already sits slightly above that, and the slot is already at its
5.5 a limit. no vbios flash / shunt mod is on the table (fe firmware also reads power-stage dcr). **inferred**: for two
cards at 400 w, budget ~1000 w of psu headroom for the gpus alone (igor's 460-500 w per card).

### 2. clocks: what gpu boost does in memory-bound vs tensor-bound phases

- **gpu boost scales core clock with power, not with memory-boundness.** in a 3090 llama.cpp decode of qwen3.6-27b
  (memory-bound, batch 1, driver 590.48.01), the core clock at each power limit was: 350 w: 1710 mhz / ~333 w drawn,
  32.0 tok/s; 300 w: 1575 mhz, 33.0 tok/s; 275 w: 1530 mhz, 32.7; 250 w: 1395 mhz, 31.7; 200 w: 480 mhz, 20.6;
  150 w: 8.3 tok/s. memory clock 9501 mhz at every level. source: jean brito, "rtx 3090 power limit: finding the sweet
  spot", https://jeanfbrito.github.io/posts/rtx-3090-power-limit-sweet-spot/ — **measured by others** (single card, one
  run per level, 4000-token essay; ±3% is within noise). takeaways: (a) the gpu does *not* lower core clocks because the
  work is memory-bound; it burns ~333 w at stock while delivering bandwidth-bound throughput; (b) a memory-bound decode
  loses <3% down to 250 w; (c) below ~200 w the core clock collapses and so does throughput — the 3090 memory subsystem
  plus uncore needs a large fixed share of the budget.
- **miners ran memory-bound ethash on 3090s with the core locked at 1080-1170 mhz** (`nvidia-smi -i $GPU -lgc
  $VALUE`, works on linux), noting "very high memory oc need higher clock". source: lolminer wiki, "fix clock nvidia for
  cuda by os", https://github.com/Lolliedieb/lolMiner-releases/wiki/Fix-Clock-Nvidia-for-Cuda-by-OS-(English) —
  **measured by others**. this also confirms `-lgc` works on geforce ampere.
- **our cycle is not pure llama.cpp decode.** plan 0.3: ~0.5 ms per verify row at the dense bf16/fp32-acc rate (71
  tflops is the spec at the 1695 mhz boost clock, see the fe vbios boost clock 1695 mhz above), row budget ~24, so
  t_flops ≈ 12 ms against a 16.3 ms weight stream. **inferred**: t_flops scales as 1695/f_core. at 1395 mhz (what a
  250 w cap gives) the 24-row compute becomes 12 x 1695/1395 = 14.6 ms, nearly co-bound with the stream; at 1900 mhz it
  is 10.7 ms. so unlike batch-1 decode, the row budget *does* depend on the sustained core clock, and power-capping is
  not free for this engine. the crossover row count (plan: ~32) moves with f_core in the same ratio.

### 3. the p2 memory-clock trap (real on the 3090)

| fact | value | source | confidence |
|---|---|---|---|
| p0 max memory clock (nvidia-smi units) | 9751 mhz = 19.5 gb/s (gpu-z 1219 mhz) | fe vbios entry above ("memory clock 1219 mhz"); `nvidia-smi -q` "max clocks memory: 9751 mhz" on 3090s, hiveon forum https://hiveon.com/forum/t/stable-3000-memory-overclock-still-too-dangerous/69913 | documented / measured by others |
| memory clock under cuda load | 9501 mhz = 19.0 gb/s (gpu-z 1188 mhz) | nicehash quickminer issue #206, nicehash dev reply: "mining is performed in p2 state and not p0 ... memory clock - which is for 200 mhz less in p2 state. that is why you see 9501 instead of 9751" https://github.com/nicehash/NiceHashQuickMiner/issues/206 ; jean brito table above (9501 mhz under llama.cpp, linux driver 590.48.01); strata-qwen3.8-flash issue #32 (9,701 = 9,501 + 200 offset under cuda load; windows driver 591.86) https://github.com/KhaledElzarw/strata-qwen3.8-flash/issues/32 | measured by others (3 independent sources, linux and windows) |
| bandwidth cost of p2 | 9501/9751 = -2.56%: 936 gb/s -> ~912 gb/s peak | — | inferred |
| why | cuda contexts on geforce raise the gpu to p2, not p0; nvidia has never documented it. "the internal workings of the power management and clock control are not publicly documented by nvidia" (njuffa) https://forums.developer.nvidia.com/t/performance-state-switches-from-p0-to-p2-when-starting-program/269166 ; "no nvreg option" to disable it https://forums.developer.nvidia.com/t/remove-p2-forced-state-from-drivers/241998 | measured by others |

how to get the full memory clock under compute — each option and its evidence:

1. **windows "cuda - force p2 state" = off (nvidia profile inspector)**: the known fix on windows (adobe community post,
   strata issue #32 plan). **not available on linux** (no profile inspector, no nvreg key). measured by others.
2. **`nvidia-smi -lmc 9751` / `nvmlDeviceSetMemoryLockedClocks(dev, 9751, 9751)`**: documented to "supersede application
   clock values and take effect regardless of whether a cuda app is running", "for ampere or newer fully supported
   devices", root, not persistent across driver reload; `-lmc` was added in nvidia-smi v450
   (https://docs.nvidia.com/deploy/nvidia-smi/index.html changelog;
   https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceCommands.html). **but** the one report I found of
   `-lmc 9751` on a card whose supported-clocks table tops out at 9751 mem / 2100 graphics (that table matches a 3090;
   gpu name not given, driver 535.161.07, 4 gpus) shows it settled at **9501** under load, with `-lgc 900` set by a
   separate call: https://forums.developer.nvidia.com/t/sudo-nvidia-smi-lgc-lmc/284479. the reply (njuffa, not nvidia
   staff) suggests setting both clocks in one call and starting from the highest graphics clock. status: **unresolved;
   test at bring-up.** also reported: a memory offset breaks the memory lock (lock ignores offsets)
   https://forums.developer.nvidia.com/t/applying-memory-clock-offset-breaks-memory-clock-locking/281722, and on a 5090
   with driver 610 `-lmc` caused a large slowdown https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1266 (measured
   by others, different gpu).
3. **`nvidia-smi --cuda-clocks=1` (OVERRIDE)**: documented only as "overrides or restores default cuda clocks ... requires
   root" (nvidia-smi manual). reported to let a titan v run compute in p0
   (https://forums.developer.nvidia.com/t/nvml-equivalent-of-nvidia-smi-cuda-clocks/297717); suggested for a 3090 together
   with `nvidia-settings -a GPUPowerMizerMode=1` (needs x), with **no confirmation that it worked**
   (https://forums.developer.nvidia.com/t/nvidia-gpu-3090-performance-mode-setting/310362). unverified.
4. **nvidia-modeset `disable_vrr_memclk_switch=1`**: one linux user reports it "forces p0 even with cuda" (thread
   241998, post of 2024-10-11). single report, unverified, side effects unknown.
5. **`CUDA_DISABLE_PERF_BOOST=1`** (new in 580.105.08): does the *opposite* — it stops cuda from boosting to p2, for
   nvdec/nvenc power savings (thread 241998, 2025-11-06 post). do not set it.
6. **memory offset instead of p0**: an offset raises the p2 memory clock too (strata #32: +200 -> 9,701; +500 -> 10,002
   under cuda load). so a memory overclock makes p2 moot: run in p2 at 9501+offset. note nicehash's warning in reverse:
   an offset tuned in p2 is ~250 mhz higher if the card ever enters p0. measured by others.
7. **`nvidia-smi -ac` (application clocks)**: not supported on geforce ampere ("setting applications clocks is not
   supported"; nvidia-smi manual: "supported on maxwell-based geforce and ... tesla/quadro/titan"), and deprecated since
   v590. documented.

### 4. locked clocks

- `nvidia-smi -lgc MIN,MAX [--mode=0|1]`: "supported on volta+. requires root." nvml `nvmlDeviceSetGpuLockedClocks`:
  "for volta or newer fully supported devices", takes effect "regardless if a cuda app is running", reset on reboot or
  driver reload. documented (nvidia-smi manual; nvml device commands). works on 3090 per lolminer/hiveos usage
  (measured by others).
- `-lmc`: added v450, ampere+ (documented); behaviour under p2 on the 3090 unresolved (section 3).
- what locking buys — nvidia's tensorrt guide
  (https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html), documented:
  - floating clocks give "unstable performance measurements across inferences because every cuda kernel can run at
    slightly different clock frequencies"; locked clocks give "consistent performance measurements", but "the average
    performance will not be as good as when the clock is floating or is locked at maximum frequency with throttling".
  - "after the clock frequency is locked, it should stay at that frequency unless power or thermal throttling occurs ...
    when the throttling kicks in, the device behaves like the clock floats." **a lock is only stable if it sits below
    what the power limit can sustain.**
  - power capping "can lead to unstable performance measurements if the measurements are taken within a short time,
    such as within 20 ms" — that is exactly our cycle length (15-25 ms).
  - gaps between inferences lower average power, so the clock is throttled less; benchmark numbers with idle gaps
    overstate the steady-state clock. relevant: our serial/latency-bound phases are low-power gaps inside each cycle.
  - thermal throttling at ~85 c "for most gpus" (moot under water).
- **no published number for cycle-time variance reduction** on a 3090 inference loop. i found none; it has to be
  measured (cv of cycle time, locked vs floating).
- **inferred**: with gpu boost free, the clock steps with power and temperature every few tens of ms; with our
  alternating ~16 ms memory phase and short tensor/serial phases, the enforced power (1 s average per nvidia-smi
  "average power draw") will sit near the cap and the controller will dither the clock across cycles. a lock at a
  clock the 400 w budget sustains under the worst phase mix removes that dither; a lock above it does nothing.

### 5. power draw of the memory system

| item | value | source | confidence |
|---|---|---|---|
| gddr6x device energy | 7.25 pj/bit (vs 7.5 for gddr6), micron | tom's hardware https://www.tomshardware.com/news/micron-reveals-gddr6x-details-the-future-of-memory-or-a-proprietary-dram ; anandtech https://www.anandtech.com/show/15978 (anandtech's text says per byte; tom's says per bit) | documented (vendor claim, secondhand) |
| dram device power at full stream | 936 gb/s x 8 = 7.49 tb/s x 7.25 pj/bit ≈ **54 w** (912 gb/s in p2: ≈ 53 w); excludes the gpu-side phys/memory controller and mvddc vrm losses | — | inferred |
| igor's pre-launch 3090 split | 350 w tgp ≈ 230 w ga102 + ~60 w for 24 gb gddr6x | tweaktown summary of igor's lab https://www.tweaktown.com/news/73190/geforce-rtx-3090-ga102-consumes-230w-24gb-gddr6-60w-power/index.html | inferred by igor (estimate, not a measurement) |
| memory-bound decode total board power | ~333 w at the 350 w limit, 1710 mhz core; ~247 w at 250 w, 1395 mhz, -1% tok/s | jean brito (above) | measured by others |
| memory-bound with core locked low | 3090 ethash at 1080-1170 mhz core | lolminer wiki | measured by others |
| nvml memory power reading | "gpu memory power readings" exist in nvidia-smi, but the manual lists them under module power (hopper+ datacenter); not expected on a 3090 | nvidia-smi manual | documented (absence on 3090 inferred) |

**inferred conclusion**: a 3090 streaming at full bandwidth spends ~55-60 w in the dram devices plus memory phy/
controller and vrm loss on top, out of 350-400 w; the rest is core voltage x clock. memory power is roughly fixed by
bandwidth, so every watt the power cap removes comes out of the core clock, which our row budget needs (section 2).

### 6. undervolting / curve

- linux has no v/f curve editor. the linux equivalent is `-lgc` (cap the clock) plus a positive graphics clock offset
  (`nvmlDeviceSetGpcClkVfOffset`, or `nvmlDeviceSetClockOffsets` from nvml r555) so the locked clock is reached at a
  lower voltage point. nvml docs mark the old offset calls deprecated in favour of `nvmlDeviceSetClockOffsets` ("works on
  maxwell onwards") (documented). per-pstate offsets don't work as documented: an offset set for one pstate applies to
  all (565 and 570 drivers), and "there is no way to overclock any performance state but 0" (user claim)
  https://forums.developer.nvidia.com/t/nvmldevicegetminmaxclockofpstate-nvmldevicesetclockoffsets-issues/318332
  (measured by others). an offset that applies to all pstates is what we want, because cuda runs in p2.
- measured effect on a 3090 llm engine (strata issue #32, windows): undervolt profile 950 mv / 1,995 mhz flat / mem +200
  / pl 100% vs a +135 mhz curve: decode 0.999 [0.985-1.012] (no change), prefill 1.009 (slightly slower), max temp
  67-70 c vs 72 c, prefill still pinned at the 388 w cap. measured by others (their decode was cpu/pcie-bound, so this
  doesn't settle it for us).
- **inferred for us**: undervolting does nothing for the memory-bound phase (memory clock and voltage are separate
  rails, mvddc). it helps only by letting a higher core lock fit under 400 w, which raises the row budget. worth one
  sweep, after correctness.

### 7. temperature limits under water

| limit | value | source | confidence |
|---|---|---|---|
| gpu thermal target / max (vbios) | 83 c / 90 c | techpowerup vgabios (above) | documented |
| thermal throttle generic | "~85 degrees celsius for most gpus" | tensorrt perf guide (above) | documented (generic) |
| gddr6x case temperature range | tc = 0 c to +95 c / +105 c (commercial) | micron mt61k256m32 8gb gddr6x brief, rev. d 05/2021, features page https://assets.sourcengine.com/datasheets/810969df-46bf-40b2-9bc7-215fdae2d760.pdf | datasheet |
| gddr6x memory-junction throttle on 3080/3090 | 110 c: "the 110 c maximum temperature, as specified by nvidia and micron ... the working memory is now automatically throttled down" | igor's lab alphacool test p.1 https://www.igorslab.de/en/hot-headed-unicorn-under-water-alphacool-eisblock-aurora-acryl-gpx-n-for-nvidia-rtx-3090-founders-edition-installation-and-test/ ; tom's hardware https://www.tomshardware.com/news/hwinfo64-adds-gddr6x-temp-monitoring-rtx30series | measured by others |
| junction sensor accuracy | ±2 c | igor's lab alphacool test p.1 | measured by others |
| fe air cooler, memory | ~100-104 c in games / ethash; ~96 c in games, up to 100 c with memory at ~21 gb/s | igor's lab fe review p.17 https://www.igorslab.de/en/nvidia-geforce-rtx-3090-founders-edition-review-between-even-between-evaluate-and-common-decadence-if-price-is-not-all/17/ | measured by others |
| **3090 fe under water (alphacool gpx-n + backplate)**, port royal loop, +100 core / +1000 mem, 115% pl | delta to water: gpu 19 k, memory junction 44 k. delta to ambient: gpu 46 -> 27 k, hotspot 60 -> 41 k, memory 76 -> 52 k (vs fe air) | igor's lab alphacool test p.5 | measured by others |

**inferred for our box**: under water the gpu core is a non-issue (water + ~19 k). the memory junction is the limit:
44 k above water in a gaming load means ~75-80 c at 30-35 c water, i.e. ~30 k of margin to 110 c. our workload
streams at near peak bandwidth for most of every cycle, which is harder on memory than port royal, and the hottest
chips are likely the 12 on the back under the passive plate. the 44 k figure may grow under a sustained full-bandwidth
stream plus a memory offset. watch it; a fan on the backplate is the cheap fix (plan 1.7 already says so).

**hotspot**: not exposed by nvml on geforce (nvidia-smi documents gpu temperature, t.limit, slowdown/shutdown and
memory temperature only). inferred from the absence in the manual; on the fe under water hotspot ran ~14 k above the
gpu diode (igor p.5: 41 vs 27 k over ambient).

### 8. persistence mode and monitoring on linux

- persistence: `nvidia-smi -pm 1` "the nvidia driver remains loaded even when no active clients ... minimizes the driver
  load latency", "does not persist across reboots", linux only (nvidia-smi manual); nvidia recommends the
  `nvidia-persistenced` daemon. power limits, locked clocks and offsets are also lost on driver unload/reboot (nvml docs
  for `nvmlDeviceSetPowerManagementLimit`, `...LockedClocks`): apply all of them from a systemd unit ordered after
  nvidia-persistenced. documented.
- clocks event reasons (renamed from "clock throttle reasons" in v530): `nvidia-smi -q -d PERFORMANCE` shows the
  performance state plus active reasons: sw power cap, hw slowdown, hw thermal slowdown, hw power brake, sw thermal
  slowdown, sync boost, display clock; "if all event reasons are returned as 'not active' it means that clocks are
  running as high as possible". since v575 there are cumulative microsecond counters:
  `--query-gpu=clocks_event_reasons_counters.{sw_power_cap,sw_thermal_slowdown,sync_boost,hw_thermal_slowdown,hw_power_brake_slowdown}`.
  nvml: `nvmlDeviceGetCurrentClocksEventReasons`. documented (nvidia-smi manual).
- power: `nvidia-smi -q -d POWER` gives "average power draw" (1 s window, ampere except ga100 and newer) and
  "instantaneous power draw", plus current/requested/default/min/max limits. documented.
- clocks and pstate under load: `nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu
  --format=csv -lms 100` or `nvidia-smi dmon -s pucvmet`. `-q -d SUPPORTED_CLOCKS` lists valid (mem, graphics) pairs.
  `-lmi`/`-lmci` reports the memory-lock style (newer drivers). documented.
- **memory junction temperature is not readable via nvidia-smi/nvml on the 3090** ("mtemp" is for hbm parts; nvidia
  forum feature request still open:
  https://forums.developer.nvidia.com/t/request-gpu-memory-junction-temperature-via-nvidia-smi-or-nvml-api/168346;
  strata #32: "memory junction temperature is not readable through nvidia-smi"). the linux workaround is the
  reverse-engineered `gddr6` tool (github.com/olealgoritme/gddr6, fork github.com/AImmd/gddr6), which needs the
  `iomem=relaxed` kernel parameter and reads a gpu register through /dev/mem. measured by others. that parameter
  loosens /dev/mem protection on the host, so it is the owner's call.

---

### implications for the plan

**bring-up settings (per card, root, from a systemd oneshot after nvidia-persistenced; nothing persists on its own):**

1. `nvidia-smi -pm 1` (or nvidia-persistenced). driver default for anything not set below.
2. `nvidia-smi -i N -pl 400`: the fe vbios maximum. plan 1.7's "at or above stock 350 w" is right. take the max,
   because power cut from the cap comes out of the core clock, which sets our row budget (section 2). under water
   there is no thermal reason to stay lower. psu: budget 460-500 w per card for transients.
3. read the operating point before locking anything: run the real decode loop (full row budget, both cards, nvlink
   exchange) unlocked at 400 w and record `clocks.sm`, `clocks.mem`, `pstate`, power, and the `sw_power_cap` event
   counter at 100 ms. the sustained sm clock under the heaviest phase mix is the lock ceiling.
4. `nvidia-smi -i N -lgc F,F` with F a step or two below that sustained clock (expected ~1800-1900 mhz under water
   at 400 w; inferred, not measured). acceptance: `clocks_event_reasons_counters.sw_power_cap` stays flat across a
   10-minute soak. if it climbs, the lock is fake (tensorrt guide: throttling turns a lock back into a floating clock);
   drop F one step.
5. memory clock: check `nvidia-smi --query-gpu=pstate,clocks.mem` under cuda load. **expect p2 and 9501 mhz, not 9751**
   (three independent reports). try in order: (a) `nvidia-smi -lgc F,F -lmc 9751,9751` in one call; (b)
   `nvidia-smi --cuda-clocks=1`; (c) nvidia-modeset `disable_vrr_memclk_switch=1`. keep whichever reads 9751 under
   load with no throughput regression. if none does, accept p2. do not set `CUDA_DISABLE_PERF_BOOST`.
6. the plan's 936 gb/s budget line (section 0.3) assumes p0. in p2 the peak is ~912 gb/s, which adds ~0.4 ms per 16.3
   ms cycle (inferred: 2.56%). either recover it with (5) or with a memory offset, or re-base the budget on 9501.

**what locked clocks buy:** (a) no gpu-boost dither between 15-25 ms cycles, so cycle time depends on the work and
not on the power controller's state; that matters for row-budget tuning and for the a/b measurements in part 2. the
tensorrt guide names sub-20 ms windows as the case where power capping makes measurements unstable. (b) a known
f_core for the t_flops model (t_flops ∝ 1695/f_core). (c) it does not raise throughput by itself. at best it holds the
peak sustainable clock; set above that, the power cap overrides it. no published variance numbers exist for this;
measure cycle-time cv locked vs floating as a phase-b check.

**the p2 trap is real:** geforce ampere runs cuda in p2 with memory at 9501 mhz (19.0 gb/s) instead of 9751 (19.5
gb/s), -2.56% bandwidth. on linux there is no documented switch; `-lmc` is documented to apply under cuda on ampere,
but the one 3090-like report shows it settling at 9501. a memory offset raises the p2 clock as well, so a memory
overclock makes the trap moot: the offset is measured from 9501.

**what limits a memory overclock:**
- **headless linux tooling.** nvidia-settings offsets need an x server. headless, use nvml
  `nvmlDeviceSetClockOffsets` (r555+) or the older `nvmlDeviceSetMemClkVfOffset`. the offset applies to all pstates
  (reported bug, useful here). units differ between tools (one script passes 2x the memory-clock offset), so check the
  resulting `clocks.mem` reading, not the offset value. an offset breaks a `-lmc` lock, so use one or the other.
- **the parts.** micron offers this gddr6x die in 19 and 21 gb/s grades. igor says the 3090's modules "could
  actually run at 21 gb/s" and overclocked to "almost 21 gb/s". inferred: ~+7.7% over 19.5, i.e. 10501 mhz in
  nvidia-smi units, is the in-rating ceiling. strata measured 10,002 mhz under cuda at +500.
- **edc replay.** gddr6x retries link errors rather than returning bad data, so instability shows up first as lost
  throughput (plan 1.7 already sweeps for peak throughput rather than peak clock). the golden-hash soak stays the
  correctness gate.
- **memory junction temperature (110 c throttle, ±2 c sensor)** is the hard thermal limit. it can't be read through
  nvml on the 3090. the gddr6 tool needs `iomem=relaxed`. the 12 back-side chips under the passive plate are the
  likely hot spot. under water, igor saw +44 k over water in a gaming load; a sustained full-bandwidth stream with an
  offset will run hotter. a backplate fan is the cheap mitigation.
- **the power budget is shared.** memory power rises with clock (and mvddc voltage, if the offset raises it), and the
  400 w cap takes it out of the core clock. re-check step 4 (sw_power_cap counter flat) after every memory step.
- **a caution from a measured case.** strata saw no decode gain from +3.1% memory clock, because their loop was bound
  by cpu, pcie and launch latency. ours is designed to be stream-bound, but only a measured cycle-time breakdown
  shows whether the extra bandwidth reaches tok/s (cf. adrien's +15% raw -> +4% tp2 in plan 1.7).

**not established (needs the hardware):** whether `-lmc 9751`, `--cuda-clocks=1` or `disable_vrr_memclk_switch` gets
p0 memory clocks under cuda on a 3090 with current drivers; the sustained sm clock at 400 w for our phase mix; the
nvidia power controller's averaging window (only the 1 s "average power draw" readout is documented); cycle-time
variance locked vs floating; memory-junction temperature under a sustained stream with the xg7's passive backplate;
the exact units of the nvml memory offset on the 3090.

## prior art (prior-art)

What the best persistent-kernel and decode work does, what survives the lossless contract (PLAN 0.2) on sm_86, and
where prior art disagrees with PLAN.md. Confidence labels: **datasheet**, **documented** (the source states it),
**measured by others** (a named repo, post or paper measured it on their hardware), **inferred** (derivation shown).
"Upstream" means geezuz at `32d1204` (local copy at `/Users/bjorn/Documents/geezuz-upstream-32d1204`). "Adrien" means
the vLLM stack on two 5090s at `/Users/bjorn/Documents/qwen3.8-27b-rtx5090`. "HyperQwen" means the vLLM stack at
`/Users/bjorn/Documents/qwen-refs/qwen38-27b-rtx3090`.

### 0. hardware facts the rest leans on

| fact | value | label / source |
|---|---|---|
| dram bandwidth | 936 GB/s | datasheet (GA102) |
| bf16/fp16 mma, fp32 accumulate, dense | 71 TFLOPS (half the fp16-accumulate 142) | datasheet (GA102 whitepaper); the 2.00x gap was **measured by others** on 3090/4090/5090 (exllamav3 source comment) |
| int8 mma, dense | 284 TOPS | datasheet |
| smem per block / per SM | 99 KB / 100 KB of a 128 KB L1 | datasheet |
| L2 | 6 MB | datasheet |
| nvlink NV4 | 56.25 GB/s per direction; 52.6 GB/s measured unidirectional | datasheet; measured by others (NVIDIA forum 232673, cited by the hw-link lane) |
| fp32-accumulate sustained | ~63 TFLOPS (Marlin, 250 W); FA2 hd256 at 54-57 TFLOPS | measured by others (HyperQwen `docs/optimizations.md`) |
| 3090 issue rate for conversion ALU | 64 int32 lanes/clk/SM vs ~6.7 bytes/clk/SM of dram share | inferred: 936e9 / 82 SMs / 1.695e9 clk = 6.7 B/clk/SM; e4m3 to bf16 costs ~1.5-3 int ops per element, so conversion is >10x under the stream |

---

### 1. Hazy Research megakernels

**"Look Ma, No Bubbles" (Llama-1B, bs 1, May 2025)**
([post](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)):

- **Instruction interpreter.** Each SM runs a per-SM instruction queue that is scheduled ahead of time on the host and
  reused across hundreds of forwards. There are 7 instruction types: rmsnorm+qkv+rope, attention, attention
  reduction, o-proj+residual, rmsnorm+up/gate+silu, down+residual, and rmsnorm+lm_head. Each is a load/compute/store
  template (documented).
- **Paged shared memory.** The first 213 KB of H100 smem is cut into 13 pages of 16 KiB. Instructions request and
  release pages, and a released page goes straight to the next instruction, so its weight loads start while the
  previous instruction is still storing (documented).
- **Counters instead of barriers.** Each instruction increments a global counter on completion. A consumer waits for
  a target count. The MLP intermediate is produced in 4 chunks with 4 counters, so the down-projection starts on
  chunk 1 (documented).
- **Measured:**
  - 78% of H100 bandwidth;
  - <1 ms per forward on H100 and ~680 µs on B200;
  - ~2.5x vLLM and >1.5x SGLang on H100 (vLLM and SGLang reach "at most 50%" of bandwidth);
  - launch cost 2.1 µs on a stream, 1.3 µs under CUDA graphs (all measured by others, H100/B200).
- **Where the remaining time goes** (B200, 600 µs per forward, measured by others):

  | share | what |
  |---|---|
  | 250 µs (42%) | storing activations, waiting for consistency, loading them: two load latencies and two store latencies per dependent instruction at ~500 ns each |
  | 200 µs | matvec |
  | 30 µs | waiting on weights |
  | 40 µs | intra-SM barrier overhead (~60 ns per async barrier even when it already passes) |
  | 80 µs | interpreter setup |

  - **Lesson for us:** counters remove the straggler and barrier cost, not the dependency-latency chain.
  - **Inferred for our cycle:** ~340 dependent phases x ~2 µs (store, flag, observe, load) ≈ 0.7 ms per cycle is
    irreducible with counters. That is ~4% of an 18 ms cycle.
- **Hopper dependence:**
  - TMA and mbarrier `expect_tx` loads are Hopper features;
  - the page sizes assume 227 KB of smem;
  - the interpreter, the paging and the counters are plain CUDA.
  - On sm_86 the same design means `cp.async` + mbarrier arrivals with ~4-5 pages of 16-20 KB (inferred from
    100 KB of smem). MPK's A100 configuration is the precedent: 5 pages of 32 KB.
- **Contract:** bf16 weights and activations, fp32 accumulate. Nothing forbidden.

**"We Bought the Whole GPU…" (Llama-70B, TP8 throughput, Sep 2025)**
([post](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)):

- **Warp roles.** Loader, consumer, storer and controller warps. **Storer warps do all cross-GPU traffic**, so loaders
  and consumers move on to the next instruction (documented).
- **Inter-instruction pipelining.** It cuts the gap between matmul stages from 10.2 to 3.4 µs, worth +2.5 to +6.6%
  decode TPS (measured by others, H100).
- **Global work queue** (an atomic instruction counter instead of a static schedule):

  | batch size | effect |
  |---|---|
  | 8,192 | +14.2% |
  | 4,096 | +6.3% |
  | 1,024 | **−1.9%** |

  Measured by others. Their own explanation: at small work sizes the atomic's latency is not hidable. **For us** (~24
  rows, µs-scale instructions) a static compile-time schedule is the right choice, as PLAN says.
- **Interleaving instruction types across SMs** ("quilts", NanoFlow-style): +6.4% at batch 8,192, 0% at 1,024-2,048
  (measured by others). Irrelevant at our size.
- **Replicated o-proj** to shrink communication 8x (documented). It cost ~15% of batch capacity. This is the same
  trade as PLAN 2.5's replicated drafter.
- **Result:** +22% end-to-end over SGLang on ShareGPT (measured by others).
- **Related posts:**
  - ThunderKittens 2.0: "a few loose fence instructions caused over 10% loss" in the megakernel (documented).
  - PGL: in-kernel peer loads and stores over NVLink hide the comms behind GEMM tiles (documented, H100 NVSwitch).
  - "Retire the abstractions" (Aug 2026): their 2026 MoE megakernel was written without the interpreter framework,
    "with agents". The ideas stay and the template library goes (documented). That matches PLAN's choice of a
    generated compile-time schedule over a general interpreter.

### 2. Mirage Persistent Kernel (MPK)

Source: [arXiv 2512.22219](https://arxiv.org/html/2512.22219v1), OSDI '26.

- **Compiler.** It lowers a PyTorch graph into an SM-level task graph (tasks alternate with events) (documented).
  - An event is created only where a producer's output region overlaps a consumer's input region.
  - Events are fused by identical successor or predecessor sets. The graph is normalized to fan-in/fan-out of 1 and
    linearized by BFS, so each event's tasks get contiguous indices.
- **Runtime.**
  - Worker SMs: one FIFO task queue each.
  - Scheduler warps: 4 SMs x 4 warps = 16. This costs 4 of 108 SMs on A100 (documented).
  - **Hybrid launch.** Ahead-of-time (AOT) tasks are pre-enqueued and need one sync (worker to worker). Just-in-time
    (JIT) tasks need two (worker to scheduler, scheduler to worker). Data-dependent operators (attention) are JIT
    until the next global barrier; everything else is AOT (documented).
  - Smem pages are 32 KB: 5 pages on A100.
- **Measured** (by others):

  | result | value |
  |---|---|
  | Qwen3-8B on A100, per-token latency | 14.5 ms (vLLM/SGLang) to 12.5 ms, against a 10 ms bandwidth bound: 80% of nominal bandwidth, +16% |
  | single batch overall | 1.0-1.7x over vLLM and SGLang |
  | 8x H100 | 1.1-1.4x |
  | compute-communication overlap (Qwen3-1.7B, 4x H100) | 1.1x |
  | cross-task pipelining on the final linear | 1.2-1.3x |

- **Ampere:** it runs on A100 (sm_80), so the design needs no Hopper feature.
- **Lesson for us:**
  - Use AOT/static assignment for the weight stream.
  - Use dynamic (JIT-style) assignment only for attention, whose cost depends on context length. That is also
    FlashInfer's planner (section 3).
  - A dedicated scheduler SM is not worth 1 of 82 SMs for a fixed graph (inferred: it costs 1.2% of bandwidth-issuing
    SMs and buys nothing on a static graph).

### 3. Flash-decoding, FlashInfer, split-k, cascade

- **Flash-decoding** ([PyTorch blog](https://pytorch.org/blog/flash-decoding/), Oct 2023).
  - Mechanism (documented): split the KV range across blocks. Each split writes its output plus a log-sum-exp (LSE),
    and a merge rescales by exp(lse_i − lse_total).
  - Results (measured by others, A100): up to 8x end-to-end on CodeLlama-34B; attention up to 50x over FA2;
    B=1 at 64k context in 64 µs vs 2,300 µs.
  - Hardware and contract: no Hopper feature, fp32 merge. It ports as is.
- **HyperQwen on the 3090** (measured by others). FA2 does not split KV when a request has 5 query rows: a 24-head
  model runs attention on 24 of 82 SMs.
  - 1.5k context: 57 µs per layer, 23 µs with split-KV.
  - 16k context: 1.3 ms per layer, 120 µs with split-KV.
  - **Verify attention must be split-KV at any row count.** PLAN has it.
- **FlashInfer** ([paper](https://arxiv.org/html/2501.01005), MLSys 2025; [cascade blog](https://flashinfer.ai/2024/02/02/cascade-inference.html)).
  - **Attention state and merge.** The state is (v, LSE) with an associative, commutative merge ⊕ (documented). It is
    exact up to summation order, so it is inside our contract.
  - **GQA packing.** Query heads are fused into the row dimension, so one KV load serves the group (documented). For
    us: 6 q heads per kv head x up to 5 verify rows = 30 rows, two m16 tiles.
  - **Load-balanced planner** (documented).
    - Deterministic and Stream-K-like, with no atomics.
    - Tile cost = α·l_q + β·l_kv. Only requests longer than total/#CTA are split.
    - Planned on the CPU once per step and reused by every layer.
    - Attention and merge run in one persistent kernel.
  - Load-balanced planner ablation (measured by others, H100, Llama-3.1-8B): inter-token latency 13.89 to 8.63 ms
    with skewed lengths, 9.16 to 8.96 on ShareGPT.
  - **Cascade (shared-prefix) attention** (measured by others, H100): up to 31x over vLLM paged attention at a 32k
    shared prefix and batch ≥128. Composable shared-prefix kernels at batch 64 with a 32k prefix: 4,090 to 254 µs.
  - sm_86 is supported through the FA2 path, which covers decode, prefill, cascade and the scheduler (documented).
- **llama.cpp FA** (master `36d818e`, documented in source).
  - It uses MMA kernels with stream-k for tile and MMA (non-vector) attention, and split-KV (`parallel_blocks`) for
    vector attention. Both are chosen by a ≥95% wave-efficiency scan, and both merge (VKQ, max, rowsum) partials.
  - On a 3090 with unquantized KV and a GQA mask, even 1-token decode takes the MMA kernel with GQA rows packed.
- **What survives:** all of it. It needs no Hopper feature, and the LSE merge is a reordering.
  - **One contract issue, inferred.** Every flash-attention implementation, including upstream's, rounds P to 16 bits
    before the PV mma. The cpu reference keeps P in fp32, so a bf16 P is a rounding change, not only a reordering.
  - Two options:
    1. Compute PV as two bf16 mmas with P split into hi + lo. That is cheap: decode attention is memory-bound, so the
       extra mma rides under the KV stream.
    2. Have the contract name P an "activation".
  - PLAN must pick one (section 9).

### 4. llama.cpp CUDA backend

Source: [ggml-cuda](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda), read at master
`36d818e`. All documented in source.

- **mmvq/mmq.** These quantize activations to q8_1: int8 via dp4a or int8 mma.
  - mmvq is used up to 8 columns (per-arch caps: 4-7 on 4090/5090 k-quants). A "few-row mmq" path is used up to 24
    columns. Larger widths use mmq with stream-k and a fixup buffer.
  - **Forbidden** (int8 activations). Only the scheduling carries over: stream-k with fixup, and the wave-efficiency
    heuristics.
- **bf16 path.** Order: mmvf (CUDA cores, ≤5 columns on Ampere), then mmf (tensor-core `mma.sync`, ≤16 columns), then
  cuBLAS. So llama.cpp switches to tensor cores for bf16 at ≥6 rows. That is consistent with PLAN's single
  tensor-core gemv for 1 to R rows (R up to 24).
- **FP8.** `GGML_TYPE_F8_E4M3` exists, but the fused path is W8A8 cuBLASLt, gated to compute capability ≥ Ada. On
  sm_86, fp8 weights are converted to a dense buffer and then go through cuBLAS. **There is no w8a16 fp8 kernel for
  Ampere in llama.cpp.**
- **Gated DeltaNet** (`GGML_OP_GATED_DELTA_NET`; PR numbers from search snippets only).
  - **Autoregressive kernel:**
    - fp32 state stored transposed;
    - one warp segment per value column;
    - per token: kv = Sᵀk, δ = β(v − g·kv), S = g·S + kδᵀ, o = Sᵀq;
    - "replay" of pending tokens, bit-identical to the sequential path, plus state snapshots for spec rollback.
  - **Chunked kernel** (`gated-delta-net-mma.cu`):
    - chunk 16, 256 threads;
    - fp32 state, with every product as **three bf16 mmas (hi·hi + hi·lo + lo·hi)**, i.e. split-bf16 emulation of
      fp32;
    - forward-substitution inverse;
    - value-column splitting when heads x sequences x 2 ≤ #SM;
    - enabled explicitly for GA10X (sm_86).
  - **Lesson:** the bf16x3 split is the **only prior art that runs a chunked deltanet on tensor cores without
    rounding the fp32 state**, so it is the template for PLAN 1.6's prefill deltanet. Value-column splitting is the
    template for decode at 8 slots (section 9, flag 3).
  - No 3090 tok/s figure could be retrieved; the PR pages were unreachable.
- **Graph capture.**
  - On by default (cc ≥ Volta), with capture after 2 unchanged warm-up calls.
  - NVIDIA blog: up to 1.2x on H100 for Llama-7B at batch 1 (measured by others).
  - Irrelevant to the persistent decode kernel. For prefill launches the saving is ~1-2 µs per kernel (Hazy's
    numbers), which is noise against a ~1 s chunk (inferred).

### 5. fla gated-deltanet kernels

Source: [fla-org](https://github.com/fla-org/flash-linear-attention), read as vendored in vLLM v0.31.0. All
documented in source.

- **Chunked WY form**, chunk 64:
  1. cumsum of g;
  2. A = β·KKᵀ (gated, strictly lower), fp32;
  3. (I+A)⁻¹ by forward substitution on 16x16 diagonal blocks in fp32, merged to 64 with `tl.dot` (precision "ieee"
     by default, with tf32 options);
  4. W = A⁻¹(βk·eᵍ), U = A⁻¹(βv);
  5. inter-chunk recurrence with fp32 accumulators;
  6. output.
- **Precision losses in fla.**
  - It stores A⁻¹ and the per-chunk h in **bf16**.
  - It computes `v_new = U − W·Sᵀ` with **S cast to bf16**.
  - **Forbidden under PLAN 0.2** (bf16 deltanet state).
- **Decode** (`fused_recurrent_gated_delta_rule`): fp32 state in registers. The state is stored back in the cache dtype.
- **vLLM defaults.**
  - vLLM's `mamba_ssm_cache_dtype` "auto" resolves to bf16 for qwen3-next/3.5.
  - Adrien serves bf16 state: it bought pool size, cost 0.05-0.1 points on his rulers, and measured 1,268 vs 1,289 at
    8 streams (measured by others).
  - HyperQwen serves fp16 state: +all 64 requests resident (measured by others).
  - **Both are forbidden for us.**
- **Upstream's prefill deltanet** (`prefill.cuh` line 189, documented): "the slice of S lives in fp32 MMA
  accumulators; it becomes an **fp16 B operand** in registers".
  - That is a 16-bit rounding of the state at every chunk. It is not inside PLAN's contract either. A straight port
    would carry it, so step 1 must use the bf16x3 split (section 4).
  - Upstream measured the deltanet at ~2% of prefill time, so 3x on that part costs ~4% of prefill (inferred).
- **Gated DeltaNet paper** ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)): it gives the UT transform and
  training throughput only, with no inference numbers (documented).

### 6. exllama family and Marlin (layout and scheduling only)

- **Marlin** ([arXiv 2408.11743](https://arxiv.org/html/2408.11743)), documented:
  - weights are pre-shuffled offline so each thread does one 16-byte load that lands as an mma fragment, and 8
    threads cover a 128 B line;
  - a multi-stage `cp.async` pipeline with **evict_first** on weights;
  - XOR-swizzled activations for conflict-free `ldmatrix`;
  - dequantization of the next B operand overlapped with the current mma;
  - striped (stream-k-like) partitioning with serial in-order global reduction. That keeps it deterministic, and vLLM
    uses fp32 reduction by default.
  - Measured by others, Llama-2-7B 4-bit on **one 3090** vs fp16: 2.69x at batch 1, 2.43x at 8, 2.30x at 16, 1.84x
    at 32. At batch 16 it still runs at ~85% of its batch-1 speed-up.
  - Inferred for 8-bit weights: the ideal is ≤2x, and the bandwidth-bound regime extends to roughly twice the batch.
- **fp8 Marlin in vLLM** ([PR 5975](https://github.com/vllm-project/vllm/pull/5975)): **exactly our w8a16 e4m3
  format on Ampere**. Documented:
  - conversion is pure bit ops: two e4m3 per 16-bit lane, `(q & 0x80008000) | ((q & 0x7F007F00) >> 4)`, then a shift
    for the other pair;
  - the 2^120 exponent bias is applied by an `hmul2` or folded into the scales;
  - `atomicAdd` with bf16 is disabled on sm8x, and the reduction is fp32;
  - W8A8 is refused.
  - Its benchmark figures are image-only and could not be retrieved.
  - Correctness note, inferred: **if the 2^120 bias is folded into the fp32 block scale**, the raw bf16 operands sit
    ~2^-120 small. Their products with activations can fall into fp32 denormals, which the tensor core may flush.
    Apply the bias before the mma. PLAN already requires an exhaustive test over all 256 codes; that test must run
    through the mma, not only through the conversion.
- **Machete is sm_90 only** (documented). Not applicable.
- **exllamav3** (local copy `@5783a93`, documented in source).
  - Marlin-style striped stream-k: grid = #SM slices, k-fastest, serial lock-ordered column reduction.
  - Tile shapes (16 x 16..32 x 128..512) chosen per (N, K) by a table plus an autotune hash.
  - fp16 accumulate with an fp32 fold every 32 K, ~14% faster at batch 1 on a 3090 (measured by others).
    **Forbidden** (fp16 accumulate).
- **HyperQwen**: retuning Marlin tiles for M ≤ 16 on sm_86 gave 3-7% per GEMM in isolation and nothing end to end:
  "the remaining gap to peak bandwidth is the memory system's ramp on 16-92 MB reads, not the kernel" (measured by
  others). Per-operator launches pay that ramp every time. **A continuous stream pays it once per cycle.** That is the
  mechanism behind PLAN's step 2.

### 7. Upstream geezuz and open-jet

**Structure** (`qwen38_mk.cu`, `batch.cuh`, documented in source):

- **Launch.** One cooperative persistent kernel per cycle: 82 blocks x 512 threads (16 warps), 1 block per SM, smem
  ≤ 99 KB.
- **Barriers.** The ~340 phases are separated by a software grid barrier (`red.release.gpu.add` on arrival, then an
  `ld.acquire.gpu` poll); there are 42 `gsync` call sites.
- **Phase order per layer.** Projections, then core (deltanet or attention), then combine, then the residual gemv
  with the cross-card sum, then FFN gate/up, then the FFN down residual. lm_head and top-k sampling follow, then MTP
  catch-up passes and the draft steps.
- **Prefetch.** Each phase prefetches the next phase's first weight rows into L2 before the barrier.
- **Tensor parallel 2.**
  - Each card holds half of every matrix.
  - Partials go peer to peer over PCIe with a flag per item (`st.release.sys` and `ld.acquire.sys`, with
    generation-tagged flags so no per-launch reset is needed).
  - Both cards add in the same order, so the residual is bit-identical on both.
- **Decode.**
  - Lazy-replay deltanet. Per (slot, value head) item, the state is read once, the accepted drafts from the ring are
    replayed, the T verify rows are computed in registers, and the committed state is written.
  - Split-K attention: up to 32 chunks per kv head, 32-position K/V tiles double-buffered on fp16 tensor cores.
  - Weights in GGUF encodings: dp4a gemv up to 10 rows, int8 `mma.m16n8k32` for 16/24/32 rows.
- **Host.** A python server; ~125 µs of host time per batched cycle plus ~55 µs of in-kernel block-table lookups,
  both "open" (documented, CHANGELOG v0.3.0).
- **open-jet** is the one-GPU origin of all of this (L-Forster). One 3090, Q4_K_M: 106.7 tok/s vs llama.cpp 63.1 on
  4 prompts, and 42.9 vs 40.4 tok/s without drafts (measured by others).

**Measured numbers** (two 3090s at 280 W, PCIe 4.0 x8, no NVLink; measured by others):

| config | ms per cycle | tok/s |
|---|---|---|
| Q8_0, 1 slot (≤5 rows) | 24.9 | 143-144 |
| Q8_0, 4 slots (16 rows) | 33.2 | 407 |
| Q8_0, 5 slots (32 rows) | 38.4 | 482 |
| Q8_0, 8 slots (32 rows) | 43.8 | 610 |
| Q4_K_M, 1 slot | 19.6 | 190 |
| Q4_K_M, 8 slots | 42.7 | 625 |
| one GPU, Q4_K_M, no drafts | — | 42.9 |

**Where the cycle goes**, Q4_K_M, 8 slots, 32 rows, ms per cycle (CHANGELOG build za, measured by others):

| phase | ms |
|---|---|
| FFN down incl. cross-card sum | 9.94 |
| DeltaNet projections | 8.30 |
| FFN gate/up | 8.15 |
| everything else (embed, attention core and combine, lm_head, sampling, MTP) | 4.46 |
| DeltaNet output incl. sum | 4.24 |
| DeltaNet core | 3.83 |
| attention projections | 2.34 |
| attention output incl. sum | 1.28 |
| **cycle** | **42.52** |

Derived readings (inferred):

- **Bandwidth fraction at 1 slot, Q8_0.**
  - Bytes per card ≈ (28.7 GB file − 1.35 GB token_embd kept on the host) / 2 ≈ 13.7 GB per cycle, so the floor is
    14.6 ms at 936 GB/s.
  - Measured 24.9 ms, so **~57% effective**. That is the barrier design's real efficiency, including the serial work.
- **One-GPU, no drafts.**
  - 42.9 tok/s x ~15.5-16 GB (16.7 GB of weights minus embeddings minus the MTP head) ≈ 665-690 GB/s, i.e.
    **~71-74%** for a pure bandwidth-bound sm_86 megakernel.
  - The byte count is approximate: it depends on which tensors were resident.
- **Exchange.** The three summing phases fell from 24.8 to 15.8 ms per cycle through bf16, then int8, partials. One
  residual call takes ~24 µs from written to sent, against a 14 µs transfer floor at 13.2 GB/s (documented).
  Removing a full system fence: 28 to 24 µs per call (documented).
- **Per-slot cost.**
  - From 5 to 8 slots at a fixed 32 rows, the Q8_0 cycle grows (43.8 − 38.4) / 3 ≈ **1.8 ms per added slot**. That is
    per-slot serial work: deltanet core, attention, sampling, MTP, rows exchanged.
  - From 4 slots/16 rows to 5 slots/32 rows it grows 5.2 ms. With ~1.8 ms of that being the slot, int8 mma rows cost
    **≈0.2 ms per row** in that design.
- **DeltaNet core.** 3.83 ms / 48 layers ≈ **80 µs per layer at 8 slots**. The state-bandwidth floor is 27 µs per
  layer: 8 slots x 24 heads x 64 KB read and written = 25 MB at 936 GB/s. The core runs at ~3x its floor:
  192 items on 82 blocks is 2.34 waves, rounded up to 3.

### 8. The two vLLM stacks

**Adrien: two 5090s, TP2, W4A4 NVFP4, NVFP4 KV, bf16 GDN state, MTP 3** (measured by others):

- **Throughput.** 263.9 / 215.9 tok/s code / prose at 1 stream; 2,648.6 / 2,239.7 aggregate at 16 streams;
  acceptance 0.61-0.68 code, 0.46-0.49 prose (README, 2026-10-20).
- **Decode-step profile** (R183, vLLM 0.29, PCIe 5.0 x8 custom all-reduce):

  | streams | ms per step | of which | all-reduce share | other |
  |---|---|---|---|---|
  | 1 | 18.4 (14.4 busy) | GEMM 7.1 ms; 265 GEMM calls | 2.7 ms (143 calls x 19 µs) = 15% | ~1,400 small kernels; 22% inter-kernel gaps |
  | 8 | 24.9 | all-reduce 49 µs per call | — | — |
  | 16 | 38.1 | all-reduce 90 µs per call | 33% | — |

  Bytes per all-reduce are rows x 5,120 x 2.
- **All-reduce kernels** (R184 microbenchmark, graph mode):
  - FlashInfer `pcie_ipc`: 5.6 µs at 10 rows, 33.5 µs at 80, 65.3 µs at 160.
  - vLLM custom: 8.8 / 49.0 / 86.0 µs.
  - All sit at the PCIe Gen5 floor (22-26 GB/s per direction) at 84 MB.
  - In-engine: the 1-stream step fell 11.51 to 10.95 ms (−4.9%); single-stream decode rose +4.6 to +5.4%, and
    +2.6% at 16 streams.
- **Fused all-reduce + residual + RMSNorm:** 9-19 µs saved per call in isolation, nothing end to end (R190b/e). The
  win was absorbed by vLLM's other overheads.
- **Drafter.** The 3 MTP draft passes take 1.74-1.81 ms of a 10.95 ms step, **~16%**. The drafter at tensor
  parallel 1 (replicated) read 308 vs a 285-308 replicate band: **null** (R183 "SP-dtp1").
- **Bandwidth.**
  - Memory overclock: +15% raw d2d bandwidth (1,422 to 1,641 GB/s), but **+4% TP2 decode**, "co-bound by
    allreduce latency and drafter compute" (R136-138).
  - The measured 1,422 GB/s at stock is **79% of the 1,792 GB/s datasheet** for a d2d copy.
- **Rejected:** suffix decoding as an MTP replacement, 31.2 vs 177-190 tok/s code. DFlash: no batch scaling, c4 ≈ c1.
- **Power:** a 400 W cap read −3.5 to +4.7% decode (noise) and −6.2 to −12.6% cold prefill.

**HyperQwen: one 3090, vLLM, W4A16 AutoRound (+int8 activations in batch mode), fp8/int8 KV, fp16 GDN state**
(measured by others):

- **Headline figures:**
  - 127 tok/s single stream (DFlash2 + context lookup);
  - 381 tok/s while reproducing its prompt with a 16-token verify block filled from the context;
  - ~1,035 tok/s aggregate decode at 64 concurrent (128 in / 512 out).
- **Batch-mode build-up at 64 concurrent:**

  | step | steady-state decode tok/s |
  |---|---|
  | int8 lm_head and embeddings | 585 |
  | + fp16 GDN state (all 64 resident; it was 37 before) | 830 |
  | + int8 activations on the MLP | 1,094 |
  | + int8 activations everywhere | 1,222 |

  - Their stated reason: at 40-64 sequences the step is **bound by fp16 tensor-core math (~63 TF sustained at 250 W)**.
- **Two 3090s with NVLink, batch profile, TP2, no spec:** 1,439 tok/s decode at 64 concurrent, +39% over one card.
  C1 single-user peaks at 182.8 tok/s (DFlash2 W4A16, issue #159).
- **"NVLink appears to buy little"** at C1 (an NVLink box at 330 W vs PCIe x8 at 275 W, within a few %).
- **Spec vs batching:** "speculation wins below ~8 concurrent users, plain batching above".
- **Per-resident cost on one card:** every resident request adds ~5 ms (MTP) to ~7 ms (DFlash2) to the forward pass,
  so per-stream decode runs 126 / 103 / 46 tok/s at 1 / 2 / 4 streams.
- **Draft vocabulary.** A draft vocab counted over the model's own outputs (97.5% coverage vs 92% from web text) was
  worth +10% single stream.
- **Prefill.** W4A16 prefill reaches ~1,437 tok/s at 1k on one card. With `INT8_ACT` it reaches 1,845, and int8
  QKᵀ attention adds a few % more. The int8 parts are forbidden.

What these imply for us is in the flags (section 9).

---

### 9. Where prior art contradicts or qualifies PLAN.md

1. **Bandwidth fraction: 88% vs 70-80%.** PLAN 0.3 budgets the stream at ~88% of 936 GB/s.

   | measured by others | fraction of peak |
   |---|---|
   | Hazy, H100 | 78% |
   | MPK, A100 (12.5 ms vs a 10 ms bound) | 80% |
   | upstream one-GPU megakernel (inferred from 42.9 tok/s) | ~71-74% |
   | upstream two-GPU at 1 slot, including serial work | ~57% |
   | a 5090 d2d copy (Adrien) | 79% |

   No prior art on any GPU shows a decode kernel above ~80%.
   - At 78% (inferred): the main pass is 13.46 GB / (0.78 x 936) = 18.4 ms, not 16.3; the drafter adds 1.85 GB, i.e.
     2.5 ms.
   - Step 2's 1-stream cycle then lands at ~21-22 ms, not ~18. That gives code ~205-215 tok/s (τ 4.46) and prose
     ~100 (τ 2.16).
   - Treat 88% as a stretch target, not a budget.
2. **DeltaNet state traffic is missing from the 0.3 budget.**
   - Each live slot reads and writes 75.5 MB of fp32 state per card per cycle (upstream reads once and writes once
     under lazy replay).
   - That is 151 MB per slot. At S = 8 it is **1.21 GB per card per cycle ≈ 1.3 ms at 936 GB/s** (inferred), ~9% on
     top of the 13.46 GB main pass.
   - The 0.3 table should carry it next to the kv reads.
3. **"Serial phases under the weight stream" cannot hide the 8-slot serial phases.**
   - PLAN 1.6 claims ~15 µs of run-ahead, and that is an upper bound. On sm_86, 82 SMs x ~64-80 KB of ring plus 6 MB
     of L2 is 11-12.5 MB, i.e. 12-13 µs (inferred). Weights marked evict_first do not all stay in L2.
   - Upstream's deltanet core alone takes **~80 µs per layer at 8 slots** (section 7). Even at its 27 µs state floor
     it exceeds the run-ahead.
   - Hazy and MPK only hide sub-10-µs bubbles; neither claims to hide a 30-80 µs dependent phase.
   - So at S = 8 the serial phases must be made **fast**:
     - deltanet value-column splitting across 2-4 SMs per head (llama.cpp's scheme);
     - state reads issued by the producers as a dependency-free stream, like weights;
     - attention tiles load-balanced by a FlashInfer-style plan.
   - They cannot be made **invisible**. At S = 1 (~10 µs per layer) the PLAN claim holds.
4. **Upstream's slot-count curve contradicts the flat 8-stream estimate.**
   - PLAN 1.9 implies the 8-stream step-2 cycle is close to the 1-stream cycle: ~1,000 tok/s at ~18 ms is ~2.25
     tokens per slot.
   - Upstream measured ~1.8 ms of extra cycle per added slot at a fixed row count (section 7).
   - Inferred estimate:
     - step 2 cuts per-slot cost to ~0.6-0.9 ms (state at its 0.16 ms floor, parallel deltanet core, nvlink
       exchange);
     - the 8-slot cycle ≈ 21.5 ms (1 slot) + 7 x 0.75 + exposed extra-row compute ≈ 27-30 ms;
     - aggregate ≈ 8 x 2.5 (τ with 3 rows per slot on code) / 0.0285 s ≈ **650-750 tok/s code**, not ~1,000.
   - Measure it in phase b with `MK_PROFILE=batch` on upstream Q8_0, at 1 to 8 slots, with and without the bridge.
5. **Step-1 cycle of ~23 ms looks optimistic.** Upstream Q8_0 moves ~13.7 GB per card per cycle in 24.9 ms. Our
   image moves ~15.3 GB (body 12.19 + bf16 lm_head 1.27 + MTP 1.2 + draft head 0.65). At equal efficiency step 1 is
   ~27.8 ms, minus ~1-2 ms for NVLink instead of PCIe exchange, i.e. **~25-27 ms** (inferred). That is code
   ~165-180 tok/s, not ~190.
6. **Prefill at ~2,000 tok/s needs in-situ GEMMs at ≥75% of the fp32-accumulate peak.**
   - Upstream's fp16-accumulate GEMMs bench at 100-108 TF, but its end-to-end 1,872 tok/s (2 cards, 3.6k context)
     implies only ~53 TF per card in situ: 48.6 GFLOP/token x 1,872 / 2 cards, with GEMMs at ~85% of time.
   - That is about **half its microbench rate**.
   - fp32 accumulate halves the mma peak (71 TF, 2.00x measured). Inferred range for us:

     | in-situ GEMM rate | prefill tok/s |
     |---|---|
     | same in-situ fraction as upstream | ~1,000-1,300 |
     | 55-60 TF per card, exchange hidden | ~1,900-2,000 |

   - HyperQwen's 1,437 tok/s on **one** card implies ≥~70 TF: 48.6 GFLOP x 1,437. That is at or above the
     fp32-accumulate peak, so its path is fp16-accumulate-class or counts tokens differently (inferred; not verified).
     **Do not use it as evidence for bf16-to-fp32.**
7. **HyperQwen's 64-way concurrency vs S = 8.** Its ~1,035 tok/s (one card) and 1,439 (two cards, NVLink) at 64
   concurrent rest on int4 weights, int8 activations and fp16 state, **all forbidden**. Its own data supports PLAN's
   regime:
   - "speculation wins below ~8 concurrent";
   - at 40-64 rows it is tensor-core bound even with fp16 math;
   - GDN state, not KV, bounded its concurrency (37 of 64 resident with fp32 state).

   Under our contract, more slots make sense only up to the row-budget crossover, and each slot costs ~0.16 ms of
   state traffic plus its serial phases. A slot row is worth ~1 token per (0.5 ms compute + 0.16 ms state). A draft
   row is worth p ≈ 0.6-0.9 tokens per 0.5 ms (inferred). The two are close, which is why 2.4's S sweep matters. No
   evidence favours S ≫ 8.
8. **Memory overclock +8% vs Adrien's measured +4%** at TP2. His decode was co-bound by vLLM's all-reduce and drafter.
   A continuous stream should pass more of the +15% through, but upstream's 57% efficiency says most of the cycle is
   not bandwidth today. Expect +4-8% until step 2 is in (inferred).
9. **NVLink.** HyperQwen's "NVLink buys little" was measured on vLLM's latency-bound bf16 all-reduces at C1. It does
   **not** carry over to fp32 partials at 24 rows:
   - each residual call per card is 24 x 5,120 x 4 B = 492 KB;
   - over PCIe 4.0 x8 at the 13.2 GB/s upstream measured: 37 µs per call, 4.8 ms per cycle over 128 calls;
   - over NVLink at 52.6 GB/s: 9.3 µs per call, 1.2 ms per cycle, mostly overlapped by per-tile push (inferred).

   PLAN's fp32-over-NVLink choice is right. At 1 slot, 5 rows: 102 KB per call, so 1.9 µs vs 7.8 µs.
10. **Replicated drafter (PLAN 2.5) likely loses.**
    - It saves ~10 serial exchanges x ~2-5 µs = 20-50 µs per cycle against +~1.3 ms of extra bytes (inferred).
    - Adrien measured drafter TP1 as a null (R183).
    - Kill it on the first measurement, or skip it.
11. **"MTP steps overlap the next cycle's layer-0 prefetch"** (1.6) can buy at most the run-ahead, ~12 µs per cycle
    (inferred). Next-cycle compute depends on the accepted tokens. Harmless, but it is not a lever.
12. **Contract gaps a straight port would carry** (documented in upstream source; inferred as contract violations):
    - fp32 deltanet state rounded to an fp16 mma operand in chunked prefill. Fix with the bf16x3 split, ~+4% prefill.
    - P rounded to fp16 before PV in both decode and prefill attention. Fix with a hi/lo split, or name P an
      activation in 0.2.
    - fp16 activations and fp16-accumulate GEMMs throughout prefill. PLAN already culls these.

    PLAN 1.6 step 1 says "straight port", so it should name all three.
13. **Power.** A memory-bound sm_86 megakernel loses little power-limited: Lucebox on Qwen3.5-0.8B, one 3090, read
    411 tok/s at 220 W vs 433 at 420 W, i.e. 95% (measured by others). Adrien's 400 W cap was noise on decode, −6 to
    −13% on prefill. PLAN's ≥350 W matters for prefill and for 24-row cycles, not for 1-slot decode.

### 10. A free scheduling lever not in PLAN: defer dependency-free weights into serial phases

Inferred from the model shapes. In a deltanet layer, `in_proj_z` (per card 5,120 x 3,072 fp8 = 15.7 MB, ~17 µs of
stream) is consumed only after the core, by norm(o)·silu(z). In an attention layer, the gate half of `q_proj`
(another 15.7 MB) is consumed only after attention.

- Streaming those matrices **while the core runs** fills the serial phase with real weight traffic:
  48 x 17 µs + 16 x 17 µs ≈ **1.1 ms per cycle** of otherwise-exposed time, ~5% of a 21 ms cycle.
- It needs no numerics change, only stream order in the bake and the schedule.
- The deltanet state read (also dependency-free) belongs in the same producer stream.

**Sources:**

- Hazy Research:
  - ["Look Ma, No Bubbles"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
  - ["We Bought the Whole GPU"](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
  - [PGL](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)
  - [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)
  - ["Retire the Abstractions"](https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions)
- [MPK, arXiv 2512.22219](https://arxiv.org/html/2512.22219v1)
- Attention:
  - [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)
  - [FlashInfer, arXiv 2501.01005](https://arxiv.org/html/2501.01005)
  - [FlashInfer cascade](https://flashinfer.ai/2024/02/02/cascade-inference.html)
- [llama.cpp ggml-cuda](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda), read at `36d818e`
- Deltanet:
  - [fla](https://github.com/fla-org/flash-linear-attention), as vendored in vLLM v0.31.0
  - [Gated DeltaNet, arXiv 2412.06464](https://arxiv.org/abs/2412.06464)
- Weight layouts:
  - [Marlin, arXiv 2408.11743](https://arxiv.org/html/2408.11743)
  - [vLLM fp8 Marlin, PR 5975](https://github.com/vllm-project/vllm/pull/5975)
  - exllamav3 local copy `@5783a93`
- [Lucebox megakernel (Show HN)](https://hn.jakechampion.name/item/47691182)
- Local repos:
  - upstream geezuz: `README.md`, `CHANGELOG.md`, `qwen38_mk.cu`, `batch.cuh`, `prefill.cuh`
  - Adrien: README, `docs/DESIGN.md`, `docs/GOTCHAS.md`, `docs/REJECTED.md`, `docs/FIDELITY.md`, `bench/RESULTS.md`,
    `bench/results/r183`, `r184`, `r27-draftshare`, `r136-r138`
  - HyperQwen: README, `docs/optimizations.md`, `docs/multi-gpu.md`, `docs/benchmarks.md`

### 11. Techniques ranked by expected gain on this hardware under the lossless contract

Gains are against the step-1 straight port. S1 = 1 slot (5 rows, cycle ~25-27 ms); S8 = 8 slots (24 rows, cycle
~40 ms). All gains are **inferred** from the arithmetic shown.

| # | technique (prior art) | expected gain, arithmetic | in PLAN? | part |
|---|---|---|---|---|
| 1 | continuous warp-specialized weight stream: `cp.async` ring, per-tile counters, no grid barriers (Hazy, MPK) | S1 **+20-25%**: 13.7-15.3 GB at 57% → 75-78% effective, i.e. 26 → ~21 ms; S8 smaller, ~+8-12%: the barrier and straggler share of 340 phases x ~3-5 µs ≈ 1-1.7 ms plus the ramp | yes (1.6 step 2) | 1 |
| 2 | fast per-slot serial phases: deltanet value-column split (llama.cpp), state as a producer stream, balanced split-KV attention plan (FlashInfer Alg. 1) | S8 **+10-15%**: deltanet core 3.8 → ~1.5 ms (27 µs/layer floor + latency) and attention/other ~4.5 → ~2.5 ms, ≈4 ms of ~40; S1 ~+1-2% | no: PLAN relies on hiding them, which the run-ahead cannot do at S8 (flag 3) | 1 (necessary for the 8-stream number); size measured in phase d |
| 3 | cascade / shared-prefix attention with exact LSE merge (FlashInfer) | S8 with a shared 30k prefix: kv 8 x 0.98 GB = 7.9 GB → 0.98 GB per card per cycle, saving ~7.4 ms ≈ **+15-30%**; 0 without sharing | yes (2.8) | 2 |
| 4 | context-lookup draft branches, extra to MTP, point-mass proposals (HyperQwen) | measured by others on one 3090: +47-55% while reproducing the prompt, +2-3% on prose, 0 on fresh text; Adrien's replacement variant lost 83% | yes (2.2) | 2 |
| 5 | memory overclock | +15% raw bandwidth (Adrien) → +4% measured at TP2; +6-8% once item 1 lands | yes (1.7) | 1, gated by the soak |
| 6 | defer dependency-free weights (`in_proj_z`, the q-proj gate half, deltanet state reads) into serial phases (section 10) | ~1.1 ms of ~21-30 → **+4-5%** at S1 and S8 | no | 1 (stream order only) |
| 7 | wave-quantization-aware tiling / split-k: ≥95% wave efficiency, stream-k fixup (llama.cpp, exllamav3, Marlin striping) | upstream pair-split measured 2.49 → 2.34 ms and 8.68 → 8.30 ms (~5% on projections) → ~2-4% of cycle | partly (compile-time schedule) | 1 |
| 8 | all-reduce fused into the gemv epilogue: per-tile fp32 push over NVLink, storer warps (Hazy PGL, MPK overlap 1.1x) | exposed tail per call ~3 µs (S1) / ~8 µs (S8) x 128 calls → 0.4 / 1.0 ms → **+1.5-2.5%** over barrier-separated NVLink exchange | yes (1.6) | 1 |
| 9 | L2 policy: evict_first weights, evict_last activations/flags/kv, `prefetch.L2` ahead of the ring (Marlin, upstream) | removes per-operator ramp effects; ~1-3% (upstream's cross-phase prefetch and HyperQwen's "ramp" finding) | yes (device style) | 1 |
| 10 | fence discipline: release/acquire per item, no `fence.sc.sys` (upstream build w, TK2) | upstream measured 28 → 24 µs per call, +0.7%; TK2 lost >10% to loose fences | yes (device style) | 1 |
| 11 | e4m3 → bf16 bit-trick conversion into fragments, plus Marlin-style pre-permuted 16-byte loads | enabling, not a gain: conversion uses <10% of the int pipe at the stream rate (section 0); the pre-permutation is what reaches 75-80% | yes (bake + prims) | 1 |
| 12 | device-resident loop (mapped ring, no per-cycle launch) | upstream host 125 µs + launch ~5-10 µs ≈ 0.13 ms → +0.3-0.6% | yes (1.6) | 1 |
| 13 | split draft vocabulary across cards (upstream) | measured 220 → 132 µs per cycle → +0.4% | kept implicitly (32k vocab) | 1 |
| 14 | bf16x3 split for chunked deltanet state products (llama.cpp); hi/lo P for PV | speed −4% prefill, ~0 decode; required for the contract (flag 12) | no | 1 (correctness) |
| 15 | replicated drafter (Hazy's replicated o-proj trade) | saves ~20-50 µs of exchange latency vs +1.3 ms of bytes → **negative** (Adrien drafter TP1: null) | yes (2.5) | 2; expect a kill |
| 16 | global atomic work queue / scheduler SMs (Hazy throughput, MPK JIT) | Hazy −1.9% at small batch; a scheduler SM costs 1.2% of SMs → negative except for attention tile assignment | no (static schedule, correctly) | do not adopt beyond attention |
| 17 | megakernel prefill (Hazy, Lucebox 3.4x on 0.8B) | 27B prefill is GEMM-bound; kernel boundaries are ~1-2 µs per launch vs ~1 s chunks → ~0% | no | do not adopt |

