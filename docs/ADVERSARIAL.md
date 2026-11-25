# ADVERSARIAL: every hardware and performance claim in PLAN.md, attacked

each claim gets the strongest first-principles argument against it, the numbers, a verdict and the PLAN.md edit it
forces. the evidence base is docs/HARDWARE.md, cited by section: [hw-sm], [hw-dram], [hw-link], [hw-power],
[prior-art], each with its own section number. three more sources:
- [ckpt]: the tensor inventory read from the safetensors headers of ~/models/Qwen3.8-27B-FP8;
- [upstream]: geezuz at `32d1204` (README tables, `qwen38_mk.cu`, `prefill.cuh`);
- [model]: the arithmetic in the last section of this document.

conventions: per card unless stated, gb = 1e9 bytes, "p2" is the 912 gb/s memory clock cuda gets on geforce ampere,
"d" is the draft depth, "R" the rows of one cycle, "S" the live slots. nothing here was measured on the box; every
"unverifiable" verdict names the microbenchmark that settles it (listed again at the end for lane `ubench`).

## decisions for the owner

**resolved by the owner:** 1a, with a correction: the unscaled three-way split is exact only for p ≥ 2^-110 (an
exhaustive check over all 1,065,353,217 fp32 values in [0, 1] found 134,217,728 inexact below that, where the low
part falls under bf16's 2^-133 subnormal step); scaling P by 2^40 first makes every value exact (2^38 already
suffices), and the row sum carries the same scale. PLAN 0.2 records it. 2a: one ptx-jit load path. 3a: `iomem=relaxed`
for the sweep and the soak only.

these need the owner because the fix would change the meaning of the lossless contract or an owner decision. none of
them was applied to PLAN.md. recommendation first.

### 1. attention P before the PV mma

every flash-attention implementation, upstream's included, rounds the softmax weights P to a 16-bit type before the PV
mma. the cpu reference keeps P in fp32, so a 16-bit P is a rounding the contract does not name. prior art proposed a
two-way hi/lo bf16 split; that is **not exact**: hi and lo carry 8 significand bits each, 16 together, against fp32's
24.

- **(recommended) a: exact, three-way split.** P = hi + mid + lo, each bf16 (8 + 8 + 8 bits = every fp32 value, since
  each remainder is exact in fp32 and fits the next bf16). three PV mmas, every product exact, fp32 accumulation. costs:
  - decode: nothing measurable. split-k attention is memory-bound. GQA packs 6 q heads per kv head, so a kv byte
    carries ~6·r flop at r rows per slot (padded to n8). the split doubles that: r = 3 (8 slots) goes from ~24 to
    ~48 flop/byte, r = 5 (1 slot) from ~32 to ~64. the ridge at 71 tflops and 912 gb/s is ~78. 1 slot at 30k context
    loses at most ~0.1 ms per cycle.
  - prefill: attention flops per token are 393k·L for mean context L. PV is half of that, so two more PVs add +1.6%
    at 4k-token prompts and +12% at 32k-token prompts.
- **b: name attention P an activation in 0.2** (bf16 P, ≤ 2⁻⁹ relative error per weight). costs nothing in speed,
  but it changes the contract, and the reference must round P the same way.
- **not an option: the hi/lo two-way split.** it is neither exact nor named.

### 2. one load path (ptx jit) vs ptx plus ptxas-built sass

hw-sm recommends shipping sass next to the ptx, so that drivers older than R610 can still run the engine. the owner's
rule is one load path: embedded ptx, jit-compiled by the driver.

- **(recommended) a: keep one load path.** the minimum driver is R610, the branch that jits ptx 9.3 [hw-sm toolchain].
  a fresh box runs a current driver. the real risk of jit is that a driver update changes results. ptx that pins its
  rounding removes that risk: explicit `.rn` on every fp32 add and mul, no `.approx`, no `.ftz`, fma written out
  (`-ffp-contract=off`). ptxas then cannot contract or approximate. it can reschedule, and rescheduling never changes
  a deterministic algorithm's bits. this pinning was applied to PLAN 1.2 and 1.8 as a static gate.
- **b: ship sass + ptx.** runs on older drivers, and the sass stays fixed across driver updates. costs: a cuda toolkit
  at build time on linux, while the mac build cannot produce the sass, so "the same ptx comes out on every platform"
  becomes two artifacts that can drift. a second load path, too.
- **c: pin a lower ptx isa.** every primitive in the plan exists by ptx 7.8 (`movmatrix`; the rest is ≤ 7.4), so
  `.version 7.8` would jit on drivers back to the cuda 11.8 era. costs: it departs from "the newest isa llvm emits".
  take it only if the box ends up on an old driver.

### 3. seeing the memory junction temperature during the overclock soak

the 110 °C gddr6x junction throttle is the binding limit for a memory overclock: 12 chips sit under the xg7's passive
backplate. nvml does not report it on the 3090. the only reader is the reverse-engineered `gddr6` tool, which needs
`iomem=relaxed` on the kernel command line [hw-dram §5-6, hw-power §8].

- **(recommended) a: enable `iomem=relaxed` only for the sweep and the soak, then remove it.** cost: /dev/mem
  protection is loosened on the host for those runs.
- **b: never.** sweep blind on junction temperature, with a backplate fan and the knee minus three offset steps
  instead of two. cost: part of the ~8% left unclaimed, and a thermal margin nobody sees.
- **c: no memory overclock.** cost: ~6-10% of cycle time (see the overclock entries).

## claims

### 0.2 the lossless contract

```text
claim:      "24.70b params in e4m3 and 3.08b in bf16 (lm_head, embeddings, norms, small linear_attn params, mtp fc)"
attack:     the bf16 count includes the vision tower, which the plan drops.
evidence:   [ckpt]: e4m3 24,699,207,680 params (body 24.33b + mtp 0.37b). text-only bf16 2.62b: lm_head 1.27b,
            embed_tokens 1.27b, in_proj_a/b 0.05b, mtp fc 0.05b, conv and norms. visual.* adds 0.46b, so 3.08b.
verdict:    wrong (2.62b text-only)
change:     "3.08b in bf16" -> "2.62b in bf16 (... mtp fc; 3.08b with the dropped vision tower)"
microbench: none
```

```text
claim:      "every matmul accumulates in fp32" ... "the only allowed difference is floating-point summation order."
attack:     sm_86 tensor cores do not accumulate in ieee fp32. with fp32 output they sum products in blocks of 8 (bf16
            inputs), aligned to the block's largest term, truncated (rz), with a 25-bit significand window and >= 3
            carry bits. that is a rounding behaviour, not a summation order, and every lossless matmul on this card
            goes through it. so the contract's wording describes a machine that does not exist. the cpu reference and
            the rulers must model the mma, or they will count tensor-core truncation as "error".
evidence:   [hw-sm §3]: measured on an rtx 3060 (sm_86) [GM §IV table I], identical to a100 [FASI].
verdict:    wrong wording; the meaning is unchanged (fp32-accumulate tensor-core matmuls were always the intent)
change:     0.2: "every matmul accumulates in fp32 on the tensor cores"; "the only allowed differences are
            floating-point summation order and the tensor core's own fp32 accumulation, which on sm_86 is not ieee:
            it sums products in blocks of 8, aligned to the largest and truncated". 1.8: geezuz-ref carries a
            bit-exact model of the sm_86 mma, used to test primitives exactly and to size the noise floor.
microbench: U6 (mma accumulation model), U7 (e4m3 path through the mma)
```

```text
claim:      (implicit in "only summation order") transcendentals, contraction and denormals match the reference
attack:     upstream computes silu, the deltanet gates and every rms norm with expf / logf / rsqrtf. those lower to
            `ex2.approx` / `rsqrt.approx`-class sequences, or to cuda's own non-correctly-rounded expf, and to ftz
            forms under fast math. the cpu reference uses libm. that is a difference beyond summation order in every
            layer. clang's cuda default `-ffp-contract=fast` also fuses mul+add into fma wherever llvm likes. and a
            driver jit (the one load path) may contract further any fp32 mul/add that lacks `.rn`, changing bits
            across driver updates.
evidence:   [upstream] qwen38_mk.cu lines 357, 642, 1655-1657, 1678, 1742, 1985 (expf, logf, rsqrtf). mufu ex2/rsqrt
            are approximate, 16/clk/SM [hw-sm §2]. ptx `mul`/`add` without a rounding modifier may be fused by the
            code generator (ptx isa, floating-point instructions).
verdict:    incomplete. the fix conforms to the contract as written; it does not reinterpret it.
change:     1.2 device code rules: every fp32 op is ieee rn (add.rn, mul.rn, fma.rn, div.rn, sqrt.rn) or a fixed
            routine built from them; no .approx, no .ftz; fp contraction explicit (-ffp-contract=off, fmas spelled
            out); exp/log/rsqrt and the silu/sigmoid/softplus built on them are fixed fp32 algorithms specified once,
            which geezuz-ref follows, tested over every fp32 input. 1.8 static gate: the ptx has no .approx, no .ftz
            and no fp32 add/mul without .rn.
microbench: U20 (software transcendental throughput vs mufu: expected comparable, ~10-30 fp32 ops vs a 1/8-rate
            mufu op; settles that pinning costs nothing measurable)
```

```text
claim:      "e4m3 to bf16 conversion is exact; the block scale is applied in fp32."
attack:     holds, but only for one specific sequence, and a tempting shortcut breaks it. the shortcut (from vllm's fp8
            marlin) folds the 2^120 exponent offset into the fp32 block scale. that is exact only while every
            tensor-core partial sum stays above 2^-126. a partial below 2^-6 in (activation x raw e4m3) units goes
            subnormal and loses bits. the two nan codes (0x7f, 0xff) silently become ±480 under the bit sequence.
evidence:   [hw-sm §6]: pre(b) = 2^-120·x(b) exactly for all 254 finite codes (simulated). x = fma.rn.bf16x2(pre,
            2^120, -0.0) is exact. sequence B costs 7 instructions per 4 weights on a bake-permuted encoding. folding
            2^120 into the scale is rejected on the measured mma model. checkpoint scales are bf16 [ckpt]: widening to
            fp32 is exact.
verdict:    holds, incomplete (the sequence, the nan rejection and the no-fold rule are not in the plan)
change:     1.2 fp8 cvt: sequence B on a bake-permuted encoding, fma by 2^120, never folded into the scale, tested over
            all 256 codes through the mma. 1.3 bake: rejects 0x7f/0xff with a typed error.
microbench: U7
```

### 0.3 the budget

```text
claim:      "body (fp8, 64 layers, half of each matrix) | 12.19 gb | 13.0 ms | 14.8 ms"
attack:     check that the split, the scales and the small bf16 params add up.
evidence:   [ckpt]: e4m3 body 24.327 gb + scales 3 mb + bf16 in_proj_a/b, conv, norms 52 mb -> 12.191 gb per card.
verdict:    holds
change:     none to the bytes; the rate column changes (see "~88%")
microbench: none
```

```text
claim:      "lm_head (bf16, half of 248,320 vocab) | 1.27 gb" and "main pass | 13.46 gb | 14.4 ms | 16.3 ms"
attack:     check them.
evidence:   [ckpt]: lm_head 2,542,796,800 b / 2 = 1.271 gb. main pass 13.463 gb.
verdict:    holds
change:     none
microbench: none
```

```text
claim:      "mtp head (~5 passes: catch-up + 4 drafts) | ~1.2 gb"
attack:     upstream's catch-up pass produces the logits of draft 1 on its last row, so d drafts take d mtp passes, not
            d + 1. the plan overcounts one pass (~0.24 gb, ~0.3 ms per cycle).
evidence:   [upstream] qwen38_mk.cu run(): the catch-up passes set `want = lg && last_pass`, then
            `for (i = 1; i < ndraft; i++) mtp_pass<1>(..)`. [ckpt]: mtp head 477.2 mb total, 238.6 mb per card per
            pass (fc k-split, attention and mlp halves).
verdict:    wrong (0.24 gb per pass, d passes: 0.95 gb at d = 4, 0.48 gb at d = 2)
change:     0.3 row: "mtp head (d passes for d drafts: catch-up, then d - 1 steps) | 0.24 gb per pass, 0.95 gb at d = 4"
microbench: none
```

```text
claim:      "draft lm_head (32k vocab, 4 steps) | ~0.65 gb"
attack:     bytes are right (32,768 x 5,120 x 2 b / 2 cards = 167.8 mb per step). but the plan does not say where those
            rows live. the mtp head has no lm_head of its own and reads lm_head.weight, so a separate draft copy would
            waste 168 mb of vram per card.
evidence:   [ckpt]: no mtp.*lm_head tensor; [upstream] splits the draft vocabulary across cards.
verdict:    holds, incomplete (layout)
change:     1.3 bake: each card's lm_head rows are permuted so that its half of the draft vocabulary is a contiguous
            prefix, and the draft lm_head is a view of it. 0.3 row: "0.17 gb per step: 0.67 gb at d = 4".
microbench: none
```

```text
claim:      "kv reads (bf16, 2 kv heads per card x 16 layers) | 32 kb x context tokens, per slot | 1.0 ms per 30k"
attack:     right for the trunk, but the mtp layer is a full-attention layer with its own kv over the whole context,
            and every mtp pass reads it: 2 kb x context per slot per pass. at 8 slots, d = 2, 30k context each that
            is 0.98 gb per cycle (~1.2 ms). kv geometry and the pool size also miss it (34 kb per token, not 32).
evidence:   [ckpt]: mtp.layers.0.self_attn.k_proj [1024, 5120] = 4 kv heads x 256. per card 2 heads x 256 x
            2 (k, v) x 2 b = 2,048 b per token. trunk 16 layers x 2,048 = 32,768 b.
verdict:    holds for the trunk; incomplete (mtp kv row missing)
change:     0.3: new row "mtp kv reads | 2 kb x context tokens, per slot and pass"; vram: 34 kb per token.
microbench: none
```

```text
claim:      (missing row) the budget table omits the deltanet state.
attack:     each live slot reads and writes its fp32 state once per cycle: 48 layers x 24 value heads per card x
            128 x 128 x 4 b = 75.5 mb, so 151 mb of traffic. at 8 slots 1.21 gb per card per cycle, ~9% on top of
            the main pass.
evidence:   [ckpt] config: linear_num_value_heads 48, key/value head dim 128, mamba_ssm_dtype float32.
            [prior-art §9 flag 2] the same figure.
verdict:    wrong (row missing; 151 mb per live slot, 1.21 gb at 8)
change:     0.3: new row "deltanet state (fp32, read and written once per cycle) | 151 mb per live slot: 1.21 gb at 8"
microbench: none
```

```text
claim:      "at ~88%" (the stream rate the table plans with: 16.3 ms for the main pass)
attack:     two sides, both partly wrong.
            - hw-dram: a clean read stream reaches ~93-96% of 936 gb/s, so 88% is conservative.
            - prior-art: no decode kernel exceeds ~80% (hazy h100 78%, mpk a100 80%, upstream one-gpu ~71-74%), so 88%
              is a stretch.
            resolution:
            (1) the two measure different things. 93-96% is a pure read kernel, a stream rate. the 78-80% figures are
                end-to-end token latency against the bound, and include dependency bubbles. hazy's own breakdown puts
                42% of its b200 forward in store/flag/load latency chains, because its phases are µs-long (a 1b
                model). ours are >= 17 µs of streaming each (the smallest is in_proj_z, 15.7 mb per card), so a
                1-3 µs handoff fits inside the ring run-ahead and does not idle dram. the plan's formula already keeps
                bubbles out of the rate, in t_serial. a stream rate is the right thing to plan with.
            (2) both sides use the wrong peak. cuda runs geforce ampere in p2, memory at 9501 mhz: 912 gb/s, not 936.
                88% of 936 = 824 gb/s = 90% of 912. so the plan's number survives, for a different reason.
            step 1 (grid barriers) should be expected near upstream's measured ~64% of 936 per cycle. step 2 should
            plan with 90% of p2 for every byte plus an honest t_serial, and target >= 93% of the measured clock's peak
            for the stream itself.
evidence:   [hw-dram §2]: c&c 3090 plateau 890-900 gb/s; a40 (ga102) 97.7%. [hw-power §3]: p2 at 9501 mhz from three
            independent reports, -2.56%. [prior-art §1, §9 flag 1]. upstream q8_0 at 1 slot: 13.7 gb of weights plus
            3 re-reads of the mtp head and 4 draft lm_heads ≈ 14.8 gb in 24.9 ms = 595 gb/s = 64% of 936 [model].
            [ckpt] per-matrix sizes: qkv 26.2, z 15.7, out 15.7, gate/up/down 44.6, q 31.5, o 15.7 mb per card.
verdict:    holds as a number; wrong basis (the reason is p2, not a conservative stream)
change:     0.3 column header "at ~88%" -> "at ~820 gb/s", with one sentence: 90% of the 912 gb/s cuda gets in p2;
            a clean stream reaches 93-96% of peak; dependency stalls are t_serial, not part of the rate.
            1.10 phase b: read-bandwidth gate, >= 93% of the measured memory clock's peak.
microbench: U1 (read stream), U3 (memory clock under cuda)
```

```text
claim:      "t_cycle ≈ max(t_bytes, t_flops) + t_serial"
attack:     right shape, but t_bytes must include every byte a cycle moves: state, both kv rows, d mtp passes. and
            t_flops is a step function of R (see below). with the corrected table a 1-stream cycle (d = 4, 4k context)
            moves 15.4 gb per card, an 8-stream one (d = 2, 4k each) 16.7 gb.
evidence:   [model]
verdict:    holds, incomplete
change:     0.3: the totals sentence under the table
microbench: none
```

```text
claim:      "~71 tflops dense per card, half the fp16-accumulate rate. one verify row costs ~27 gflop per card, ~0.5 ms
            realistically: t_flops(R) ≈ 0.5 ms × R"
attack:     (1) the flop count is 25.6 gflop (24.3 body + 1.27 lm_head), 0.36 ms at peak. "~0.5 ms" is 72% tensor
                efficiency, a fair planning number.
            (2) 71 tflops is at the rated 1695 mhz boost and scales with the sustained clock: 58.6 at 1395, ~79.8 at
                1.9 ghz.
            (3) the formula is linear in R. the hardware sells rows in n8 tiles (mma.m16n8k16: weights are the m16
                a operand, tokens the n8 b operand), so t_flops(R) = 4 ms × ⌈R/8⌉. R = 25 costs what 32 costs.
evidence:   [hw-sm §3]: 512 flop/clk/SM bf16->fp32, measured 505 on ga104 [SUN]. one m16n8k16 holds an SMSP's tensor
            core for 32 clocks. [ckpt] flops.
verdict:    holds (rate), wrong (27 -> 25.6 gflop), incomplete (granularity, clock)
change:     0.3: "~25.6 gflop per card (24.3 body, 1.3 lm_head), ~0.36 ms at peak, ~0.5 ms realistically";
            "t_flops(R) ≈ 4 ms × ⌈R/8⌉"; "71 tflops at the rated 1695 mhz; scales with the sustained clock".
microbench: U4 (mma rate, hfma2 contention), U18 (sustained clock at 400 w)
```

```text
claim:      "compute crosses the 16.3 ms weight stream at about 32 rows ... a row budget of ~24 rows per cycle"
attack:     do the crossover per tensor core, because math and stream share every SM.
            - a 256-byte weight fragment reaches each SMSP every ~152 clocks at 936 gb/s and 1695 mhz, ~173 at
              820 gb/s, ~194 at 820 gb/s and 1.9 ghz, ~143 at 820 gb/s and base 1395.
            - each n8 tile costs 32 clocks: 24 rows 96, 32 rows 128 (+8 if hfma2 shares the tensor pipe), 40 rows 160.
            so the crossover sits between 32 and 40 rows. 24 rows (63% busy at 936) is safe. 32 rows (74-79% at 820
            gb/s) probably fits at any sustained clock >= ~1.6 ghz. nothing between them can be bought.
            32 rows at 8 slots means 3 drafts per slot instead of 2 (code τ 2.66 -> 3.34 at upstream's 8-slot
            acceptance) for one more mtp pass (~0.5 ms): ~+20% aggregate at 8 streams.
            step 1 runs near upstream's ~64% stream efficiency (fragments every ~237 clocks), so 32 rows is certainly
            under the stream there. upstream itself ran 32 rows from 5 slots. at 24 rows, step 1's 8-stream number
            would fall below upstream's measured 610.
evidence:   [hw-sm §6 table and implication 1]; [upstream] README "parallel requests" (32 rows from 5 slots, 610 tok/s
            at 8); [model] τ from upstream's 8-slot run, chain acceptance p ≈ 0.88.
verdict:    holds (24 as the step-2 default); incomplete (n8 granularity; 32 is a real candidate worth ~+20% at 8
            streams; step 1 should run 32)
change:     0.3 compute-ceiling paragraph rewritten with the per-tensor-core arithmetic; "row budget of 24 rows per
            cycle (3 tiles) ... 32 (4 tiles) is the first thing measured". 1.4: "24 rows in step 2 (32 measured
            first), 32 in step 1, as upstream". 2.4 rewritten around {24, 32}. 2.1, 2.3 and risks: "~24" -> "24-32".
microbench: U5 (streaming gemv at R = 8..40)
```

```text
claim:      "~13.5 gb weights plus ~1 gb workspace leaves ~9 gb: about 280k tokens of bf16 kv"
attack:     - resident weights are 13.70 gb (main pass 13.46 + mtp head 0.24), not 13.5. the draft lm_head is free
              only if it is a view (bake permutation).
            - deltanet state at 8 slots (0.6 gb) is not in the subtraction.
            - kv per token is 34 kb (trunk 32 + mtp 2), not 32.
            - a 24 gib card has 25.77 gb, ~25.3 after the cuda context.
            25.3 - 13.7 - 1.0 - 0.6 ≈ 10 gb; minus snapshots (75.5 mb each, at most a quarter of the pool in
            upstream) that is ~9.5 gb of kv at 34 kb: ~275k tokens. the headline survives by cancellation.
evidence:   [ckpt]; [upstream] README (snapshots capped at a quarter of the pool).
verdict:    holds (~275k), incomplete (state, mtp kv, the view)
change:     0.3 vram bullets rewritten with those terms
microbench: none
```

```text
claim:      "deltanet state is ~75 mb per slot (fp32), ~600 mb at 8 slots, plus snapshots"
attack:     check it.
evidence:   [ckpt]: 75.5 mb per slot per card, plus ~3 mb of fp32 conv state (48 x 5,120 x 3 x 4 b).
verdict:    holds
change:     adds the conv state and the snapshot size (75.5 mb each)
microbench: none
```

```text
claim:      "the embedding table (2.54 gb bf16) lives in pinned host memory" (and hw-link's proposal to add the
            32k draft-vocab rows to vram, 335.5 mb per card, so that serial draft steps avoid pcie)
attack:     two adjudications.
            - the plan: right. the table is 2.54 gb [ckpt].
            - hw-link's proposal, on its own numbers: one host fetch per serial draft step costs ~1.5 µs of latency plus
              S x 10 kb at ~24 gb/s, ~5 µs at 8 slots, ~2 µs at 1. there are d - 1 such steps per cycle, so ≤ ~15 µs
              (<0.1% of a ~20 ms cycle). against that, 335.5 mb is ~10k tokens of kv pool (~3.6%). the trade does not
              favour vram unless the measured fetch is several times worse.
            what is missing instead: each row should cross pcie once. the embedding the mtp step fetches for a draft
            token is the same row the next verify pass needs, so stage it in vram and reuse it. the main pass then
            fetches only the rows no draft step touched. also: with upstream's k-split eh_proj only card 0 reads
            embedding columns at all.
evidence:   [hw-link §5] (latency ~1-2 µs, 24 gb/s effective, both slots on cpu0);
            [upstream] mtp_pass (eh_proj k-split).
verdict:    plan holds; the hw-link vram proposal is not supported by the numbers (kept as a measured experiment)
change:     0.3 vram: host-resident on numa node 0, each row fetched once into a vram staging buffer by the step
            that first needs it. new 2.15: draft-vocab embedding rows in vram, killed if a draft step's measured
            fetch is under ~10 µs.
microbench: U14 (mapped host fetch latency and bandwidth, node 0 vs node 1)
```

### 1.2 device style (each idiom against the sm_86 ptx feature list)

```text
claim:      "warp specialization: producer warps own the weight stream, consumer warps own the math"
attack:     right, but sm_86 has no setmaxnreg: a producer warp holds as many registers as a consumer, so every
            extra producer warp costs a consumer's worth of registers. one cp.async.cg warp instruction moves 512 b,
            and an SM needs one every ~76 clocks: one producer warp has the issue rate. what is unknown is how many
            requests one warp can keep in flight. little's law needs ~11 kb per SM at a 1 µs loaded latency.
evidence:   [hw-sm implication 6]; [hw-dram §3] (in-flight table; one 3090 SM sustains 47.6 gb/s, ~12.9 kb in flight).
verdict:    holds, incomplete (producer count is a register cost; settle it by measurement)
change:     1.2 warp specialization: "no setmaxnreg on sm_86 ... as few producer warps as keep ~11 kb per SM in flight"
microbench: U1 (producer warps 1/2/4/8 x in-flight depth)
```

```text
claim:      "async copy: cp.async.cg 16-byte transfers, completion signalled with cp.async.mbarrier.arrive.noinc"
attack:     right, and it is the cheapest "copy landed" signal on sm_86: the copy's own completion arrives, and the
            producer never waits. missing from the plan:
            - the .L2::256B prefetch hint, to match the 256-byte dram chunk;
            - that producers issue whole contiguous 16 kb tiles, for dram row locality.
evidence:   [hw-sm §5, §7]; [hw-dram §1] (a 256 b chunk maps to one bank and row; a strided 128 b fragment layout costs
            activates, trc ≈ 45 ns).
verdict:    holds, incomplete
change:     1.2 async copy: adds .L2::256B and whole 16 kb tiles. 1.3 bake: tile-contiguous runs.
microbench: U2 (contiguous vs strided tiles)
```

```text
claim:      "mbarrier ... wait is a test_wait.parity spin with nanosleep backoff ... named barriers only for fixed
            sub-groups"
attack:     correct for sm_86 (no try_wait, no expect_tx; arrivals counted, not bytes). one alternative the plan does
            not weigh. ring slots of a whole 16 kb tile in a 64-80 kb ring give 4-5 slots, so the slot-empty signal
            can be a hardware named-barrier pair (16 ids per cta allow ~7 full/empty pairs). that may be cheaper than
            a shared-memory mbarrier spin. per-operation costs on sm_86 are unpublished (hazy measured ~60 ns per async
            barrier on h100).
evidence:   [hw-sm §7, implication 6]; [prior-art §1].
verdict:    holds; unverifiable which is cheaper
change:     1.2 mbarrier bullet: "with a ring of <= 7 slots, named-barrier full/empty pairs are the alternative; a
            microbenchmark picks one"
microbench: U8 (mbarrier arrive/test_wait vs bar.arrive/bar.sync; cp.async completion latency; nanosleep grain)
```

```text
claim:      "l2 control: createpolicy.fractional with evict_first on the weight stream, evict_last on activations, kv
            tiles and flags"
attack:     kv is a single-use stream within a cycle. GQA packing means each slot's kv is read exactly once per cycle,
            by one cta per split-k chunk, and the next read is a cycle (~20 ms) away. evict_last on gigabytes of kv
            would push activations, partials and flags out of a 6 mb l2. the same holds for deltanet state reads.
            evict_last belongs only on what is reused within microseconds.
evidence:   [hw-sm §4, implication 7]; [hw-dram implication 2] ("l2 is not a safe ring"); kv per cycle at 8 slots and
            4k context 1.07 gb, at 30k 7.9 gb [model].
verdict:    wrong (kv evict_first)
change:     1.2 l2 control: "evict_first on every single-use stream (weights, kv, deltanet state); evict_last only on
            activations, partials and flags; prefetch.global.L2::evict_last ahead of the ring, applypriority
            evict_normal after use; discard.global.L2 on dead scratch"
microbench: U15 (persisting-l2 attributes; survival of evict_last lines under an evict_first stream)
```

```text
claim:      "fragments: ldmatrix (and .trans) from xor-swizzled shared memory with zero bank conflicts" (for every
            operand)
attack:     weights need no ldmatrix and no swizzle. the bake can write each lane's a-fragment bytes for two k16 steps
            as one contiguous 16 b run, bit-permuted for the decode. a warp's 512 bytes are then a single
            conflict-free ld.shared.v4 (4 whole wavefronts), straight into the decode and the mma. ldmatrix stays for
            the bf16 activation b fragments.
evidence:   [hw-sm §6 "fragment layout", implication 3]; marlin's pre-shuffled 16-byte loads [prior-art §6].
verdict:    incomplete (the weight path is simpler than the plan says)
change:     1.2 fragments bullet rewritten; 1.3 bake: per-lane fragment order and bit permutation
microbench: U5
```

```text
claim:      "cross-card: ... st.release.sys / ld.acquire.sys flags with nanosleep backoff, or red.release.sys counters"
attack:     red.release.sys on peer memory is a peer atomic. it is atomic at system scope only if
            cudaDevP2PAttrNativeAtomicSupported is 1, which is unestablished on a 3090 pair. and the protocol needs
            no atomics: single-writer flags with monotonic epochs, each card writing only its own receive slots in the
            peer's memory. also missing: sm_86 has one memory-sync domain, so a .sys release waits on unrelated
            in-flight writes, including pcie writes to the host ring (fence interference).
evidence:   [hw-link §4] (scopes, push-not-pull, fence interference, atomics attribute; upstream measured
            fence.sc.sys + flag 28 µs -> st.release.sys 24 µs per call).
verdict:    wrong (red.release.sys counters), incomplete (fence interference)
change:     1.2 cross-card bullet: push only, single-writer epoch flags polled locally, one release per tile, no peer
            atomics, host-ring writes once per cycle away from release points
microbench: U10, U12, U13
```

```text
claim:      "reductions: redux.sync for integers, butterfly shfl.sync for floats in a fixed order"
attack:     correct: float redux.sync is sm_100a. one addition. widen bf16 to fp32 with prmt or shl 16 (integer pipe,
            64/clk/SM), never cvt.f32.bf16 (conversion pipe, 16/clk/SM).
evidence:   [hw-sm §2, ptx list].
verdict:    holds
change:     none (the widening note goes in the fp8 cvt bullet as part of the decode)
microbench: none
```

```text
claim:      "tma: producers issue the baked image's precomputed 16-byte transaction stream ... a pointer bump plus
            cp.async"
attack:     right, and the elegant core of the design. what it leaves unsaid: the stream per cycle depends on the draft
            depth, since the mtp head and the draft lm_head are re-read d times. with the fixed allocation, d is a
            function of the live-slot count. the cycle's shape is therefore an enum case (d = 1..4), each a
            compile-time stream program, not a runtime decision.
evidence:   [upstream] mtp pass structure; [model].
verdict:    holds, incomplete
change:     1.6 step 2 compile-time schedule: "one stream program per draft depth"
microbench: none
```

```text
claim:      "fp8 cvt: an exact e4m3-to-bf16 bit sequence (prmt / lop3 / shifts) converting straight into mma
            a-fragments"
attack:     not concrete enough to be checked. there is no exact integer-only sequence: zero and the e4m3 subnormals
            defeat any additive exponent fix. so the sequence needs one fma.rn.bf16x2 per pair. the cheapest found is
            5 integer ops plus 2 fma per 4 codes, on a bake-permuted encoding, ~13% of issue slots at R = 24. the
            plan's list names prmt, which the best sequence does not need.
evidence:   [hw-sm §6] sequences A and B, the cost table; 0x7f/0xff map to ±480.
verdict:    incomplete
change:     1.2 fp8 cvt bullet rewritten (sequence, fma by 2^120 with a -0 addend, no folding, test through the mma)
microbench: U4, U7
```

```text
claim:      "clusters / distributed shared memory: none. cross-SM handoff goes through l2-resident flags and buffers"
            and "banned: ... cooperative-groups grid.sync ... per-phase kernel launches ... texture paths"
attack:     correct on sm_86 (clusters, dsmem, griddepcontrol and elect.sync are all sm_90). each cross-SM hop costs
            several hundred ns (l2 ~138-144 ns unloaded): store, flag, observe and load make ~1-2 µs. hazy measured
            exactly this chain as 42% of a small model's forward. here it hides under the ring, because our phases
            are >= 17 µs of streaming.
evidence:   [hw-sm §7]; [hw-dram §3]; [prior-art §1].
verdict:    holds
change:     none
microbench: U9 (cross-SM counter hop and the full dependency chain)
```

### 1.2 toolchain

```text
claim:      "one load path, no cudart. the ptx ... loaded with cuModuleLoadData, which jit-compiles it; the driver's
            compute cache keeps start-up cheap" vs hw-sm's "ship sass built by ptxas alongside the ptx"
attack:     the case for sass is older drivers and fixed codegen. the box will run a current driver (>= R610 for ptx
            9.3). fixed codegen matters for numerics only if the ptx leaves the jit room to change rounding, and that
            is closed by the rounding pins above. so sass buys little and costs the single artifact.
evidence:   [hw-sm toolchain]: llvm 23.1.5 emits ptx 9.3; minimum driver R610; minor-version compatibility does not
            cover ptx.
verdict:    holds (the owner's rule); the competing proposal is an owner decision (decision 2)
change:     none beyond the rounding pins (1.2, 1.8)
microbench: none
```

### 1.3 the baked image

```text
claim:      "lays out e4m3 tiles and scales in exact stream order, 16-byte aligned"
attack:     16-byte alignment is the cp.async minimum, not the right unit. the unit is the 256-byte dram chunk, one bank
            and row. the layout should be:
            - each 128x128 tile as one contiguous 16 kb run, 64 whole chunks;
            - inside it, each lane's 16 bytes in a-fragment order, bit-permuted for the decode;
            - scales in a separate small array (inline 2-byte scales would break chunk alignment);
            - the nan codes rejected;
            - the lm_head rows permuted so the draft vocabulary is a contiguous view.
evidence:   [hw-dram §1]; [hw-sm §6, implication 3].
verdict:    incomplete
change:     1.3 bake bullets rewritten
microbench: U2
```

### 1.4 server

```text
claim:      "row budget: ~24 rows per cycle ... the rest split evenly as draft depth, capped at 4 per slot"
attack:     see the crossover entry. also: a slot's verify rows must compute exactly what plain decoding would, or
            "greedy is exact" holds only in distribution. upstream concedes it does not: "greedy output after a pause or
            in a different batch is a valid continuation but not always token-identical". on this design batch
            invariance is free:
            - mma columns are independent;
            - the k-split is static and independent of R;
            - attention chunks can sit at fixed absolute positions, merged in index order, with balancing done only
              by assigning chunks to ctas;
            - the deltanet recurrence is per token anyway.
evidence:   [upstream] README "shared kv pool and prefix cache", last bullet.
verdict:    holds, incomplete
change:     1.4 row budget wording; 1.8 determinism harness: batch invariance
microbench: none (the golden hashes check it)
```

### 1.6 device: decode and prefill

```text
claim:      step 1 "straight port, the correctness baseline ... grid barriers stay"
attack:     a straight port carries upstream's numerics outside the contract unless the plan names them:
            - chunked-prefill deltanet: the fp32 state becomes an fp16 mma b operand every chunk;
            - attention P rounded to fp16 before PV, in decode and prefill;
            - fp16 activations and fp16-accumulate prefill gemms (already culled);
            - fast-math transcendentals.
            also, its row count: at 24 rows its 8-stream number falls below upstream's own 610 (see the crossover
            entry).
evidence:   [upstream] prefill.cuh line 189 ("becomes an fp16 b operand"); [prior-art §5, §9 flag 12]; transcendentals
            entry above.
verdict:    incomplete
change:     1.6 step 1: a paragraph naming the four gaps and their fixes. 1.4: step 1 runs 32 rows
microbench: none
```

```text
claim:      step 2 "~8 mb of shared memory plus 6 mb of l2 gives ~15 µs of run-ahead"
attack:     - 8 mb is all of shared memory (82 x 100 kb). a ring gets 64-80 kb per SM, since activation staging
              (k128 x R bf16, double-buffered), partial-sum buffers and barriers need the rest: 5.4-6.7 mb per card,
              ~6-8 µs at 820 gb/s.
            - l2 is not a safe ring: 6 mb shared with kv, activations and nvlink partials; evict_first weights do not
              stay; persisting set-aside ≤ ~4.5 mb (inferred). evict_last prefetch might add 2-4 µs, so 8-11 µs is the
              ceiling, ~6-8 µs the reliable part.
            - run-ahead is not a bandwidth need. little's law needs only ~0.9 mb in flight card-wide (~11 kb per SM at
              1 µs). the ring beyond that is a jitter buffer: how long consumers may stall before dram idles.
evidence:   [hw-sm implication 5]; [hw-dram implication 2]; [prior-art §9 flag 3].
verdict:    wrong (~6-8 µs reliable, ≤ ~11 µs with l2 prefetch; a jitter buffer)
change:     1.6 step 2 first bullet rewritten
microbench: U16 (ring jitter buffer), U15
```

```text
claim:      "weights have no data dependency, so dram never has to wait"
attack:     producers never wait on data, but they wait on ring space. when a consumer phase stalls longer than the
            run-ahead (the deltanet core at 8 slots, see below), dram idles.
evidence:   as above.
verdict:    wrong as stated (true only while every stall < ~6-8 µs)
change:     folded into the rewritten step-2 bullets
microbench: U16
```

```text
claim:      "serial phases under the weight stream: deltanet core, attention, lm_head merge, sampling and the mtp steps
            run while producers prefetch"
attack:     at 8 slots the deltanet core moves 8 x 24 heads x 64 kb = 12.6 mb of state in and 12.6 mb out per layer,
            ~27 µs at 936 gb/s. upstream's core took ~80 µs per layer: 192 items on 82 blocks, 3 waves. both are far
            past a 6-8 µs run-ahead, and the state traffic is itself dram time. so "hide under the stream" is
            impossible at 8 slots. what works is to make the serial phases bandwidth-bound:
            - the core as one wave over all SMs at state bandwidth (value columns are independent chains; a head's
              128 columns stay in one cta for the output norm);
            - state reads as a dependency-free producer stream;
            - split-k attention from a balanced per-cycle plan (fixed chunks, assignment balanced);
            - dependency-free weights streamed during the core and attention: in_proj_z (15.7 mb, ~17 µs) is consumed
              only after the core, and the gate half of q_proj (15.7 mb) only after attention.
            the mtp steps are not latency-bound phases: each streams 0.41 gb (head + draft lm_head), ~0.5 ms, with a
            few µs of exchange and argmax latency. at 1 slot (core ~10 µs) the plan's claim holds.
evidence:   [prior-art §7] (core 3.83 ms / 48 layers at 8 slots; floor 27 µs), [prior-art §9 flag 3, §10];
            [ckpt] matrix sizes.
verdict:    wrong at 8 slots (made fast, not hidden); holds at 1 slot
change:     1.6 step 2 "serial phases" bullet rewritten
microbench: U17 (deltanet core at 8 slots: state bandwidth and the per-token chain latency)
```

```text
claim:      "the mtp steps overlap the next cycle's layer-0 prefetch"
attack:     that buys at most one run-ahead (~6-11 µs) per cycle, since the next cycle's compute needs the accepted
            tokens. and with a continuous producer stream it happens by itself: the stream wraps from the last draft
            lm_head to layer 0. it is not a lever.
evidence:   [prior-art §9 flag 11].
verdict:    holds trivially; not worth a bullet
change:     removed from the step-2 bullet
microbench: none
```

```text
claim:      "all-reduce fused into the gemv epilogue: fp32 partials go over nvlink per tile as they finish"
attack:     size check:
            - 128 row-parallel exchanges per main pass (48 out_proj + 16 o_proj + 64 down);
            - at 24 rows each is 24 x 5,120 x 4 b = 492 kb per direction, ~9.3 µs of link time at the measured
              52.7 gb/s, under a layer's ~200 µs of streaming;
            - exposed per exchange: the last tile's partial (~12 kb, ~0.25 µs) plus one handoff (~1-2 µs), i.e.
              ~0.15-0.48 ms per cycle;
            - unfused it would cost ~1.4-2.1 ms. fp32 is forced by the contract anyway.
            the handoff latency is unmeasured on a 3090 pair, and that is the uncertainty.
evidence:   [hw-link §1, §3, §7]; [prior-art §9 flag 9].
verdict:    holds; unverifiable (handoff latency)
change:     the step-2 bullet adds the exposed cost
microbench: U10 (flag ping-pong), U11 (sm-store bandwidth vs size), U12 (fence interference)
```

```text
claim:      "device-resident loop: ... exchanges cycle descriptors and results through a mapped, pinned ring"
attack:     right (upstream paid ~125 µs of host time per cycle). but a gpu poll of host memory is a pcie round trip
            (~1-2 µs), so it must never sit on the critical path. the device can build each next cycle itself from the
            accepted tokens. the host's mailbox (admissions, cancellations, a yield for a prefill launch) is read with
            a load issued at cycle start and consumed at cycle end. also unsaid: the resident kernel owns every SM,
            so a prefill chunk needs the loop to yield at a cycle boundary. that costs one relaunch (~10 µs) against
            ~1 s chunks.
evidence:   [hw-link §5] (polling ~1-2 µs; posted writes count against the next .sys release).
verdict:    holds, incomplete
change:     1.6 step 2 resident-loop bullet
microbench: U14
```

```text
claim:      "why it's certain: upstream's measured cycle (~20.9 ms on q4_k_m) is about twice its weight-read floor
            (~9.3 ms)"
attack:     check it. 20.85 ms is upstream's python-function run: 213.9 tok/s at τ 4.46. 8.7 gb per card / 936 gb/s
            = 9.3 ms. counting the 3 mtp re-reads and the draft lm_heads, the floor is ~10 ms, still ~2x.
evidence:   [upstream] README "one request" and "two gpus".
verdict:    holds
change:     none
microbench: none
```

```text
claim:      prefill "~71 tflops per card, so ~2,000 tok/s total is the target"
attack:     each card does 24.3 gflop per token (its half of every matrix, for every token), so 71 tflops is a
            ~2,900 tok/s ceiling. 2,000 needs ~50 tflops per card sustained in situ, plus attention, deltanet and
            exchange hidden. nothing on this card shows fp32-accumulate gemms above ~54-63 tflops even in isolation:
            marlin ~63 at 250 w, fa2 54-57. upstream reached only ~53 tflops in situ on an fp16-accumulate path with a
            142 peak, ~half its microbenchmark. hyperqwen's 1,437 tok/s on one card implies >= 70 tflops and cannot
            be fp32-accumulate. the realistic range is ~1,300-2,000.
evidence:   [prior-art §0, §9 flag 6]; [ckpt] 48.6 gflop per token for the body.
verdict:    optimistic (~1,600 is the planning number; 2,000 is the stretch)
change:     1.6 prefill bullet
microbench: U22 (in-situ prefill gemm rate)
```

```text
claim:      prefill "chunked (wy) deltanet on tensor cores with fp32 state"
attack:     this is outside the contract twice.
            (1) the chunked form is not a summation reorder of the recurrence. it re-associates products of gates
                (exp of cumulative-sum differences instead of step-by-step multiplication) and adds a triangular solve.
            (2) its tensor-core operands round the state: upstream to fp16, llama.cpp's "bf16x3" (hi·hi + hi·lo +
                lo·hi) to ~16 bits. hw/prior-art called bf16x3 contract-safe; it is not.
            the alternative is cheap. value columns are independent chains (3,072 per layer per card). run the decode
            recurrence itself in fp32 on the cuda cores: ~4 fp32 ops per state element per token, ~0.2-0.4 ms per layer
            per 2,048-token chunk (throughput- and chain-latency-bound estimates), ~10-19 ms per chunk over 48 layers,
            ~1-2% of prefill. that is about what upstream's chunked kernel cost (~2%). it leaves bit for bit the state
            decode would have, so chunk boundaries and prefix snapshots cannot change results, and it deletes a kernel.
evidence:   [upstream] prefill.cuh lines 185-192; [prior-art §4, §5]; arithmetic in "what the plan misses" item 1.
verdict:    wrong (contract) — the fix conforms to the contract, so it is applied
change:     1.6 prefill bullet: the recurrence in fp32 on the cuda cores; step-1 gap list names the fp16 state operand
microbench: U17
```

### 1.7 platform

```text
claim:      "nvlink: nvidia-smi topo -m shows NV4 ... check that the 2-slot bridge spacing matches PCIE_6 / PCIE_4"
attack:     the owner has confirmed the fit. the bring-up gotchas: links stay inactive until persistence mode is on;
            seating. iommu=pt is fine for nvlink, iommu=off is the fallback.
evidence:   [HARDWARE facts reported by the owner]; [hw-link §2].
verdict:    holds; incomplete
change:     1.7 nvlink bullet
microbench: U13
```

```text
claim:      "power: limit at or above stock 350 w, persistence mode, locked clocks so boost doesn't oscillate"
attack:     take 400 w, the fe vbios maximum. memory power is roughly fixed by bandwidth (~55-60 w for the dram
            devices), so every watt a cap removes comes out of the core clock, and t_flops ∝ 1/f_core sets how many
            rows fit. at 1395 mhz, 24 rows of compute is 14.6 ms against a 16.3 ms stream. a lock is real only below
            what 400 w sustains under the heaviest phase mix; above it, throttling turns it back into a floating
            clock. accept it only with the sw_power_cap counter flat. nothing persists across reboot or driver
            reload.
evidence:   [hw-power §1, §2, §4, §5, implications].
verdict:    incomplete (400 w; lock acceptance rule)
change:     1.7 power bullet
microbench: U18
```

```text
claim:      (missing) the memory clock cuda actually runs at
attack:     cuda contexts put geforce ampere in p2 with memory at 9501 mhz (19.0 gb/s): -2.56%, 912 gb/s. linux has no
            documented switch. -lmc 9751 is documented to apply under cuda, but the one 3090-like report settled at
            9501. a memory offset raises the p2 clock, so an overclock is counted from 9501.
evidence:   [hw-power §3] (three independent reports, linux and windows).
verdict:    wrong (missing; the budget's 936 is not what cuda gets)
change:     1.7 new memory-clock bullet; 0.3 rate column on 912
microbench: U3
```

```text
claim:      "memory overclock: ... adrien's 5090s got +15% raw bandwidth but only +4% at tp2 ... our fused nvlink
            exchange should let more of it through" and 1.9 "+ memory overclock (~8%)"
attack:     the cycle gain is the stream-bound share of the cycle times the clock gain. in step 2 that share is ~97%.
            the in-rating ceiling is ~21 gb/s (micron's -21 bin), +10.5% over p2's 19.0 (+7.7% over p0), so ~+8-10%
            tok/s if the knee and the power budget allow. the binding limit is back-side junction temperature, not
            edr. the soak cannot rely on golden hashes alone: edr's crc covers link data only (not the array, not the
            command/address bus, not the gpu side), and some multi-bit link errors escape it.
evidence:   [hw-dram §4, §5, implications 3-4]; [hw-power implications]; [prior-art §8, §9 flag 8].
verdict:    holds (~8%), incomplete (base is p2; soak needs an image hash and a pattern test)
change:     1.7 memory overclock bullet; 1.9 row
microbench: U19 (offset sweep)
```

```text
claim:      "memory thermals: ... watch the memory junction temperature during the soak; add a backplate fan if it
            climbs"
attack:     it cannot be watched through nvml on a 3090. under water, igor measured +44 k over water at +1000 mem in a
            gaming load, and a continuous stream is harder on memory. make the fan the default, not the fallback.
evidence:   [hw-dram §5]; [hw-power §7, §8].
verdict:    incomplete
change:     1.7 thermals bullet; decision 3
microbench: U19
```

```text
claim:      "host: iommu passthrough, hugepage-backed pinned memory ..., host thread pinned to one core"
attack:     both slots hang off cpu0 (datasheet and block diagram). pinned buffers and the loop thread belong on numa
            node 0.
evidence:   [hw-link §5].
verdict:    incomplete
change:     1.7 host bullet
microbench: U14
```

### 1.9 must-do outcome

```text
claim:      upstream row: "~25 ms | ~143 | ~85 (est) | 610"
attack:     the code column mixes benchmarks. τ ≈ 4.46 is the python-function prompt. 143 comes from the --slots run
            (variants of one coding prompt), whose τ is 143 x 0.0249 ≈ 3.6. at τ 4.46 and 24.9 ms the q8_0 cycle gives
            ~179. so the table overstates step 1's gain over upstream.
evidence:   [upstream] README: "Python function ... 4.46"; parallel-requests table (1 request 144 tok/s at 24.9 ms).
verdict:    wrong (~179 at τ 4.46; 143 measured at τ ≈ 3.6)
change:     1.9 upstream row
microbench: none
```

```text
claim:      step 1: "~23 ms | ~190 | ~95 | ~700"
attack:     step 1 keeps upstream's algorithm and barriers, so expect upstream's efficiency (~595 gb/s for every byte).
            its image moves 15.2 gb per 1-stream cycle (main 13.46, 4 mtp passes 0.95, 4 draft lm_heads 0.67, state
            0.15): ~25.6 ms, minus ~0.5-1 ms of nvlink instead of pcie exchange, so ~25 ms. at 8 streams: upstream's
            1.8 ms per added slot, 32 rows: ~42 ms; 8 x τ(3) ≈ 3.34 / 42 ms ≈ 640. at 24 rows it would be ~550.
evidence:   [prior-art §9 flag 5] (~25-27 ms); [model].
verdict:    optimistic (~25 ms, ~180, ~87, ~640 at 32 rows)
change:     1.9 step-1 row
microbench: none (phase b measures upstream's breakdown)
```

```text
claim:      step 2: "~18 ms | ~250 | ~120 | ~1,000"
attack:     1 stream: 15.4 gb at 820 gb/s is 18.8 ms, plus 0.3-0.6 ms exposed (sampling and merge, a few µs per draft
            step, cycle start, exchange tails): ~19.1-19.4 ms, ~230 code, ~112 prose. 18 ms needs p0 clocks and a 93%
            stream. prior art's ~21-22 ms assumes 78-80% cycle efficiency, which comes from µs-scale phases (see
            "~88%").
            8 streams: the plan's ~1,000 at an ~18 ms cycle implies τ ≈ 2.25 per slot. the cycle is longer, but τ is
            higher:
            - 16.7 gb (4k context each, d = 2) at 820 gb/s = 20.3 ms, plus 0.5-3.3 ms exposed: 21-24 ms;
            - τ(2) ≈ 2.66 at upstream's 8-slot acceptance, so ~900-1,000.
            prior-art's 650-750 assumed 0.6-0.9 ms of per-slot cost on a 21.5 ms base. it is right at long contexts:
            at 30k per slot, kv adds ~8 ms and the aggregate falls to ~700.
            what decides the 8-stream number is how much serial time per slot step 2 leaves exposed (upstream ~1.8 ms;
            the range assumes ≤ 0.35 ms). and 32 rows, if they fit, give ~1,100-1,250.
evidence:   [model]; [prior-art §9 flags 1, 4]; [upstream] slot sweep.
verdict:    1 stream optimistic (~19 ms, ~230, ~112); 8 streams holds at 4k context (~900-1,000), falls to ~700 at 30k
change:     1.9 table restructured (1-stream and 8-stream cycles separate) with the assumptions stated
microbench: U5, U16, U17 (and phase b's MK_PROFILE breakdown of upstream)
```

```text
claim:      "+ memory overclock (~8%) | ~16.8 ms | ~265 | ~128 | ~1,080"
attack:     at 21 gb/s and 90%: 1 stream ~17.3-17.6 ms, ~255, ~123; 8 streams ~19-22 ms, ~980-1,100.
evidence:   [model]; the overclock entry above.
verdict:    holds roughly (re-based on the corrected step 2)
change:     1.9 row
microbench: U19
```

```text
claim:      "S = 8" (locked) against hyperqwen's ~1,035 tok/s at 64 concurrent on one 3090
attack:     hyperqwen's 64-way rests on int4 weights, int8 activations and fp16 deltanet state, all excluded. its own
            data supports our regime: "speculation wins below ~8 concurrent", 40-64 rows are tensor-bound even with fp16
            math, and fp32 state capped it at 37 residents. on our numbers, at 32 rows:
            - S = 16 (1 draft each) gives ~1,230 aggregate against ~1,210 for S = 8 (3 drafts each), flat;
            - per-stream speed halves (77 vs 151 tok/s);
            - state vram doubles (+0.6 gb);
            - the ~275k-token pool caps S x context: 16 slots get ~17k each.
            nothing favours S ≫ 8.
evidence:   [prior-art §8, §9 flag 7]; [model].
verdict:    holds
change:     2.4 notes the S arithmetic
microbench: U5 (the row sweep is what S tuning needs)
```

### part 2 and risks

```text
claim:      2.5 replicated drafter: "+0.48 gb vram each ... ~+1.3 ms per cycle to save the exchange latency of ~10
            serial steps"
attack:     it adds the other card's mtp half to every pass (+0.24 gb x d per card per cycle, ~1.2 ms at d = 4) to save
            ~2-3 exchanges per pass at ~1-2 µs each over nvlink (~20-50 µs per cycle). break-even needs ~100 µs per
            exchange, implausible over nvlink. adrien measured drafter tp1 as null; hazy's replicated o-proj cost
            ~15% capacity.
evidence:   [hw-link §7]; [prior-art §8, §9 flag 10].
verdict:    killed by arithmetic
change:     2.5 replaced by a short "killed" entry with the arithmetic (numbering kept)
microbench: U10 settles it in an hour if anyone doubts it
```

```text
claim:      2.6 "drafter bytes (~1.85 gb per card per cycle as shipped)"
attack:     d passes, not d + 1: 4 x (0.24 + 0.17) = 1.63 gb at d = 4, 0.81 gb at d = 2.
evidence:   mtp entry above.
verdict:    wrong (1.63 / 0.81 gb)
change:     2.6 numbers
microbench: none
```

```text
claim:      2.1 / 2.3 / risks: "the bf16 compute ceiling caps rows at ~24 per cycle"
attack:     24 is a safe default, not the ceiling: rows come in n8 tiles and 32 probably fits (see the crossover entry).
            at 1 stream, 5 of 32 rows are used, so tree drafts there compete with nothing but their mtp passes and
            per-branch state.
evidence:   crossover entry.
verdict:    incomplete
change:     "24-32" wording in 2.1, 2.3, risks
microbench: U5
```

```text
claim:      2.9 "the ~280k-token pool"
attack:     ~275k at 34 kb per token.
evidence:   vram entry.
verdict:    holds roughly
change:     "~275k"
microbench: none
```

```text
claim:      2.10 "saves up to ~0.6 gb per card per cycle"
attack:     lm_head is 1.27 gb per card per cycle. a low-precision shadow plus recomputed survivors can save about
            half, if the bounds prune well.
evidence:   [ckpt].
verdict:    holds
change:     none
microbench: none
```

```text
claim:      risks: "the bf16 compute ceiling: the lossless contract caps verify width at ~24 rows per cycle"
attack:     as above. the larger risk the list misses is the 8-stream number itself: it rests on step 2 leaving
            ≤ ~0.35 ms of serial time exposed per slot (upstream ~1.8 ms).
evidence:   1.9 entries.
verdict:    incomplete
change:     risks: "24-32 rows", plus a per-slot serial-time risk
microbench: U17
```

## what the plan misses

each item says whether it is part-1-certain (applied to PLAN.md) or part-2-experimental (added to part 2), and its
expected gain with the arithmetic.

1. **prefill deltanet = the decode recurrence (part 1, applied).** value columns of a value head are independent: for
   column j, δⱼ = β(vⱼ − g·Σᵢ Sᵢⱼkᵢ), Sᵢⱼ ← g·Sᵢⱼ + kᵢδⱼ, oⱼ = Σᵢ Sᵢⱼqᵢ. that gives 24 heads x 128 columns = 3,072
   chains per layer per card, each 128 fp32 values.
   - throughput: ~4 fp32 ops per state element per token, 1.57 m ops per layer-token per card. 2,048 tokens x 1.57 m
     / (128 x 82 x 1.7 ghz) ≈ 0.18 ms per layer.
   - chain latency: ~300 clocks per token for a 128-element column on one warp (two 5-step shuffle reductions plus
     fma), so ~0.36 ms per layer per 2,048 tokens.
   - 48 layers: 9-17 ms per 2,048-token chunk against ~1.3 s of gemm time per chunk at 1,600 tok/s: **~1-1.5% of
     prefill**, about what upstream's chunked kernel cost.
   - gain: contract conformance, one deltanet function instead of two, and prefill state bit-identical to decode.
2. **batch invariance (part 1, applied).** no reduction order depends on R or the slot mix: static k-split, attention
   chunks at fixed absolute positions merged in index order, balance by assignment only. cost ~0. gain: speculative
   greedy equals plain greedy token for token, and preemption or a different batch never changes output (upstream
   could not promise this).
3. **one primitive per synchronization relation (part 1, applied across 1.2).**

   | relation | cheapest sm_86 primitive |
   |---|---|
   | copy landed -> consumer warps (same cta) | `cp.async.mbarrier.arrive.noinc` -> `mbarrier.test_wait.parity` spin |
   | ring slot free -> producer (same cta) | `mbarrier.arrive` per consumer warp, or a `bar.arrive`/`bar.sync` pair when the ring has ≤ 7 slots (U8 picks) |
   | fixed warp sub-group inside a cta | named barrier `bar.sync id, n` |
   | tile output ready -> other SMs (same card) | weak stores, then `red.release.gpu.add` on a per-phase counter; `ld.acquire.gpu` poll + `nanosleep` |
   | partial ready -> peer card | push weak stores into the peer's receive slot, then a single-writer `st.release.sys` epoch flag; the peer polls its own memory with `ld.acquire.sys` |
   | host -> device (mailbox) | load issued at cycle start, consumed at cycle end (`ld.acquire.sys` on mapped pinned memory) |
   | device -> host (results) | posted stores once per cycle, then one `st.release.sys`, away from peer release points |
   | inside a warp | `shfl.sync.bfly` fixed-order trees for floats, `redux.sync` for integers, `vote.sync.ballot` for masks |

   gain: no `fence.sc.sys` anywhere (upstream: 28 -> 24 µs per call from that alone), no peer atomics, no proxy
   fences.
4. **the image as the transaction stream, cut at dram chunks (part 1, applied).** whole contiguous 16 kb tiles (64 x
   256-byte chunks), per-lane fragment order inside, the decode bit permutation, scales apart, nan codes rejected,
   lm_head permuted so the draft vocabulary is a view. gain: no ldmatrix or swizzle on the weight path, 1/3 fewer
   decode ops than natural byte order (7 vs 9 per 4 weights), dram row locality, 168 mb of vram per card not spent on
   a draft copy.
5. **dependency-free weights stream during serial phases (part 1, applied).** in_proj_z during the deltanet core
   (15.7 mb, ~17 µs per layer), the gate half of q_proj during attention. gain: at 1 slot the core (~10 µs) disappears
   under it. at 8 slots it covers the core's latency tail. ~0.2-0.6 ms per cycle (1-3%); prior art's upper bound is
   ~1.1 ms.
6. **rms norm folded into the producing epilogue (part 1, applied as part of "one arithmetic").** each output tile's
   epilogue writes its per-row partial sum of squares. the next phase's consumers sum the 40 partials in a fixed
   order. that removes a dependent hop (~1 µs) at each of ~128 phase boundaries, ~0.1 ms per cycle, while staying
   deterministic.
7. **the cycle's shape is an enum, not a runtime decision (part 1, applied).** the producer stream per cycle depends
   only on the draft depth d ∈ {1..4}, and that is a function of the live-slot count under the fixed allocation. four
   compile-time stream programs, chosen per cycle by one enum case.
8. **the resident loop drives itself and yields for prefill (part 1, applied).** the device builds the next cycle from
   the accepted tokens. the host mailbox read overlaps the cycle. a prefill launch is a yield at a cycle boundary
   (~10 µs against ~1 s chunks). gain: no pcie round trip on the critical path (upstream: ~125 µs of host time per
   cycle).
9. **32 rows (part 2, folded into 2.4).** at 90% of p2, 4 n8 tiles use ~128 of ~173 tensor clocks per weight fragment.
   at 8 streams: τ 2.66 -> 3.34 per slot for +0.41 gb per cycle, **~+20% aggregate** (≈ 1,000 -> ≈ 1,200).
10. **prompt rows in decode cycles (part 2, new 2.16).** a prefill chunk is a verify whose rows are all accepted.
    spare rows of every cycle can carry prompt tokens of a slot that is prefilling, and they ride a weight stream
    that is read anyway.
    - one stream decoding at 32 rows leaves 27 spare rows: 27 / 19.5 ms ≈ **1,400 tok/s of prefill** with the
      decoding stream unslowed (plus the prompt's own kv reads). the plan's time-slice instead stops decode for
      25% of the time and gives ~400 tok/s.
    - at 32 rows with every row a prompt row, the decode path prefills at ~32 / 16.4 ms ≈ 1,950 tok/s, comparable
      to the dedicated gemm path's realistic 1,300-2,000. if it holds, the dedicated prefill kernels (gemm,
      attention, exchange) could go entirely: one kernel, bit-identical prefill and decode for free.
    - uncertain: the allocator's policy and long-prompt attention cost.
11. **deferred deltanet write-back (part 2, new 2.12).** write a slot's committed state every k cycles and lazy-replay
    k cycles of accepted tokens. the replay is the same recurrence, so it is bit-identical. k = 4 saves 3/4 x 75.5 mb
    per slot per cycle: 0.45 gb at 8 slots, **~0.55 ms (~2.5%)**. it costs ~k·τ x 0.18 µs of replay per layer,
    ≲ 0.1 ms per cycle, mostly under the core's state stream. snapshot points and preemption force a write.
12. **windowed mtp attention (part 2, new 2.13).** the drafter is outside the lossless boundary, so its attention may
    see only the last w tokens. at 8 slots, d = 2, 30k context, w = 4k: mtp kv reads 0.98 -> 0.13 gb, **~1 ms (~3%)**;
    at 1 stream, 30k context, d = 4: ~0.3 ms. uncertain: acceptance.
13. **prefill/decode bit identity (part 2, new 2.14).** with item 1, the state is identical. make the prefill gemms walk
    k in decode's static split order with the same per-k128 scale step, and use decode's attention chunking: every
    logit is then bit-identical whichever path computed it, and the decode-path ruler becomes an exact gpu-vs-gpu
    check. cost: ~2-3% of prefill (fixed split-k partials: ~71 mb extra per large matrix per 2,048 tokens, ~0.08 ms
    against ~3.3 ms of gemm). moot if item 10 replaces the prefill path.
14. **exact attention P costs nothing in decode** (decision 1): memory-bound, so the three-way split rides under the kv
    stream. only prefill pays (+1.6% at 4k-token prompts, +12% at 32k).

## microbenchmarks for lane `ubench`

each lives in `bench/ubench/`, runs at bring-up (phase b), and settles one decision. expected values come from
docs/HARDWARE.md.

| id | measures | expected (source) | settles |
|---|---|---|---|
| U1 | pure read stream: `cp.async.cg` 16 b + `.L2::256B` + evict_first, whole 16 kb tiles, 82 SMs; sweep producer warps per SM (1, 2, 4, 8) and in-flight depth (2-32 kb per SM); in p2 and, if reachable, p0 | 93-96% of the memory clock's peak: ~850-875 gb/s in p2, 870-900 in p0 [hw-dram §2: c&c 890-900, a40 97.7%] | the 0.3 planning rate (90% of p2), the phase-b gate, the producer warp count |
| U2 | U1 with tiles stored row-major (128 b fragments at 16 kb strides) vs contiguous 16 kb runs | contiguous ≥ strided by a few % [hw-dram §1: 256 b chunk = one bank/row, trc ≈ 45 ns] | the bake's tile-contiguous layout (confirmation) |
| U3 | pstate and clocks.mem under a cuda load; `-lgc F,F -lmc 9751,9751` in one call; `--cuda-clocks=1`; `disable_vrr_memclk_switch=1` | p2 at 9501 mhz [hw-power §3, three reports] | 912 vs 936 as the budget base; the overclock's reference clock |
| U4 | `mma.sync.m16n8k16` bf16->fp32 throughput per SM, alone and interleaved with `fma.rn.bf16x2`; `prmt`/`lop3` throughput | 512 flop/clk/SM (505 measured on ga104 [SUN]); hfma2 pipe unknown [hw-sm §2] | whether the decode's 2 fma per 4 weights cost tensor clocks (+8 per fragment); R = 32 margin |
| U5 | the real streaming gemv (ring + sequence-B decode + mma + scale ffma) on a 44.6 mb matrix at R = 8, 16, 24, 32, 40 | flat gb/s through 32, a drop at 40 [hw-sm §6: 128 vs 160 of ~152-173 clocks] | the row budget: 24 or 32 |
| U6 | crafted bf16 inputs through `mma.m16n8k16` vs the reference's model (8-product blocks, rz, 25-bit window, carry bits, subnormals) | bit-exact agreement with the model [GM §IV, FASI] | geezuz-ref's mma model; noise-floor sizing |
| U7 | all 254 finite e4m3 codes x activations spanning the exponent range, through sequence B + fma(2¹²⁰) + mma, vs the model; also the folded-scale variant | sequence B exact; folded variant loses bits when partials fall below 2⁻¹²⁶ [hw-sm §6] | confirms the decode and the no-fold rule |
| U8 | `mbarrier.arrive` + `test_wait.parity` round trip vs `bar.arrive`/`bar.sync`; `cp.async.mbarrier.arrive.noinc` completion latency; `nanosleep` granularity | unpublished for sm_86; hazy ~60 ns per async barrier on h100 [prior-art §1] | ring signalling primitive |
| U9 | cross-SM hop: weak store, `red.release.gpu.add`, `ld.acquire.gpu` observe, load; full chain latency | several hundred ns per hop, ~1-2 µs per chain [hw-dram §3: l2 138-144 ns unloaded] | whether a 6-8 µs ring covers every phase boundary |
| U10 | nvlink flag ping-pong, one warp per card, 10⁵ iterations: `st.release.sys`/`ld.acquire.sys`, relaxed, `fence.sc.sys` + store | one-way handoff ~1-2 µs (±2x) [hw-link §3, inferred] | exposed tail per exchange (0.15-0.48 ms per cycle); mtp-chain exchange cost; 2.5 stays dead unless > ~100 µs |
| U11 | push R x 20 kb fp32 with 16 b stores (R = 1, 8, 24, 32), sweep sending ctas; same as pulls | ~52.7 gb/s, copy-engine class [hw-link §1] | per-tile push granularity; push over pull |
| U12 | cost of a `.sys` release: idle, with background peer stores, with background host-ring writes | unknown; the cuda guide documents fence interference [hw-link §4] | flag granularity; host-ring write placement |
| U13 | `nvidia-smi nvlink -s`, `topo -m`, p2p attributes (native atomics, performance rank, access) on the open and proprietary modules | NV4, 4 x 14.062 gb/s; atomics unknown [hw-link §1-2] | link health; that peer atomics stay unused |
| U14 | kernel loads of R x 10 kb rows from mapped pinned host memory, numa node 0 vs 1; host-write-to-device-seen latency | ~1-2 µs + ~24 gb/s [hw-link §5] | 2.15 (draft-vocab rows in vram only if a draft step's fetch > ~10 µs); mailbox overlap |
| U15 | `cudaDevAttrMaxPersistingL2CacheSize`, `accessPolicyMaxWindowSize`; survival of `prefetch.global.L2::evict_last` lines under an evict_first 13 gb stream | ~4.5 mb (inferred) [hw-sm §4] | how much of the run-ahead l2 can carry (0 or 2-4 µs) |
| U16 | streaming loop with consumers stalled for t µs: dram idle time vs ring size 32-80 kb per SM | idle once t > ring bytes / bandwidth, ~6-8 µs at 64-80 kb [hw-sm implication 5] | ring size and shared-memory split |
| U17 | deltanet recurrence: per-token chain latency (one warp per column vs 8 lanes per column), throughput over 3,072 chains; decode core at 8 slots | ~300 clocks per token per chain; the 8-slot core at ≥ 90% of the stream rate, ~30 µs per layer [prior-art §7, model] | prefill recurrence cost (~1-2%); per-slot exposed serial time |
| U18 | a decode-like loop at 400 w: clocks.sm, sw_power_cap counter, power; then locked `-lgc` one step below | ~1.8-1.9 ghz under water (inferred) [hw-power §2, implications] | the lock value; t_flops scale; R = 32 margin |
| U19 | memory offset sweep (+250 linux units per step): pure-read gb/s, power, sm clock, clock-event reasons, junction temperature if allowed | gb/s linear to the edr knee; 21-22.6 gb/s mining-stable [hw-dram §5] | the overclock offset (knee minus two steps) |
| U20 | throughput of the shared software exp/log/rsqrt vs mufu, in a softmax and an rms norm | ~10-30 fp32 ops vs a 1/8-rate mufu op: comparable [hw-sm §2] | confirms the rounding pins cost nothing measurable |
| U21 | decode split-k attention at r = 3 and 5 rows per slot with one vs three PV mmas | memory-bound at r = 3, near the ridge at r = 5 [model] | decision 1's decode cost |
| U22 | in-situ prefill gemm: 2,048 tokens on a 44.6 mb e4m3 matrix with the register decode, bf16->fp32 | 50-60 tflops [prior-art §0: marlin ~63, fa2 54-57] | the prefill planning number (1,600 vs 2,000) |

## the arithmetic ([model])

per card. bytes from [ckpt]. τ from upstream:
- 1 stream: code 4.46, prose 2.16, at d = 4; per-position chain acceptance p ≈ 0.943 code, 0.563 prose;
- 8 streams: p ≈ 0.881 (from 610 tok/s at 43.8 ms with 3 drafts per slot), giving τ(2) ≈ 2.66 and τ(3) ≈ 3.34.

| cycle | bytes | at 820 gb/s | + exposed | code | prose / aggregate |
|---|---|---|---|---|---|
| 1 stream, d = 4, 4k context | 15.41 gb | 18.8 ms | 19.1-19.4 ms | ~230 | ~112 |
| 8 streams, 24 rows, d = 2, 4k each | 16.69 gb | 20.3 ms | 20.8-23.6 ms | — | ~900-1,020 |
| 8 streams, 32 rows, d = 3, 4k each | 17.16 gb | 20.9 ms | 21.4-24.2 ms | — | ~1,100-1,250 |
| 8 streams, 24 rows, 30k each | 24.33 gb | 29.6 ms | ~30.6 ms | — | ~700 |
| 1 stream at 80% of p2 (prior art's view) | 15.41 gb | 21.1 ms | ~21.5 ms | ~205 | ~100 |
| 1 stream at 21 gb/s, 90% | 15.41 gb | 17.0 ms | 17.3-17.6 ms | ~255 | ~123 |

step 1: upstream q8_0 moved ~14.8 gb per 1-stream cycle in 24.9 ms (595 gb/s). our image's 15.2 gb at that rate is
25.6 ms, minus ~0.5-1 ms of nvlink exchange: ~25 ms.

tensor clocks per 256-byte weight fragment per SMSP: 152 (936 gb/s, 1695 mhz), 173 (820, 1695), 194 (820, 1900),
143 (820, 1395). one n8 tile costs 32.
