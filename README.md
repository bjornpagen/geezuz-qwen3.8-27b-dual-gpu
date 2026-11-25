# Geezuz

One lossless fp8 engine for Qwen3.8-27B on two NVLinked RTX 3090s, tensor parallel 2. It runs Qwen's own
`Qwen3.8-27B-FP8` checkpoint as w8a16: e4m3 weights converted exactly to bf16, fp32 accumulation, bf16 activations
and kv cache, exact speculative verification and exact sampling. One model, one weight format, two cards, eight
request slots, sm_86; nothing is configurable at run time.

## Status

Mid-rewrite, code-only phase. The previous engine (a GGUF k-quant megakernel descended from open-jet) has been
removed from this tree. The new engine is being written without a GPU: it builds and type-checks, and nothing has
run on a GPU yet. There are no measurements for the new engine. [PLAN.md](PLAN.md) is the spec; [AGENTS.md](AGENTS.md)
holds the rules the code follows.

## Build

```sh
cmake --preset dev && cmake --build --preset dev
```

Needs a recent clang with libc++, cmake and ninja. Device code is compiled by clang straight to PTX, so no CUDA SDK
is needed to build. Running needs an NVIDIA driver and two RTX 3090s.

## Layout

See section 1.3 of PLAN.md. `data/` holds the 32k draft vocabulary and its format.

## License and credits

The megakernel is the work of **L-Forster** in [open-jet](https://github.com/L-Forster/open-jet): the decode megakernel, the prefill kernels, the loader, the server and CLI, the one-GPU benchmarks, the quality tooling and the draft vocabulary. This repository is a modified version of `megakernel/` at commit `2993f47`, imported with its git history, licensed under the GNU Affero General Public License v3.0 and distributed under the same license (see `LICENSE`).

Modifications, October 2026: sources moved to the repository root; per-tensor Q4_K / Q5_K / Q6_K / Q8_0 weights in decode and prefill; tensor parallelism over two GPUs via PCIe P2P (decode and prefill) with the two-GPU tuning; batched decode of up to 8 slots (tensor-core GEMVs, int8 exchange, server scheduler); shared paged KV pool with prefix cache and preemption; the benchmarks and quality checks.
