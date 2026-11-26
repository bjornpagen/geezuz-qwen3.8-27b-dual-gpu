# device/

Quarantine zone: CUDA device code, compiled by clang straight to ptx for sm_86 at ptx isa 9.3, with no CUDA SDK.
Device code cannot import modules, so it is headers (`.h`) and translation units (`.cu`), included relative to this
directory. AGENTS.md §17.

- `target.h` — the build graph's facts entering device code, once: `ptx_isa`, `sm`, `checks`, `profiler`.
- `model.h` — the model description: every model and layout constant, the rounding points and summation orders, the
  memory plan.
- `prims/` — named primitives (the only place inline ptx exists), budgets, the tile encoding, the fixed
  transcendentals, and their self-check kernels.
- `abi/` — the device half of the host/device abi; its image is the header every engine image shares.
- `schedule/` — the decode cycle as compile-time data: phases, stream order, image layout and offset table, the
  split over the SMs, workspace geometry.
- `ring/` — the host/device mailbox protocol.
- `sampling/` — the sampling arithmetic shared with the host.
- `gemv/` — the weight-stream matmuls: the decode gemv and the prefill gemm, plus their check and bench kernels.
- `attn/` — attention: the kv block layout, split-k decode attention, causal prefill attention, the rope table.
- `deltanet/` — the gated deltanet recurrence, lazy replay and snapshot layouts, decode and prefill.
- `decode/` — the step-1 persistent decode megakernel.
- `prefill/` — prefill's row and gemm kernels and its per-chunk layout.
- `soak/` — the on-device image hash and the kv pattern test for the soak.
- `ubench/` — the microbenchmark kernels.

A device image is one translation unit: `engine_device_ptx(<name> SOURCE <tu>.cu)` in `device/<name>/CMakeLists.txt`
produces `<build>/ptx/<name>.ptx`, gated for pinned rounding and declared budgets on every platform, by `ptxas` on
linux, and by clang-tidy in `lint`. A foreign component embeds it (§17.1). Its clang-tidy profile is the quarantine
profile (`.clang-tidy` links to `cmake/quarantine.clang-tidy`).
