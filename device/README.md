# device/

Quarantine zone: CUDA device code, compiled by clang straight to ptx for sm_86 at ptx isa 9.3, with no CUDA SDK.
Device code cannot import modules, so it is headers (`.h`) and translation units (`.cu`), included relative to this
directory. AGENTS.md §17.

- `target.h` — the build graph's facts entering device code, once: `ptx_isa`, `sm`, `checks`, `profiler`.
- `model.h` — the model description, the single source of every model and layout constant.
- `prims/` — named primitives; the only place inline ptx exists.
- `abi/` — the device half of the host/device abi, checked by the device compiler; its image is the header every
  engine image shares.

A device image is one translation unit: `engine_device_ptx(<name> SOURCE <tu>.cu)` in `device/<name>/CMakeLists.txt`
produces `<build>/ptx/<name>.ptx`, gated by `ptxas` on linux and by clang-tidy in `lint`. A foreign component embeds
it (§17.1). Its clang-tidy profile is the quarantine profile (`.clang-tidy` links to `cmake/quarantine.clang-tidy`).
