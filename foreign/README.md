# foreign/

Quarantine zone for external interfaces whose headers, macros or ABI may not leak into dialect code: the CUDA driver
api (declared here, `libcuda.so.1` loaded with `dlopen`), the stdexec and simdjson boundaries, the embedded ptx images
and data files, and the device headers' re-exports. Each component exports a narrow, typed, owning module upward.
Ownership rules hold in full; raw pointers are transient ABI arguments only. AGENTS.md §5.2.

- `exec/` — `engine.exec`: the sender vocabulary over stdexec's own `stdexec` module, plus `wait<E>`.
- `card_scheduler/` — `engine.card_scheduler`: a stdexec scheduler per card over driver events; `polled`.
- `simdjson/` — `engine.simdjson`: simdjson on-demand, exceptions off, one kernel per platform.
- `ptx/` — `engine.ptx`: the `Image` type, `consteval` header validation, the abi image, budget records.
- `driver/` — `engine.driver`: the CUDA driver api and NVML subset, declared and loaded at run time.
- `driver_double/` — `engine.driver_double`: the test double behind the same tables (tests only).
- `model/` — `engine.model`: `device/model.h` re-exported.
- `prims/`, `schedule/`, `sampling/`, `gemv/`, `attn/`, `deltanet/`, `decode/`, `prefill/`, `soak/`, `ubench/` — the
  device headers re-exported for host code, with each device image embedded and its kernels declared.
- `tokenizer_image/`, `draft_vocabulary/`, `bake_data/` — embedded data: the tokenizer, the draft vocabulary, the
  checkpoint's pinned SHA-256s.
- `tokenizer_fixtures/`, `chat_fixtures/` — embedded test fixtures (tests only).
