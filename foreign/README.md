# foreign/

Quarantine zone for external interfaces whose headers, macros or ABI may not leak into dialect code: the CUDA driver
api (declared here, `libcuda.so.1` loaded with `dlopen`), the stdexec boundary, the embedded ptx images and the
model-description re-export. Each component exports a narrow, typed, owning module upward. Ownership rules hold in
full; raw pointers are transient ABI arguments only. AGENTS.md §5.2.

- `exec/` — `engine.exec`: the sender vocabulary over stdexec's own `stdexec` module, plus `wait<E>`. The only file
  that spells `stdexec::`.
- `ptx/` — `engine.ptx`: the `Image` type, `consteval` header validation, and the device abi image every other image
  is checked against.
