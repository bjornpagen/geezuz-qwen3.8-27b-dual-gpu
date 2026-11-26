# tests/

Dialect zone: one directory per component under test, `tests/<name>/`, registered with `engine_test`,
`engine_death_test`, `engine_checkpoint_test` or `engine_compile_fail` (AGENTS.md §4.3, §31). Test doubles are named
as doubles and live here or in `foreign/driver_double/`.

- `testing/` — `engine.testing`: the harness (`Case`, `report`, a seeded generator).
- `compile_fail/` — misuses that must not compile: a discarded `Result`, an unclassified error type, a handle used
  with another table, and the misuses each component pins.
- `numerics/` — each kernel family's host emulation held to the cpu reference bit for bit.
- every other directory tests the component of the same name; `checkpoint`-labelled tests read the real checkpoint
  and run with `ctest --preset checkpoint`.

Tests check the contract against an oracle and search rather than enumerate. Run a lane's tests with
`ctest --test-dir build/lanes/<lane>/<preset> -R '^<name>'`.
