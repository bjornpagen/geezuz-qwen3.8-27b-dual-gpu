# tests/

Dialect zone: one directory per component under test, `tests/<name>/`, registered with `engine_test`,
`engine_death_test` or `engine_compile_fail` (AGENTS.md §4.3, §31).

- `testing/` — `engine.testing`: the harness (`Case`, `report`, a seeded generator).
- `compile_fail/` — misuses that must not compile: a discarded `Result`, an unclassified error type, a handle used
  with another table.

Tests check the contract against an oracle and search rather than enumerate. Run a lane's tests with
`ctest --test-dir build/lanes/<lane>/<preset> -R '^<name>'`.
