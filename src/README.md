# src/

Dialect zone: the engine's components. Each is one named module `engine.<name>` in `src/<name>/`, built as target
`engine_<name>` by the `CMakeLists.txt` in that directory, which lists every module file and links its module
dependencies. Directories are discovered; adding a component touches no shared file. AGENTS.md §4.2 and §6.

- `core/` — `engine.core`: `Result` (an `expected` whose error classifies retryability), `invariant`, `narrow`,
  `try_narrow`.
- `handle/` — `engine.handle`: generational handle tables for slots, kv blocks and connections.
