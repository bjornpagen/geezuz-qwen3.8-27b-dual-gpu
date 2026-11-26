# apps/

Dialect zone: the programs, each in `apps/<name>/` with its own `CMakeLists.txt` declaring one executable that links
the components it needs. AGENTS.md §4.2.

- `geezuz-server` — the openai responses api server: `geezuz-server <image-dir> <address>`.
- `geezuz` — the cli over the same pipeline: `respond <image-dir> <request.json>`, `say <image-dir> <prompt-file>`.
- `geezuz-bake` — the checkpoint converter: `geezuz-bake <checkpoint-dir> <output-dir>`.
- `geezuz-ref` — the cpu reference and the rulers.
- `geezuz-ubench` — the bring-up microbenchmarks.
- `geezuz-soak` — the memory soak and its integrity checks.
