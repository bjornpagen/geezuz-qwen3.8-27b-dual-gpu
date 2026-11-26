# src/

Dialect zone: the engine's components. Each is one named module `engine.<name>` in `src/<name>/`, built as target
`engine_<name>` by the `CMakeLists.txt` in that directory, which lists every module file and links its module
dependencies. Directories are discovered; adding a component touches no shared file. AGENTS.md §4.2 and §6.

foundations
- `core/` — `engine.core`: `Result` (an `expected` whose error classifies retryability), `invariant`, `narrow`,
  `try_narrow`, `alternative`.
- `handle/` — `engine.handle`: generational handle tables (slots, kv blocks, connections, requests).
- `sha256/` — `engine.sha256`: streaming and compile-time SHA-256 over `engine.sha256_block`.
- `safetensors/` — `engine.safetensors`: typed decode of a safetensors file, byte ranges tiling its data.
- `mma_model/` — `engine.mma_model`: the bit-exact host model of sm_86's bf16 mma with fp32 accumulation.

the checkpoint and the image
- `image/` — `engine.image`: the baked card and host images: catalog from the schedule, writer, verifying reader.
- `bake/` — `engine.bake`: the checkpoint converter (verify, split, tile, permute, write).
- `vocabulary/` — `engine.vocabulary`: the one map between each card's lm_head rows and token ids.
- `ref/` — `engine.ref`: the cpu reference, the rulers and the golden-hash tooling.

text and the api
- `tokenizer/` — `engine.tokenizer`: qwen byte-level bpe, exact against hugging face's tokenizer.
- `chat/` — `engine.chat`: the qwen3.8 chat format rendered to ids, the streaming reply parser, the cast rules.
- `regex/` — `engine.regex`: the regex subset grammars use, compiled to automata.
- `grammar/` — `engine.grammar`: constrained decoding: lark, regex and json-schema frontends, the composed reply
  grammar, the per-slot matcher and its token masks.
- `json/` — `engine.json`: bounded json over simdjson on-demand; the responses request, response and event shapes.
- `http/` — `engine.http`: bounded http/1.1 with bodies, server-sent events and backpressure, over `engine.net`.
- `openai/` — `engine.openai`: the openai responses api: validation, the service, its front for the pipeline.
- `store/` — `engine.store`: stored responses, bounded and evicted least recently used.

scheduling and the engine
- `sampling/` — `engine.sampling`: exact sampling and exact speculative verification over one random stream.
- `kv/` — `engine.kv`: the paged kv pool shared with deltanet snapshots.
- `prefix/` — `engine.prefix`: the radix prefix cache with deltanet snapshot points.
- `sched/` — `engine.sched`: slots, admission, preemption, chunked prefill, turn replay, the cycle plan.
- `device/` — `engine.device`: the two cards over the driver api: contexts, memory, transfers, launches, the ring,
  telemetry.
- `decode/` — `engine.decode`: the step-1 decode engine on the cards (host side of the resident megakernel).
- `prefill/` — `engine.prefill`: chunked prefill over both cards.
- `cards/` — `engine.cards`: decode and prefill fitted to the pipeline's cards seam; the shared bring-up.
- `pipeline/` — `engine.pipeline`: the engine loop as senders between the api and the cards.
- `platform/` — `engine.platform`: the accepted memory clock, the start-time clock check, the soak.
