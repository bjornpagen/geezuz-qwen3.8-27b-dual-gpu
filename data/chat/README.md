<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# data/chat

Reference fixtures for `engine.chat`: chat requests rendered by the checkpoint's chat template, model replies parsed by
its response template, and JSON numbers as the template's `tojson` writes them. `foreign/chat_fixtures/` embeds them
into the chat tests (`tests/chat/`); the `chat.checkpoint` test compares the SHA-256s recorded in every file with the
checkpoint's own files.

## Source

The checkpoint `Qwen/Qwen3.8-27B-FP8` at commit `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`:

| file | role | SHA-256 |
|---|---|---|
| `chat_template.jinja` | renders chats | `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375` |
| `tokenizer_config.json` | its `response_template` parses replies | `207c03c41a06a9c3b0cc527d8a56115159c1eac6b59a4ffae85b646e773d074e` |
| `tokenizer.json` | tokenizes | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |

The references, from PyPI: transformers 5.24.1 (`apply_chat_template`, and `parse_response` / `ResponseParser`
with the response template), tokenizers 0.23.3, jinja2 3.1.6, regex 2026.11.24, under CPython 3.13.16 (Unicode
15.1). Transformers renders with `trim_blocks`, `lstrip_blocks` and its own `tojson`: Python's `json.dumps` with
`ensure_ascii=False` and the default `", "` and `": "` separators (not jinja's HTML-escaping, key-sorting filter).

## How it was made

Once, outside the repository, with the script in the coordinator's scratchpad (`chat/make_fixtures.py`), from that
directory:

```sh
uvx --with transformers==5.24.1 --with tokenizers==0.23.3 --with jinja2==3.1.6 --with regex==2026.11.24 \
    python -I make_fixtures.py ~/models/Qwen3.8-27B-FP8 <repo>/data/chat/fixtures
```

The run is deterministic (one seeded generator, `random.Random(20261125)`); a second run writes identical files.
Outputs (SHA-256): `fixtures/render.json` `cd931c482ecedfab92bebcf58d9787043055651c581371ba0b4106b12302c383`,
`fixtures/parse.json` `36137073609a09189782dd928c90a4fd3193d55f0e2cdd3f3c39162b78942904`,
`fixtures/numbers.json` `9570fd68cae95cebe02b73f33f9fc5200f23f184825329518ce23a67de5326a6`.

What each part does, so it can be rewritten if the script is lost:

- **render** (431 cases): 31 written by hand (single and multi-turn, thinking on and off, every reasoning effort,
  past reasoning kept and dropped, system messages empty, blank and absent, tools with nested schemas and unicode,
  parallel calls, tool responses alone and grouped, `content: null`, text parts, arguments as an object, as a
  string and empty, every JSON type and number form, old-style `<tool_response>` user turns, a tool response first,
  client text spelling control tokens, spellings that span client and template text, and the template's three
  refusals) and 400 random conversations, every fifth with control-token spellings in its client text. Each case is
  a full OpenAI request body, written as raw JSON text (tools and arguments with varied spacing, `\u` escapes and
  number literals such as `1E2`, `-0`, `5e0`), so the engine decodes it as the endpoint does. Python reads the same
  text back and the template renders it with `add_generation_prompt=True`. The reference ids are
  `tokenizers` on that text, added-token spellings read as tokens. The engine ids come from a model of the engine's
  renderer written in the script (template text, client text and placed tokens as separate pieces): the script
  checks that the model's text equals the template's on every case, encodes each run between tokens with the same
  tokenizer.json stripped of its added tokens, recognizing spellings only inside template text, and checks that the
  two id sequences are equal exactly where the reference reads no added token from client text.
- **parse** (683 cases): 33 written by hand, 150 assistant turns of random conversations as the template writes
  them (the model's text after the generation prompt's `<think>\n`), and 500 random replies built from fragments of
  the format with noise (partial and misplaced markers, every whitespace, junk in calls, missing close tags,
  repeated parameters) and typed parameters (integer, number, boolean, null, object, array, unions, `nullable`)
  with raw values in every form Python's `int`, `float` and `json.loads` take or refuse. The expected parse is the
  reference's `ResponseParser` fed the generation prompt as its prefix, with the request's tools; a call's arguments
  are its argument dict written by `json.dumps(..., ensure_ascii=False)`. Replies where the engine differs on
  purpose (below) are left out.
- **numbers** (3000 cases): 50 edge literals (signed zeros, the double range's ends, subnormals, halfway cases,
  `1e16` and `1e-5` where `repr` switches form, integers past 64 bits) and random literals with up to 26 digits and
  exponents up to ±340, each with `json.dumps(json.loads(literal))`, or `out_of_range` where Python reads an
  infinity.

## Where the engine differs from the reference, on purpose

- **Reasoning regions join.** The response template keeps only the last reasoning region; a reasoning region after
  streamed reasoning would take that reasoning back. The engine joins reasoning regions as the template joins
  content regions. Replies with at most one non-empty reasoning region (every reply the template's format writes)
  parse identically.
- **Object and array casts read strict JSON.** The reference casts with Python's `json.loads`, which also reads NaN,
  infinities, numbers past the double range, repeated keys and lone surrogate escapes, and then writes arguments
  that are not JSON. The engine reads with `engine.json` (bounded by `json::fragment_bounds`) and keeps such a value
  as text.
- **Client text cannot spell control tokens.** The reference tokenizes the rendered string, so a message that spells
  `<|im_end|>`, `<think>`, `<tool_call>` or `<tool_response>` becomes that token; the engine reads every spelling in
  client text as characters. This includes old-style tool responses sent as user text wrapped in
  `<tool_response>...</tool_response>`, which the template still recognizes as tool responses by their text.

## Files

One JSON document each: `{"template_sha256", "tokenizer_config_sha256", "cases": [...]}`, compact, UTF-8.

- `render.json` cases: `name`, `request` (the request body's text); then either `error` (`NoUserQuery` or
  `MisplacedSystem`, the template's refusals), or `text` (the template's rendering), `ids` (the engine's),
  `reference_ids` and `forged` (whether the reference reads an added token from client text).
- `parse.json` cases: `name`, `thinking` (whether the generation prompt left the model inside `<think>`), `tools`
  (each tool's JSON text), `output` (the model's text), and the parse: `reasoning`, `content`, `calls` (`name`,
  `arguments`).
- `numbers.json` cases: `literal`, and `dumps` or `out_of_range: true`.
