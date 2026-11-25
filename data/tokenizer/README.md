<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# data/tokenizer

The tokenizer image the engine builds in (`qwen.tok`, embedded by `foreign/tokenizer_image/`) and the reference
fixtures its tests check against (`fixtures/`, embedded by `foreign/tokenizer_fixtures/`). `engine.tokenizer` reads
the image with typed errors (`Tokenizer::parse`); the `tokenizer.checkpoint` test re-derives the vocabulary, the merges
in rank order and the added tokens from the checkpoint's tokenizer.json and compares them with the image.

## Source

- `tokenizer.json` of `Qwen/Qwen3.8-27B-FP8` at commit `c3a462cc3d38cf351f3b26cbd763469eeb74ef81`, SHA-256
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
- the reference implementation: Hugging Face `tokenizers` 0.23.3 (PyPI), the oracle for every table and fixture.

## How it was made

Once, outside the repository, with the scripts in the coordinator's scratchpad (`tokenizer/`), from that directory:

```sh
uvx --with tokenizers==0.23.3 python -I probe_classes.py     # -> classes.json
uvx --with tokenizers==0.23.3 python -I unicode_model.py     # -> unicode_tables.json
uvx --with tokenizers==0.23.3 python -I convert.py ~/models/Qwen3.8-27B-FP8/tokenizer.json <repo>/data/tokenizer/qwen.tok
uvx --with tokenizers==0.23.3 python -I fixtures.py ~/models/Qwen3.8-27B-FP8/tokenizer.json <repo>/data/tokenizer/fixtures
```

Outputs (SHA-256): `qwen.tok` `a91b9f6a5bdac6fbb4f888ad6a9c2b5a96f637b004a577c0fab7be1b31395d30`,
`fixtures/encode.fix` `cfb9400056a8ad9892394c3a0b2547808dd1d0e720a9a4f6adfb2b200f61f06b`,
`fixtures/decode.fix` `64eae616840c1a12bf59ae0cfc8649526c96f54874eeec404d308a338274b998`.

What each step does, so it can be rewritten if the scripts are lost:

- **classes**: the membership of every codepoint (surrogates excluded) in `\p{L}`, `\p{M}`, `\p{N}` and `\s`, probed
  through `tokenizers.pre_tokenizers.Split(Regex(...), "removed")`, so the sets are the reference regex engine's
  (Oniguruma; unicode 16 letters), not the platform's or Python's. The four sets are disjoint.
- **unicode tables**: canonical decompositions from the reference `NFD` normalizer on each codepoint (Hangul
  excluded: it is arithmetic); combining classes found by probing the reference's canonical reordering against one
  reference mark per class; compositions are Python's two-character canonical decompositions that the reference `NFC`
  composes. The reference normalizer's unicode version is older than both Python's and its regex engine's, which is
  why nothing here comes from a platform table. The NFC model built from these tables agrees with the reference on
  every codepoint (NFD and NFC) and on 300,000 random sequences of starters, marks, composites and jamo.
- **convert**: asserts tokenizer.json's pipeline is the one implemented (no truncation or padding, NFC, the split
  pattern, byte-level stages without prefix space or regex, BPE without dropout, unk, prefixes or byte fallback),
  maps every vocabulary entry back through the byte-level alphabet to its bytes, and writes the merges by rank.
- **fixtures**: a corpus (English, Python, C++ with LF, CRLF and CR line ends, JSON, a diff, a rendered chat template,
  CJK, Arabic, Devanagari, Thai, Greek, Hebrew, Cyrillic, emoji sequences, NFC pairs, Hangul jamo, unicode 16
  letters, every whitespace, contractions in every case-folding, every combining class, every added token and
  fragments of them, long runs) plus 6,000 swarm-random strings over the hard character classes. Each is encoded
  twice: with the reference tokenizer (added-token spellings read as tokens) and with the same tokenizer.json
  stripped of its added tokens (spellings read as characters). Decode cases are 4,000 random id sequences (raw byte
  tokens, so invalid utf-8 is common; random ids; encodings of corpus text) with the reference's
  `decode(ids, skip_special_tokens=False)`.

## `qwen.tok`, format version 1

Little-endian. Header: the 8 bytes `gzqwntok`, then nine `u32`: version (1), token count, token byte total, merge
count, class run count, combining run count, decomposition count, decomposed codepoint total, composition count.
Then, in order, each a sequence of `u32` unless stated:

| section | layout |
|---|---|
| token ends | one per token: the end offset of its bytes in the token bytes |
| token bytes | every token's raw bytes in id order, zero-padded to 4 bytes |
| merges | `left, right, result` per merge, in rank order |
| class runs | `start << 3 \| class` (0 other, 1 letter, 2 mark, 3 number, 4 whitespace), ascending from 0 |
| combining runs | `start << 8 \| combining class`, ascending from 0 |
| decomposition keys | ascending codepoints that decompose |
| decomposition ends | end offset of each key's decomposition in the decomposed codepoints |
| decomposed codepoints | full canonical decompositions, concatenated |
| compositions | `first, second, composite`, ascending by `(first, second)` |

The token and merge counts must equal the compiled vocabulary (248,044) and merge count (247,587); the unicode tables
are bounded by named maxima; the length must be exactly what the header implies. The added tokens are not in the
image: they are compiled in (`engine::tokenizer::Added`) and checked against tokenizer.json by the checkpoint test.

## `fixtures/`

Little-endian `u32` throughout, byte strings zero-padded to 4 bytes.

- `encode.fix`: `gzqwnenc`, version 1, case count; per case: spelling (0 characters, 1 token), text byte count, id
  count, the text, the ids.
- `decode.fix`: `gzqwndec`, version 1, case count; per case: id count, text byte count, the ids, the text.
