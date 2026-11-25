# data

## draft_vocab_32k.bin

The draft vocabulary: the token ids the mtp draft head is allowed to score. Restricting the draft lm_head to these
rows only changes which drafts are proposed (acceptance rate), never the output, because verification always uses
the full vocabulary.

Byte layout:

- 131072 bytes: 32768 little-endian `int32` ids, no header, no padding.
- ids are strictly ascending and unique; the smallest is 0 and the largest is 248076, inside the 248,320-entry
  vocabulary.
- the set is every control and user-defined (special) token, which are always draftable, plus the most frequent
  tokens over a text and code corpus, up to 32768 ids in total; the list is sorted by id, not by frequency.

Origin: written by upstream geezuz's `tools/vocab_freq` (llama.cpp tokenization of a file list, counting token
frequencies, special tokens forced in, top ids kept, then sorted ascending). Upstream consumed it by reading
`int32`s until end of file, then giving each card every second entry so both cards draft over equally many rows.
