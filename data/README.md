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
frequencies, special tokens forced in, top ids kept, then sorted ascending). Upstream gave each card every second
entry; this engine does the same through `engine.vocabulary` (each card's lm_head rows are its half of the draft ids,
then its half of the rest, PLAN.md 1.3), and the image digest covers the file's SHA-256.

## tokenizer/

The tokenizer's merges and vocabulary in the engine's own format (`qwen.tok`, embedded in the binary) and the
fixtures checked against hugging face; `tokenizer/README.md` records how they were made.

## chat/

The chat-format fixtures rendered from the checkpoint's own templates; `chat/README.md` records how.

## checkpoint/

The SHA-256 hugging face publishes for every checkpoint file at the pinned commit; the baker refuses anything else.

## platform/

The systemd units and drop-ins docs/BRINGUP.md installs on the linux box.
