<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# the checkpoint's published SHA-256s

`sha256.txt` lists the SHA-256 that hugging face publishes for every LFS file of `Qwen/Qwen3.8-27B-FP8` at commit
`c3a462cc3d38cf351f3b26cbd763469eeb74ef81` (the 64 `layers-*.safetensors`, `mtp.safetensors`, `outside.safetensors`
and `tokenizer.json`), one `<sha256>  <file>` line each, sorted bytewise by name. it was read from
`https://huggingface.co/api/models/Qwen/Qwen3.8-27B-FP8/revision/main?blobs=true` (the `lfs.sha256` of each sibling) on
2026-11-25, when `main` was that commit. a full download was verified against it with `shasum -a 256 -c`.

the baker verifies every file it reads against this list before using a byte of it; a re-uploaded checkpoint is
refused, never half-baked. integrity is SHA-256 only: the checkpoint's own md5 and crc32 lists are ignored.
