// SPDX-License-Identifier: AGPL-3.0-only
//
// The SHA-256 compression function on the machine's own SHA instructions: the bake hashes a 29 GB checkpoint and two
// ~14 GB images, and the engine hashes each image's payload at every load, which software rounds would take minutes
// for. One implementation per machine, chosen in the build graph; dialect code reaches it through `engine.sha256`.
export module engine.sha256_block;

export import :constants;
export import :compress;
