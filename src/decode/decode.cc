// SPDX-License-Identifier: AGPL-3.0-only
//
// Decoding on both cards: the resident decode loops, the host's side of their rings and mask buffers, the ledger of
// what the cards hold per slot, and the kv pool's commands lowered for the kernel. A cycle is `begin` (the descriptor
// goes out and the body starts streaming), `publish` (the grammar's masks, computed meanwhile) and `try_complete` (both
// cards' agreeing verdict, as the batch's outcome).
export module engine.decode;

export import :error;
export import :ledger;
export import :commands;
export import :session;
export import :engine;
