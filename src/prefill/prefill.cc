// SPDX-License-Identifier: AGPL-3.0-only
//
// Prompt prefill on both cards: what a chunk borrows from the decode engine's memories, the fixed sequence of kernel
// launches a chunk is on each card, and the chunks' runner, which owns the prefill workspaces and programs, keeps each
// slot's carried final-norm row, and enqueues a chunk's sequence on both cards' streams.
export module engine.prefill;

export import :memories;
export import :steps;
export import :chunks;
