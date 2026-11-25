// SPDX-License-Identifier: AGPL-3.0-only
//
// The checkpoint converter: verifies the checkpoint against its published SHA-256s, checks every tensor against the
// model description, proves that attention cannot overflow, and writes each card's image as the decode schedule lays
// it out (tensor-parallel split, tile encoding, vocabulary permutation) and the host image. Every byte it writes is a
// copy or a permutation of a checkpoint byte.
export module engine.bake;

export import :bound;
export import :error;
export import :plan;
export import :produce;
export import :qwen;
export import :source;
export import :verify;
export import :write;
