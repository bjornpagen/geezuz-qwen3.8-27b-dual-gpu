// SPDX-License-Identifier: AGPL-3.0-only
//
// Both cards for the pipeline: the adapter from decode and prefill to the pipeline's cards, and the bring-up that
// opens them from a directory of baked images, refusing a machine or a clock the engine does not run on.
export module engine.cards;

export import :adapter;
export import :start;
