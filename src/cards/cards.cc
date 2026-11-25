// SPDX-License-Identifier: AGPL-3.0-only
//
// Both cards for the pipeline: the adapter from decode and prefill to the pipeline's cards, the bring-up that opens
// them from a directory of baked images (refusing a machine or a clock the engine does not run on), and the engine as
// the programs start it, the pipeline over the cards behind the responses api's front.
export module engine.cards;

export import :adapter;
export import :start;
export import :serving;
