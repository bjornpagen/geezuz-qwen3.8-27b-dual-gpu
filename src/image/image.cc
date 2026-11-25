// SPDX-License-Identifier: AGPL-3.0-only
//
// The baked weight image: one file per card, holding that card's extents in the order its producers stream them,
// stamped with the model description and checkpoint they come from and the digest of every payload byte. The baker
// writes it; the engine maps it and accepts it whole or not at all.
export module engine.image;

export import :catalog;
export import :checkpoint;
export import :description;
export import :error;
export import :layout;
export import :model_catalog;
export import :read;
export import :write;
