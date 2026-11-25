// SPDX-License-Identifier: AGPL-3.0-only
//
// The baked images: one per memory the engine loads whole (each card's, and pinned host memory), holding that
// memory's extents in the order the schedule lays them out, stamped with the model description and checkpoint they
// come from and the digest of every payload byte. The baker writes them; the engine maps each and accepts it whole or
// not at all.
export module engine.image;

export import :catalog;
export import :catalogs;
export import :checkpoint;
export import :description;
export import :error;
export import :layout;
export import :read;
export import :write;
