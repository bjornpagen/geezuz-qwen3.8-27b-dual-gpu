// SPDX-License-Identifier: AGPL-3.0-only
//
// What a bake is generic over: the files it reads and their published digests, the tensors it places, the three image
// catalogs and the placements that fill them, the row-block selections and the vocabulary permutation. The model's own
// plan is all compile-time data from engine.model, engine.schedule and engine.vocabulary; a test supplies a tiny plan of
// the same shape, so the bake's every step runs on checkpoints a test can write.
export module engine.bake:plan;

import std;
import engine.image;
import engine.model;
import engine.schedule;
import engine.sha256;

namespace engine::bake {

/** A checkpoint file the bake reads, and the SHA-256 it was published with. */
export struct Pinned {
	std::string_view name;
	sha256::Digest digest;
};

export template<class P>
concept Plan = requires(schedule::Select select, std::size_t tensor, model::Card card, std::size_t row) {
	{ std::span<model::Tensor const>{P::inventory} };
	requires image::Catalog<typename P::CardZero> && image::Catalog<typename P::CardOne> && image::Catalog<typename P::Host>;
	/** Card `card`'s placements, extent for extent of its catalog. */
	{ P::placements(card) } -> std::same_as<std::span<schedule::Placement const>>;
	/** The inventory index of each host catalog extent: a host tensor, whole. */
	{ std::span<std::size_t const>{P::host_tensors} };
	{ P::blocks(select, tensor, card) } -> std::same_as<schedule::Blocks>;
	/** The token a Vocabulary-split tensor's row `row` on `card` holds. */
	{ P::token(card, row) } -> std::convertible_to<std::size_t>;
	/** The attention value projection and the norm before it, by inventory index: the overflow proof's inputs. */
	{ P::v_proj } -> std::convertible_to<std::size_t>;
	{ P::input_norm } -> std::convertible_to<std::size_t>;
};

/** A plan for a published checkpoint: the files the bake reads with their published SHA-256s, and its revision. */
export template<class P>
concept Published = Plan<P> && requires {
	{ std::span<Pinned const>{P::files} };
	{ P::commit } -> std::convertible_to<image::Commit>;
};

export inline constexpr std::array cards = {model::Card::Zero, model::Card::One};

static_assert(cards.size() == model::cards);

/** The catalog of card `card`'s image under plan `P`. */
export template<class P, model::Card card>
using CardCatalog = std::conditional_t<card == model::Card::Zero, typename P::CardZero, typename P::CardOne>;

/** The checkpoint's name for the copy `at` of `tensor`: its scope's prefix, and its layer for a layer's tensor. */
export [[nodiscard]] auto checkpoint_name(model::Tensor const& tensor, schedule::Instance const& at) -> std::string {
	switch (tensor.scope) {
	case model::Scope::Model:
		return std::string{tensor.name};
	case model::Scope::Mtp:
		return std::format("{}{}", model::mtp_prefix, tensor.name);
	case model::Scope::DecoderLayer:
	case model::Scope::LinearAttentionLayer:
	case model::Scope::FullAttentionLayer:
		break;
	}
	// A layer's tensors sit in the trunk's layers or the mtp head's, which the copy's stream tells apart.
	auto const prefix = at.stream == model::Stream::Mtp ? std::string_view{model::mtp_layer_prefix} : std::string_view{model::trunk_layer_prefix};
	return std::format("{}{}.{}", prefix, at.layer, tensor.name);
}

/** The checkpoint's name for an e4m3 tensor's block scales. */
export [[nodiscard]] auto scale_name(std::string_view weight) -> std::string {
	return std::format("{}{}", weight, model::scale_suffix);
}

/** Every placement's bytes are its catalog extent's, in order: the plan describes the images it names. */
template<class P, model::Card card>
[[nodiscard]] consteval auto placements_fill_the_catalog() -> bool {
	auto const placed = P::placements(card);
	auto const& extents = CardCatalog<P, card>::extents;
	if (placed.size() != std::size(extents)) {
		return false;
	}
	for (auto index = std::size_t{0}; index < placed.size(); ++index) {
		if (placed[index].bytes != std::span{extents}[index].bytes || placed[index].read.part != std::span{extents}[index].part) {
			return false;
		}
	}
	return true;
}

template<class P>
[[nodiscard]] consteval auto host_tensors_fill_the_catalog() -> bool {
	auto const tensors = std::span<std::size_t const>{P::host_tensors};
	auto const& extents = P::Host::extents;
	if (tensors.size() != std::size(extents)) {
		return false;
	}
	for (auto index = std::size_t{0}; index < tensors.size(); ++index) {
		auto const& tensor = std::span{P::inventory}[tensors[index]];
		if (tensor.split != model::Split::Host || model::stored_bytes(tensor.format, tensor.shape) != std::span{extents}[index].bytes) {
			return false;
		}
	}
	return true;
}

/** A plan whose placements and host tensors are exactly its catalogs' extents. */
export template<class P>
concept Consistent = Plan<P> && placements_fill_the_catalog<P, model::Card::Zero>() && placements_fill_the_catalog<P, model::Card::One>() &&
                     host_tensors_fill_the_catalog<P>();

}
