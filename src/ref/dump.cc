// SPDX-License-Identifier: AGPL-3.0-only
//
// The teacher-forced logit dump: per row its best candidates and enough to measure against another dump, in a compact
// binary file that ends in the SHA-256 of everything before it. All numbers are little-endian:
//
//   magic        8 bytes  "GZTOPK01"
//   source       u32      `Source`
//   prompt       u32      tokens the run was given; the rest it chose
//   rows         u32      one per token
//   candidates   u32      kept per row
//   vocabulary   u32      the sampled vocabulary the logits cover
//   tokens       rows x u32; row r's logits are computed after token r
//   next         (rows - 1) x f32: row r's logit of token r + 1
//   per row      f64 log-sum-exp of all its logits; then `candidates` x (u32 id, f32 logit) in rank order
//   digest       32 bytes: SHA-256 of every byte above
export module engine.ref:dump;

import std;
import engine.core;
import engine.model;
import engine.sampling;
import engine.sampling_device;
import engine.sha256;
import :forward;

namespace engine::ref {

namespace model = engine::model;

using model::Count;

/** What computed a dump's logits. */
export enum class Source : std::uint8_t {
	/** The reference under the sm_86 tensor core's accumulation: the engine's arithmetic. */
	Tensor,
	/** The reference under ieee fp32 accumulation: the noise floor's other side. */
	Ieee,
	/** The engine on the cards. */
	Engine,
};

/** One ranked candidate: a token and its logit. */
export struct Ranked {
	Token id;
	float logit;

	[[nodiscard]] auto operator==(Ranked const&) const -> bool = default;
};

export struct DumpRow {
	double logsumexp;
	std::vector<Ranked> best;

	[[nodiscard]] auto operator==(DumpRow const&) const -> bool = default;
};

export struct Dump {
	Source source;
	Count prompt;
	std::vector<Token> tokens;
	/** Row r's logit of token r + 1, for every row but the last. */
	std::vector<float> next;
	std::vector<DumpRow> rows;

	[[nodiscard]] auto operator==(Dump const&) const -> bool = default;
};

export inline constexpr Count dump_candidates = static_cast<Count>(sampling_device::top_k_bound);
inline constexpr auto magic = std::string_view{"GZTOPK01"};
inline constexpr Count header_bytes = 8 + (5 * 4);

/** One row's best candidates (exactly as the cards' merge ranks them) and the log-sum-exp of all its logits. */
export [[nodiscard]] auto summarize(std::span<float const> logits) -> DumpRow {
	auto const most = std::ranges::max(logits);
	auto total = 0.0;
	for (auto const logit : logits) {
		total += std::exp(static_cast<double>(logit) - static_cast<double>(most));
	}
	auto best = std::vector<Ranked>{};
	for (auto const& candidate : sampling::candidates(logits)) {
		best.push_back({.id = core::narrow<Token>(candidate.id), .logit = candidate.logit});
	}
	return {.logsumexp = static_cast<double>(most) + std::log(total), .best = std::move(best)};
}

/** The rows of a run over `tokens` from their logits; `prompt` of them were given. */
export [[nodiscard]] auto dump_of(Source source, Count prompt, std::span<Token const> tokens, std::span<std::vector<float> const> logits) -> Dump {
	core::invariant(tokens.size() == logits.size() && prompt <= tokens.size());
	auto dump = Dump{.source = source, .prompt = prompt, .tokens = {tokens.begin(), tokens.end()}, .next = {}, .rows = {}};
	for (auto const [row, values] : std::views::enumerate(logits)) {
		auto const index = static_cast<Count>(row);
		if (index + 1 < tokens.size()) {
			dump.next.push_back(values[tokens[index + 1]]);
		}
		dump.rows.push_back(summarize(values));
	}
	return dump;
}

struct Writer {
	std::vector<std::byte> bytes;

	auto u32(std::uint32_t value) -> void {
		for (auto shift = 0U; shift < 32U; shift += 8U) {
			bytes.push_back(std::byte{static_cast<std::uint8_t>(value >> shift)});
		}
	}

	auto u64(std::uint64_t value) -> void {
		u32(static_cast<std::uint32_t>(value));
		u32(static_cast<std::uint32_t>(value >> 32U));
	}
};

export [[nodiscard]] auto encode(Dump const& dump) -> std::vector<std::byte> {
	core::invariant(dump.next.size() + 1 == dump.tokens.size() && dump.rows.size() == dump.tokens.size());
	auto out = Writer{};
	for (auto const letter : magic) {
		out.bytes.push_back(std::byte{static_cast<std::uint8_t>(letter)});
	}
	out.u32(std::to_underlying(dump.source));
	out.u32(core::narrow<std::uint32_t>(dump.prompt));
	out.u32(core::narrow<std::uint32_t>(dump.tokens.size()));
	out.u32(core::narrow<std::uint32_t>(dump_candidates));
	out.u32(core::narrow<std::uint32_t>(model::sampled_vocabulary));
	for (auto const token : dump.tokens) {
		out.u32(token);
	}
	for (auto const logit : dump.next) {
		out.u32(std::bit_cast<std::uint32_t>(logit));
	}
	for (auto const& row : dump.rows) {
		core::invariant(row.best.size() == dump_candidates);
		out.u64(std::bit_cast<std::uint64_t>(row.logsumexp));
		for (auto const& candidate : row.best) {
			out.u32(candidate.id);
			out.u32(std::bit_cast<std::uint32_t>(candidate.logit));
		}
	}
	auto const digest = sha256::digest(std::span<std::byte const>{out.bytes});
	out.bytes.insert(out.bytes.end(), digest.bytes.begin(), digest.bytes.end());
	return std::move(out.bytes);
}

export enum class DumpFailure : std::uint8_t {
	/** Fewer bytes than the header announces, or more. */
	Truncated,
	/** Not a dump: the magic is wrong. */
	Foreign,
	/** The trailing SHA-256 is not the digest of the bytes before it. */
	Corrupt,
	/** A source, candidate count or vocabulary this reader does not hold. */
	Unknown,
	/** An id outside the sampled vocabulary, candidates out of rank order, or a prompt longer than the tokens. */
	Inconsistent,
};

export struct [[nodiscard]] DumpError {
	DumpFailure failure;

	[[nodiscard]] constexpr auto retry() const -> core::Retry {
		switch (failure) {
		case DumpFailure::Truncated:
		case DumpFailure::Foreign:
		case DumpFailure::Corrupt:
		case DumpFailure::Unknown:
		case DumpFailure::Inconsistent:
			return core::Retry::Permanent;
		}
		std::unreachable();
	}

	[[nodiscard]] constexpr auto operator==(DumpError const&) const -> bool = default;
};

struct Reader {
	std::span<std::byte const> bytes;
	Count at;

	[[nodiscard]] auto u32() -> std::uint32_t {
		auto value = std::uint32_t{0};
		for (auto index = 0U; index < 4U; ++index) {
			value |= std::to_integer<std::uint32_t>(bytes[at + index]) << (8U * index);
		}
		at += 4;
		return value;
	}

	[[nodiscard]] auto u64() -> std::uint64_t {
		auto const low = std::uint64_t{u32()};
		return low | (std::uint64_t{u32()} << 32U);
	}
};

export [[nodiscard]] auto decode(std::span<std::byte const> bytes) -> core::Result<Dump, DumpError> {
	auto const fail = [](DumpFailure failure) {
		return std::unexpected(DumpError{.failure = failure});
	};
	if (bytes.size() < header_bytes + sha256::digest_bytes) {
		return fail(DumpFailure::Truncated);
	}
	if (!std::ranges::equal(bytes.first(magic.size()), magic, [](std::byte byte, char letter) {
		    return std::to_integer<char>(byte) == letter;
	    })) {
		return fail(DumpFailure::Foreign);
	}
	auto in = Reader{.bytes = bytes, .at = magic.size()};
	auto const source = in.u32();
	auto const prompt = Count{in.u32()};
	auto const rows = Count{in.u32()};
	auto const candidates = Count{in.u32()};
	auto const vocabulary = Count{in.u32()};
	if (source > std::to_underlying(Source::Engine) || candidates != dump_candidates || vocabulary != model::sampled_vocabulary) {
		return fail(DumpFailure::Unknown);
	}
	auto const row_bytes = 8 + (candidates * 8);
	if (rows == 0 || bytes.size() != header_bytes + (rows * 4) + ((rows - 1) * 4) + (rows * row_bytes) + sha256::digest_bytes) {
		return fail(DumpFailure::Truncated);
	}
	auto const body = bytes.first(bytes.size() - sha256::digest_bytes);
	auto const digest = sha256::digest(body);
	if (!std::ranges::equal(digest.bytes, bytes.last(sha256::digest_bytes))) {
		return fail(DumpFailure::Corrupt);
	}
	auto dump = Dump{.source = static_cast<Source>(source), .prompt = prompt, .tokens = {}, .next = {}, .rows = {}};
	for (auto row = Count{0}; row < rows; ++row) {
		dump.tokens.push_back(in.u32());
	}
	for (auto row = Count{0}; row + 1 < rows; ++row) {
		dump.next.push_back(std::bit_cast<float>(in.u32()));
	}
	for (auto row = Count{0}; row < rows; ++row) {
		auto each = DumpRow{.logsumexp = std::bit_cast<double>(in.u64()), .best = {}};
		for (auto rank = Count{0}; rank < candidates; ++rank) {
			auto const id = in.u32();
			each.best.push_back({.id = id, .logit = std::bit_cast<float>(in.u32())});
		}
		dump.rows.push_back(std::move(each));
	}
	auto const ranked = std::ranges::all_of(dump.rows, [](DumpRow const& row) {
		auto const in_vocabulary = std::ranges::all_of(row.best, [](Ranked const& each) {
			return each.id < model::sampled_vocabulary;
		});
		return in_vocabulary && std::ranges::is_sorted(row.best, [](Ranked const& a, Ranked const& b) {
			       return sampling_device::ranks_before({.logit = a.logit, .id = static_cast<int>(a.id)},
			                                                      {.logit = b.logit, .id = static_cast<int>(b.id)});
		       });
	});
	auto const tokens_known = std::ranges::all_of(dump.tokens, [](Token token) {
		return token < model::sampled_vocabulary;
	});
	if (!ranked || !tokens_known || prompt == 0 || prompt > rows) {
		return fail(DumpFailure::Inconsistent);
	}
	return dump;
}

}
