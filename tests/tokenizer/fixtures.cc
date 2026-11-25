// SPDX-License-Identifier: AGPL-3.0-only
export module engine_test_tokenizer.fixtures;

import std;
import engine.core;
import engine.tokenizer;
import engine.tokenizer_fixtures;

namespace engine_test_tokenizer {

using engine::tokenizer::Id;
using engine::tokenizer::Spelling;

/** A text and the ids the reference encodes it to, reading added-token spellings as `spelling` says. */
export struct EncodeCase {
	std::string text;
	Spelling spelling;
	std::vector<Id> ids;
};

/** Ids and the text the reference decodes them to. */
export struct DecodeCase {
	std::vector<Id> ids;
	std::string text;
};

/** Reads little-endian words and padded byte strings from a fixture file, which the test trusts. */
class Cursor {
public:
	explicit Cursor(std::span<std::byte const> bytes) : bytes_{bytes} {}

	[[nodiscard]] auto word() -> std::uint32_t {
		auto value = std::uint32_t{0};
		for (auto shift = 0U; shift < 32U; shift += 8U) {
			value |= std::to_integer<std::uint32_t>(bytes_[at_]) << shift;
			++at_;
		}
		return value;
	}

	[[nodiscard]] auto words(std::size_t count) -> std::vector<Id> {
		auto out = std::vector<Id>{};
		for (auto index = std::size_t{0}; index < count; ++index) {
			out.push_back(word());
		}
		return out;
	}

	[[nodiscard]] auto text(std::size_t count) -> std::string {
		auto out = std::string{};
		for (auto const byte : bytes_.subspan(at_, count)) {
			out.push_back(std::bit_cast<char>(byte));
		}
		at_ += (count + 3) / 4 * 4;
		return out;
	}

	[[nodiscard]] auto finished() const -> bool {
		return at_ == bytes_.size();
	}

private:
	std::span<std::byte const> bytes_;
	std::size_t at_ = 0;
};

/** The fixture's cases, or nothing when its header or length is wrong. */
export [[nodiscard]] auto encode_cases() -> std::vector<EncodeCase> {
	auto cursor = Cursor{engine::tokenizer_fixtures::encode()};
	if (cursor.text(8) != "gzqwnenc" || cursor.word() != 1) {
		return {};
	}
	auto const count = cursor.word();
	auto out = std::vector<EncodeCase>{};
	for (auto index = std::uint32_t{0}; index < count; ++index) {
		auto const spelling = cursor.word() == 0 ? Spelling::Characters : Spelling::Token;
		auto const bytes = cursor.word();
		auto const ids = cursor.word();
		auto text = cursor.text(bytes);
		out.push_back(EncodeCase{.text = std::move(text), .spelling = spelling, .ids = cursor.words(ids)});
	}
	return cursor.finished() ? out : std::vector<EncodeCase>{};
}

export [[nodiscard]] auto decode_cases() -> std::vector<DecodeCase> {
	auto cursor = Cursor{engine::tokenizer_fixtures::decode()};
	if (cursor.text(8) != "gzqwndec" || cursor.word() != 1) {
		return {};
	}
	auto const count = cursor.word();
	auto out = std::vector<DecodeCase>{};
	for (auto index = std::uint32_t{0}; index < count; ++index) {
		auto const ids = cursor.word();
		auto const bytes = cursor.word();
		auto decoded = cursor.words(ids);
		out.push_back(DecodeCase{.ids = std::move(decoded), .text = cursor.text(bytes)});
	}
	return cursor.finished() ? out : std::vector<DecodeCase>{};
}

}
