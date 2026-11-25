// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.file;
import engine.model;
import engine.testing;

namespace {

namespace model = engine::model;

/** The draft vocabulary file: little-endian 32-bit token ids, nothing else. */
[[nodiscard]] auto decode_ids(std::span<std::byte const> bytes) -> std::vector<std::uint32_t> {
	auto ids = std::vector<std::uint32_t>{};
	for (auto offset = std::size_t{0}; offset + 4 <= bytes.size(); offset += 4) {
		auto id = std::uint32_t{0};
		for (auto index = std::size_t{0}; index < 4; ++index) {
			id |= std::to_integer<std::uint32_t>(bytes[offset + index]) << (std::size_t{8} * index);
		}
		ids.push_back(id);
	}
	return ids;
}

}

auto main() -> int {
	auto const mapping = engine::file::Mapping::open("draft_vocab_32k.bin");
	if (!mapping) {
		std::println("FAIL: data/draft_vocab_32k.bin maps");
		return 1;
	}
	auto const bytes = mapping->bytes();
	auto const ids = decode_ids(bytes);
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the draft vocabulary file holds exactly draft_vocab whole 32-bit ids",
	        .holds = bytes.size() % 4 == 0 && ids.size() == model::draft_vocab,
	    },
	    engine::testing::Case{
	        .name = "draft ids are strictly ascending and inside the vocabulary",
	        .holds =
	            std::ranges::adjacent_find(ids, std::ranges::greater_equal{}) == ids.end() && !ids.empty() && ids.back() < model::vocab,
	    },
	};
	return engine::testing::report(cases);
}
