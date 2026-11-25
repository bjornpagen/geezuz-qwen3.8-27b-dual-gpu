// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.prims;
import engine.ptx;
import engine.testing;

namespace {

using engine::ptx::Header;
using engine::ptx::parse_header;

[[nodiscard]] auto abi_image_is_one_terminated_string() -> bool {
	auto const text = engine::ptx::abi.text;
	return !text.empty() && text.back() == '\0' && std::ranges::count(text, '\0') == 1;
}

[[nodiscard]] auto abi_header_matches_its_text() -> bool {
	auto const body = std::string_view{engine::ptx::abi.text.data(), engine::ptx::abi.text.size() - 1};
	return parse_header(body) == engine::ptx::abi.header;
}

[[nodiscard]] auto header_reads_every_directive() -> bool {
	return parse_header("//\n.version 8.7\n.target sm_90\n.address_size 32\n") == Header{.isa = 87, .sm = 90, .address_bits = 32};
}

[[nodiscard]] auto header_refuses_missing_or_malformed_directives() -> bool {
	auto const refused = std::array<std::string_view, 6>{
	    ".target sm_86\n.address_size 64\n",
	    ".version 9.3\n.address_size 64\n",
	    ".version 9.3\n.target sm_86\n",
	    ".version 93\n.target sm_86\n.address_size 64\n",
	    ".version 9.x\n.target sm_86\n.address_size 64\n",
	    ".version 9.10\n.target sm_86\n.address_size 64\n",
	};
	return std::ranges::none_of(refused, [](std::string_view text) {
		return parse_header(text).has_value();
	});
}

// Budget records as the device compiler emits them: the longer name first, so a reader that matched a name's prefix
// would take its record for the shorter one, and one record with its trailing zero bytes omitted.
constexpr auto records_text =
    std::to_array(".version 9.0\n.target sm_86\n.address_size 64\n\n"
	              ".visible .const .align 4 .b8 engine_budget_probe_wide[16] = {64, 0, 0, 0, 40, 0, 0, 0, 0, 16, 0, 0, 2};\n"
	              ".visible .const .align 4 .b8 engine_budget_probe[16] = {32, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 1};\n"
	              ".visible .const .align 4 .b8 engine_budget_probe_short[16] = {128, 0, 0, 0, 32};\n\n"
	              ".visible .entry probe()\n{\n\tret;\n}\n");

constexpr auto records = engine::ptx::image(std::span<char const>{records_text});

[[nodiscard]] constexpr auto same(engine::device::prims::Budget left, engine::device::prims::Budget right) -> bool {
	return left.threads == right.threads && left.registers_per_thread == right.registers_per_thread &&
	       left.shared_bytes == right.shared_bytes && left.blocks_per_sm == right.blocks_per_sm;
}

/** A record is read for exactly the entry named, never for one whose name it merely begins. */
[[nodiscard]] auto records_match_whole_names() -> bool {
	constexpr auto probe = engine::ptx::recorded_budget(records, "probe");
	constexpr auto wide = engine::ptx::recorded_budget(records, "probe_wide");
	return same(probe, {.threads = 32, .registers_per_thread = 255, .shared_bytes = 0, .blocks_per_sm = 1}) &&
	       same(wide, {.threads = 64, .registers_per_thread = 40, .shared_bytes = 4096, .blocks_per_sm = 2});
}

/** Each field is four little-endian bytes, and the bytes the compiler omits are zero. */
[[nodiscard]] auto records_read_little_endian_fields() -> bool {
	constexpr auto short_record = engine::ptx::recorded_budget(records, "probe_short");
	return same(short_record, {.threads = 128, .registers_per_thread = 32, .shared_bytes = 0, .blocks_per_sm = 0});
}

}

auto main() -> int {
	auto const cases = std::array{
	    engine::testing::Case{
	        .name = "the embedded abi image is exactly one NUL-terminated string",
	        .holds = abi_image_is_one_terminated_string(),
	    },
	    engine::testing::Case{
	        .name = "the abi image's header is what its embedded text states",
	        .holds = abi_header_matches_its_text(),
	    },
	    engine::testing::Case{
	        .name = "a header reads the isa, target and address width directives",
	        .holds = header_reads_every_directive(),
	    },
	    engine::testing::Case{
	        .name = "a header with a missing or malformed directive is refused",
	        .holds = header_refuses_missing_or_malformed_directives(),
	    },
	    engine::testing::Case{
	        .name = "a budget record is found by its whole entry name",
	        .holds = records_match_whole_names(),
	    },
	    engine::testing::Case{
	        .name = "a budget record's fields are little-endian words, omitted bytes zero",
	        .holds = records_read_little_endian_fields(),
	    },
	};
	return engine::testing::report(cases);
}
