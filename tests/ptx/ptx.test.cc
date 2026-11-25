// SPDX-License-Identifier: AGPL-3.0-only
import std;
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
	};
	return engine::testing::report(cases);
}
