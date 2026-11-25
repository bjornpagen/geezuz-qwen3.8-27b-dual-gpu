// SPDX-License-Identifier: AGPL-3.0-only
//
// The bake's round trip on the real checkpoint: bakes it into a scratch directory, opens the three images against the
// identity the engine expects, inverts them back into every tensor the model reads and compares each with the
// checkpoint bit for bit, then deletes the images. Run it in the release build; it hashes the checkpoint twice and
// writes and reads back about 30 GB.
import std;
import engine.bake;
import engine.bake_inverse;
import engine.image;
import engine.testing;

namespace {

namespace bake = engine::bake;
namespace image = engine::image;
namespace inverse = engine::bake_inverse;

using Clock = std::chrono::steady_clock;

[[nodiscard]] auto seconds(Clock::duration elapsed) -> double {
	return std::chrono::duration<double>(elapsed).count();
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 2) {
		std::println("FAIL: usage: <checkpoint directory>");
		return 1;
	}
	auto const checkpoint = std::filesystem::path{arguments[1]};
	auto const scratch = std::filesystem::current_path() / "bake.checkpoint.scratch";
	std::filesystem::remove_all(scratch);
	std::filesystem::create_directories(scratch);

	auto const start = Clock::now();
	auto const report = bake::bake<bake::Qwen>(checkpoint, scratch);
	auto const baked = Clock::now();
	if (!report) {
		std::println("FAIL: the checkpoint bakes: {}", report.error().describe());
		return 1;
	}
	auto verified = bake::verify(checkpoint, bake::Qwen::files, bake::Qwen::commit);
	auto source = verified ? bake::Source::open<bake::Qwen>(*std::move(verified)) : std::unexpected(verified.error());
	auto const zero = image::Image<bake::Qwen::CardZero>::open(scratch / bake::image_name(image::Target::CardZero), bake::checkpoint);
	auto const one = image::Image<bake::Qwen::CardOne>::open(scratch / bake::image_name(image::Target::CardOne), bake::checkpoint);
	auto const host = image::Image<bake::Qwen::Host>::open(scratch / bake::image_name(image::Target::Host), bake::checkpoint);
	auto const opened = Clock::now();
	if (!source || !zero || !one || !host) {
		std::println("FAIL: the checkpoint reads back and every image opens against the pinned identity");
		std::filesystem::remove_all(scratch);
		return 1;
	}
	auto const verdict = inverse::invert<bake::Qwen>(
	    *source, {.cards = {inverse::extents_of(*zero), inverse::extents_of(*one)}, .host = inverse::extents_of(*host)});
	auto const inverted = Clock::now();
	std::filesystem::remove_all(scratch);
	for (auto const& name : verdict.mismatched) {
		std::println("mismatched: {}", name);
	}
	std::println("bake {:.1f} s (bound {:e} from {}), reopen and hash {:.1f} s, inversion of {} tensor copies {:.1f} s, total {:.1f} s",
	             seconds(baked - start), report->bound.value, report->bound.tensor, seconds(opened - baked), verdict.copies,
	             seconds(inverted - opened), seconds(inverted - start));
	auto const cases = std::array{
	    engine::testing::Case{.name = "the baked images carry the identity the engine expects of the pinned checkpoint",
	                          .holds = report->checkpoint == bake::checkpoint},
	    engine::testing::Case{.name = "every tensor copy rebuilt from the card and host images is the checkpoint's, bit for bit",
	                          .holds = verdict.mismatched.empty() && verdict.matched == verdict.copies && verdict.copies > 0},
	    engine::testing::Case{.name = "the attention bound is below 2^70", .holds = report->bound.value < bake::overflow_limit},
	};
	return engine::testing::report(cases);
}
