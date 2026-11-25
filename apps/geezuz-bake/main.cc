// SPDX-License-Identifier: AGPL-3.0-only
//
// geezuz-bake <checkpoint-dir> <output-dir>: converts the Qwen3.8-27B-FP8 checkpoint into the engine's two card images
// and its host image. It takes no options: the model, the split and the layout are the engine's own.
import std;
import engine.bake;
import engine.image;
import engine.sha256;

namespace {

namespace bake = engine::bake;

auto print_report(bake::Report const& report) -> void {
	std::println("verified {} checkpoint files, {} bytes, against their published SHA-256s", report.files, report.checkpoint_bytes);
	auto const commit = engine::sha256::hex(report.checkpoint.commit.bytes);
	std::println("checkpoint commit {}, manifest {}", std::string_view{commit.data(), commit.size()}, report.checkpoint.manifest);
	std::println("attention bound: |v| <= {:e} ({}), below 2^70 = {:e}", report.bound.value, report.bound.tensor, bake::overflow_limit);
	for (auto const& image : report.images) {
		std::println("{}: {} bytes, payload sha256 {}", image.path.string(), image.bytes, image.payload);
	}
}

}

auto main(int argc, char** argv) -> int {
	auto const arguments = std::span{argv, static_cast<std::size_t>(argc)};
	if (arguments.size() != 3) {
		std::println("usage: geezuz-bake <checkpoint-dir> <output-dir>");
		return 2;
	}
	auto const report = bake::bake<bake::Qwen>(std::filesystem::path{arguments[1]}, std::filesystem::path{arguments[2]});
	if (!report) {
		std::println("refused: {}", report.error().describe());
		return 1;
	}
	print_report(*report);
	return 0;
}
