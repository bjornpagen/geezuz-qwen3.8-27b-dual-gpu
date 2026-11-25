// SPDX-License-Identifier: AGPL-3.0-only
//
// Prints the specimen image's bytes to standard output, for the file test to map: dialect code cannot write a file,
// and the test of the mapped path needs one.
import std;
import engine.image_synthetic;

auto main() -> int {
	auto const baked = engine::image_synthetic::specimen_image();
	auto text = std::string(baked.file.size(), '\0');
	std::ranges::transform(baked.file, text.begin(), [](std::byte byte) { return std::bit_cast<char>(byte); });
	std::print("{}", text);
	return 0;
}
