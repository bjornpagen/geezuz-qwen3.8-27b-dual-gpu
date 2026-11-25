// SPDX-License-Identifier: AGPL-3.0-only
import std;
import engine.sampling;

// A NaN logit is a bug upstream of sampling, never a value to sample around: the reference refuses it by aborting.
auto main() -> int {
	auto const logits = std::array{1.0F, std::numeric_limits<float>::quiet_NaN(), 0.5F};
	return static_cast<int>(engine::sampling::candidates(logits).size());
}
