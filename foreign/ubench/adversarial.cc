// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ubench_device:adversarial;

import std;

namespace engine::ubench_device {

constexpr char adversarial_bytes[] = {
#embed "ADVERSARIAL.md" suffix(, '\0')
};

/**
 * The adversarial review of the plan, as built: its microbenchmark table names every open hardware question, what
 * each expects and the decision it settles, and the drivers answer that table rather than a copy of it.
 */
export inline constexpr std::string_view adversarial = {adversarial_bytes, sizeof(adversarial_bytes) - 1};

}
