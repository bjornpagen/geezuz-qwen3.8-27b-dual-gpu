// SPDX-License-Identifier: AGPL-3.0-only
export module engine.soak:images;

import std;
import engine.ptx;

namespace engine::soak {

constexpr char soak_text[] = {
#embed "soak.ptx" suffix(, '\0')
};

/** The integrity kernels: the image digest's absorbing and finishing launches, and the pattern's write and check. */
export inline constexpr ptx::Image image = ptx::image(soak_text);

static_assert(image.header == ptx::abi.header, "every image shares the engine's isa and target");

}
