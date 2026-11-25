// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ptx:abi;

import std;
import :image;

namespace engine::ptx {

constexpr char abi_text[] = {
#embed "abi.ptx" suffix(, '\0')
};

/**
 * The device abi image: no code, only the header every engine image shares. Every other image is checked against
 * its header, and loading it is the cheapest proof that the driver can jit the engine's ptx isa.
 */
export inline constexpr Image abi = image(abi_text);

static_assert(abi.header.sm == 86, "the two cards are sm_86");
static_assert(abi.header.address_bits == 64, "device addresses travel as 64-bit integers");

}
