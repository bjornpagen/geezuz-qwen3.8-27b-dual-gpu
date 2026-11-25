// SPDX-License-Identifier: AGPL-3.0-only
//
// The ring's slot compiled by the device compiler itself: the persistent kernel reads the same definition the host
// does, and this unit proves it is valid device code with the layout laws holding on the device side too. The image
// holds no code.
#include "ring/ring.h"

namespace {

struct Words {
	unsigned long long word[15];
};

struct Byte {
	unsigned char value;
};

}

using engine::device::ring::Layout;
using engine::device::ring::line;

static_assert(Layout<Words, Byte>::response == 2 * line, "a descriptor past one line pushes the response half down");
static_assert(Layout<Words, Byte>::bytes == 3 * line);
static_assert(Layout<Byte, Words>::response == line && Layout<Byte, Words>::bytes == 2 * line);
static_assert(sizeof(engine::device::ring::Command) == 1);
