// SPDX-License-Identifier: AGPL-3.0-only
//
// The units memory moves in, as plain layout types the host and the device share: buffers the host fills and kernels
// copy are arrays of them.
#pragma once

namespace engine::device::prims {

/** Sixteen bytes: the unit of every vector load, store and async copy. */
struct alignas(16) Chunk {
	unsigned word[4];
};

/** A 128-byte l2 line: the unit `discard` and `applypriority` act on. */
struct alignas(128) Line {
	unsigned char bytes[128];
};

static_assert(sizeof(Chunk) == 16 && sizeof(Line) == 128);

}
