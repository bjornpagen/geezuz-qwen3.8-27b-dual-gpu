// SPDX-License-Identifier: AGPL-3.0-only
export module engine.ubench_device:images;

import std;
import engine.ptx;

namespace engine::ubench_device {

constexpr char stream_text[] = {
#embed "ubench_stream.ptx" suffix(, '\0')
};

constexpr char sync_text[] = {
#embed "ubench_sync.ptx" suffix(, '\0')
};

constexpr char compute_text[] = {
#embed "ubench_compute.ptx" suffix(, '\0')
};

constexpr char link_text[] = {
#embed "ubench_link.ptx" suffix(, '\0')
};

constexpr char prims_check_text[] = {
#embed "prims_check.ptx" suffix(, '\0')
};

/** The weight-stream kernels. */
export inline constexpr ptx::Image stream_image = ptx::image(stream_text);
/** The synchronisation kernels. */
export inline constexpr ptx::Image sync_image = ptx::image(sync_text);
/** The arithmetic kernels. */
export inline constexpr ptx::Image compute_image = ptx::image(compute_text);
/** The cross-card and host kernels. */
export inline constexpr ptx::Image link_image = ptx::image(link_text);
/** The primitives' self-check kernels, which the prims component builds and the bring-up drivers run. */
export inline constexpr ptx::Image prims_check_image = ptx::image(prims_check_text);

static_assert(stream_image.header == ptx::abi.header && sync_image.header == ptx::abi.header, "every image shares the engine's isa and target");
static_assert(compute_image.header == ptx::abi.header && link_image.header == ptx::abi.header, "every image shares the engine's isa and target");
static_assert(prims_check_image.header == ptx::abi.header, "every image shares the engine's isa and target");

}
