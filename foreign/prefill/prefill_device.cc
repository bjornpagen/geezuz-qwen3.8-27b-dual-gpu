// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill's kernels as the host sees them: the workspace and parameter blocks they share with it, where each weight of
// a layer lies in the image, and the image with every kernel's typed declaration.
export module engine.prefill_device;

export import :layout;
export import :kernels;
