// SPDX-License-Identifier: AGPL-3.0-only
//
// The gated deltanet as the host sees it: the arithmetic and layouts the kernels compile, the lazy-replay ledger a
// slot keeps between cycles, the fp32 emulation of every kernel in the device's own order, and the kernels' typed
// declarations over the embedded image.
export module engine.deltanet;

export import :arithmetic;
export import :replay;
export import :emulation;
export import :kernels;
