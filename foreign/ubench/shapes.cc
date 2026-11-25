// SPDX-License-Identifier: AGPL-3.0-only
//
// The microbenchmarks' shapes, parameters and result layouts re-exported to host code, so a driver launches the
// geometry each kernel was compiled for and reads its results by the layout that wrote them. Device code cannot import
// modules, so they are a device header; the primitives it builds on are `engine.prims`'s and come in through the
// global module fragment, so only the header's own declarations become this module's surface.
module;

#include "prims/budget.h"
#include "prims/selfcheck.h"

export module engine.ubench_device:shapes;

export extern "C++" {
#include "ubench/shapes.h"
}
