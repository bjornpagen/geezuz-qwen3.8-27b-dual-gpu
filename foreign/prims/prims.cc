// SPDX-License-Identifier: AGPL-3.0-only
//
// The device primitives that have a meaning on the host, re-exported unchanged so host code (the typed launch
// functions, the baker, the cpu reference and the tests) computes with the very definitions the kernels compile.
// Device code cannot import modules, so each definition is a device header; including it inside the purview, under
// one export, makes its declarations this module's surface without restating a name.
export module engine.prims;

export extern "C++" {
#include "prims/budget.h"
}
