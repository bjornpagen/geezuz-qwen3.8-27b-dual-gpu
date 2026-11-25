// SPDX-License-Identifier: AGPL-3.0-only
//
// Attention's geometry, arithmetic and kernel shapes re-exported to host code, so the decode loop, the bring-up
// drivers, the reference and the tests compute with the very definitions the kernels compile. Device code cannot
// import modules, so each definition is a device header. Their dependencies (the model description, the primitives,
// the schedule) are included before the purview and stay those modules' surface; the attention headers themselves are
// included inside it, under one export, which makes their declarations this module's surface without restating a name.
module;

#include "model.h"
#include "prims/budget.h"
#include "prims/fragment.h"
#include "prims/numeric.h"
#include "prims/selfcheck.h"
#include "prims/transcendental.h"
#include "schedule/program.h"

export module engine.attn:spec;

export extern "C++" {
#include "attn/geometry.h"
#include "attn/arithmetic.h"
#include "attn/layout.h"
#include "attn/selfcheck.h"
}
