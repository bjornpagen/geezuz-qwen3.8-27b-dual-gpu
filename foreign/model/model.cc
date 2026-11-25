// SPDX-License-Identifier: AGPL-3.0-only
//
// The model description re-exported to host code. Device code cannot import modules, so the description is a device
// header; including it inside the purview, under one export, makes every one of its declarations this module's
// surface without restating a single name.
export module engine.model;

export extern "C++" {
#include "model.h"
}
