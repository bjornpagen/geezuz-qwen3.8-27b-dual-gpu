// SPDX-License-Identifier: AGPL-3.0-only
//
// Processes at their boundary: a child process the caller owns, spawned from a path and arguments, its standard output
// read as bounded lines with a deadline, asked to stop with SIGTERM, and reaped with a deadline (dropping it kills and
// reaps whatever is left); and this process's own standard output, flushed as it is written.
export module engine.process;

export import :error;
export import :child;
export import :output;
