# SPDX-License-Identifier: AGPL-3.0-only
engine_compile_fail(discarded_expected
	SOURCE discarded_expected.cc
	LINK engine_core
	MATCH "ignoring return value of .*nodiscard"
)

engine_compile_fail(unclassified_error
	SOURCE unclassified_error.cc
	LINK engine_core
	MATCH "constraints not satisfied for alias template 'Result'"
)

engine_compile_fail(handle_kind_mismatch
	SOURCE handle_kind_mismatch.cc
	LINK engine_handle
	MATCH "no viable conversion from 'const SlotHandle' to 'BlockHandle'|no matching member function for call to 'try_get'"
)
