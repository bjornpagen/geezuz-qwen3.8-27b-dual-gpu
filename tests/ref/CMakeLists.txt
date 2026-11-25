# SPDX-License-Identifier: AGPL-3.0-only
engine_test(ref.sums SOURCES sums.test.cc LINK engine_ref)
engine_test(ref.deltanet SOURCES deltanet.test.cc LINK engine_ref)
engine_test(ref.attention SOURCES attention.test.cc LINK engine_ref)
engine_test(ref.project SOURCES project.test.cc LINK engine_ref engine_schedule)
engine_test(ref.rulers SOURCES rulers.test.cc LINK engine_ref)
engine_test(ref.golden SOURCES golden.test.cc LINK engine_ref engine_sha256)
engine_test(ref.thread_pool SOURCES thread_pool.test.cc LINK engine_thread_pool)
engine_checkpoint_test(ref.checkpoint SOURCES checkpoint.test.cc LINK engine_ref engine_tokenizer)
