# SPDX-License-Identifier: AGPL-3.0-only
# Attention's kernels: prefill, the standalone decode core and merge, U21's variants and the self-checks.
engine_device_ptx(attn SOURCE attn.cu)
