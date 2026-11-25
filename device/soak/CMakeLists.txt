# SPDX-License-Identifier: AGPL-3.0-only
# The soak's integrity kernels; `engine.soak` embeds the image and the bring-up soak runs it.
engine_device_ptx(soak SOURCE soak.cu)
