#ifndef MOTIONBRICKS_INFERENCE_H
#define MOTIONBRICKS_INFERENCE_H
#include <motionbricks/motionbricks.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Stateless, composed G1 MotionBricks inference. See docs/API-INFERENCE.md.
   No style selection, springs, agent state, seam blending or world placement.
   Use a common canonical Y-up, metre, 30 FPS frame for both boundaries.
   All handles are opaque. Setters copy inputs; getters copy into caller buffers.
   Serialize calls sharing a model (including agent calls), and calls sharing a
   request. Request/model need only outlive a call; output owns its motion data. */
typedef struct mb_inference_request mb_inference_request;
#define MB_INFERENCE_GLOBAL_ROOT UINT32_C(0) /* 8 x 5 floats */
#define MB_INFERENCE_LOCAL_ROOT  UINT32_C(1) /* 8 x 4 floats */
#define MB_INFERENCE_POSE        UINT32_C(2) /* 8 x 303 floats */
#define MB_INFERENCE_DURATIONS   UINT32_C(3) /* mask only: 11 choices, 24..64 frames */

/* New requests require all three feature fields, or both pose boundaries.
   Default masks: all enabled except local-root slot 3; all durations allowed.
   Default sampling: seed 0, Gumbel temperature 1 (argmax disabled). */
MB_API mb_status mb_inference_request_create(mb_inference_request ** output,
    char * error, uint64_t error_capacity);
MB_API void mb_inference_request_free(mb_inference_request * request);
MB_API mb_status mb_inference_request_set_features(mb_inference_request * request,
    uint32_t field, const float * data, uint64_t count, char * error, uint64_t error_capacity);
/* NULL/zero queries size. Unset fields read as zero, but cannot be inferred. */
MB_API mb_status mb_inference_request_get_features(const mb_inference_request * request,
    uint32_t field, float * data, uint64_t capacity, uint64_t * count,
    char * error, uint64_t error_capacity);
/* Masks use uint32_t 0/1, eight entries per feature field, eleven for durations.
   At least one duration must be enabled. Global-root slot zero must stay enabled
   because it supplies the reconstruction origin/heading. */
MB_API mb_status mb_inference_request_set_mask(mb_inference_request * request,
    uint32_t field, const uint32_t * data, uint64_t count, char * error, uint64_t error_capacity);
MB_API mb_status mb_inference_request_get_mask(const mb_inference_request * request,
    uint32_t field, uint32_t * data, uint64_t capacity, uint64_t * count,
    char * error, uint64_t error_capacity);
MB_API mb_status mb_inference_request_set_seed(mb_inference_request * request,
    uint64_t seed, char * error, uint64_t error_capacity);
MB_API mb_status mb_inference_request_get_seed(const mb_inference_request * request,
    uint64_t * seed, char * error, uint64_t error_capacity);
MB_API mb_status mb_inference_request_set_sampling_argmax(mb_inference_request * request,
    uint32_t enabled, char * error, uint64_t error_capacity);
MB_API mb_status mb_inference_request_get_sampling_argmax(const mb_inference_request * request,
    uint32_t * enabled, char * error, uint64_t error_capacity);

/* Optional non-neural adapter: boundary 0=source, 1=target. Exactly four frames:
   roots[4,3], local XYZW[4,34,4]. Caller places/canonicalizes both boundaries.
   Copies encoded features and resets that boundary's masks to enabled, except
   source local slot 3 (unknown outgoing velocity). Target final velocity repeats
   its preceding velocity. Applies upstream virtual-endpoint orientation rules.
   Inputs must be finite, |value|<=1e4; quaternion norms must be within 1 +/- .01.
   Failed setters leave the request unchanged. */
MB_API mb_status mb_inference_request_set_boundary_poses(mb_inference_request * request,
    const mb_model * model, uint32_t boundary, const float * roots, uint64_t root_count,
    const float * local_xyzw, uint64_t rotation_count, char * error, uint64_t error_capacity);

/* Features are raw (not normalized), finite, |value|<=1e4. Returns 24..64 frames
   in multiples of four; duration is argmax among the enabled choices. Output
   includes boundary frames; no playback seam is removed/blended. Target getter
   buffers are empty (no controller-placed target metadata). *output=NULL on
   failure. Does not mutate the request or retain an RNG/history between calls. */
MB_API mb_status mb_model_infer(const mb_model * model, const mb_inference_request * request,
    mb_motion ** output, char * error, uint64_t error_capacity);

/* Plans count independent requests through one shared model call. Requests may
   select different durations; native batching groups equal durations while
   preserving input/output order. Backends may choose independent execution
   when that is faster. `outputs` has count entries and every entry is set to
   NULL on failure. Count must be 1..64. */
MB_API mb_status mb_model_infer_batch(const mb_model * model,
    const mb_inference_request * const * requests, uint64_t count,
    mb_motion ** outputs, char * error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
