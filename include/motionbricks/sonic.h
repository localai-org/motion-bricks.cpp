#ifndef MOTIONBRICKS_SONIC_H
#define MOTIONBRICKS_SONIC_H
#include <motionbricks/motionbricks.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mb_sonic mb_sonic;
#define MB_SONIC_MAX_BATCH UINT32_C(64)
/* Original-release SONIC G1 mode 0 and optional SMPL mode 2, F32. Options are copied.
   Inference only: no physics, observation history, timing or motor control.
   encode: 1762 observation floats -> 64 FSQ token floats.
   decode: 994 observation floats (tokens first) -> 29 unscaled actions.
   Observation layouts/order: docs/API-SONIC-PHYSICS.md.
   Calls on a handle are serialized; free requires no calls in flight.
   Buffers belong to the caller; they are never retained. Counts are floats.
   obs[0] selects 0 (G1) or 2 (SMPL). Mode 2 requires a GGUF converted with
   --include-smpl. Legacy G1-only files reject mode 2; modes 1/3 are unsupported.
   SMPL uses [922,1642) joints, [1642,1702) anchor orientations, and
   [1702,1762) wrists, each frame-major across ten samples. See the layout doc.
   Observations must be finite with absolute value <= 1e6; larger values are
   rejected (not clamped) before arithmetic to prevent overflow in the backend. */
MB_API mb_status mb_sonic_load(const char * gguf_file, const mb_runtime_options * options,
    mb_sonic ** output, char * error, uint64_t error_capacity);
MB_API void mb_sonic_free(mb_sonic * model);
MB_API mb_status mb_sonic_encode(mb_sonic * model, const float * observations,
    uint64_t count, float * tokens, uint64_t token_count, char * error, uint64_t error_capacity);
MB_API mb_status mb_sonic_decode(mb_sonic * model, const float * observations,
    uint64_t count, float * actions, uint64_t action_count, char * error, uint64_t error_capacity);
/* Caller-controlled batching with no internal queue or waiting window. Buffers
   contain batch_size tightly packed requests in request-major order. Counts are
   total float counts: batch_size*1762 and *64 for encode, *994 and *29 for
   decode. All requests in an encoder batch must select the same mode.
   Batch sizes 1..MB_SONIC_MAX_BATCH are supported. Graph scratch is
   cached by size; weights remain single-copy. Calls on a handle are serialized. */
MB_API mb_status mb_sonic_encode_batch(mb_sonic * model, const float * observations,
    uint64_t count, float * tokens, uint64_t token_count, uint32_t batch_size,
    char * error, uint64_t error_capacity);
MB_API mb_status mb_sonic_decode_batch(mb_sonic * model, const float * observations,
    uint64_t count, float * actions, uint64_t action_count, uint32_t batch_size,
    char * error, uint64_t error_capacity);
/* Diagnostic layer: 0..4 encoder preactivations; 0..6 decoder preactivations.
   Encoder traces refer to the last successful encoder call, in either mode.
   Calling encode/decode updates these traces. A batch trace is request-major
   and its count includes every request. Returns the exact required count; NULL
   data / capacity 0 is a size query. No borrowed C++ layouts. */
MB_API mb_status mb_sonic_layer(mb_sonic * model, uint32_t encoder, uint32_t layer,
    float * data, uint64_t capacity, uint64_t * count, char * error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
