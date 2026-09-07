#ifndef MOTIONBRICKS_SONIC_H
#define MOTIONBRICKS_SONIC_H
#include <motionbricks/motionbricks.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mb_sonic mb_sonic;
/* Original-release SONIC G1 mode 0, batch one, F32. Options are copied.
   Inference only: no physics, observation history, timing or motor control.
   encode: 1762 observation floats -> 64 FSQ token floats.
   decode: 994 observation floats (tokens first) -> 29 unscaled actions.
   Observation layouts/order: docs/API-SONIC-PHYSICS.md.
   Calls on a handle are serialized; free requires no calls in flight.
   Buffers belong to the caller; they are never retained. Counts are floats.
   Other encoder modes are rejected, never silently treated as G1.
   Observations must be finite with absolute value <= 1e6; larger values are
   rejected (not clamped) before arithmetic to prevent overflow in the backend. */
MB_API mb_status mb_sonic_load(const char * gguf_file, const mb_runtime_options * options,
    mb_sonic ** output, char * error, uint64_t error_capacity);
MB_API void mb_sonic_free(mb_sonic * model);
MB_API mb_status mb_sonic_encode(mb_sonic * model, const float * observations,
    uint64_t count, float * tokens, uint64_t token_count, char * error, uint64_t error_capacity);
MB_API mb_status mb_sonic_decode(mb_sonic * model, const float * observations,
    uint64_t count, float * actions, uint64_t action_count, char * error, uint64_t error_capacity);
/* Diagnostic layer: 0..4 encoder preactivations; 0..6 decoder preactivations.
   Calling encode/decode updates these traces. Returns the exact required count;
   NULL data / capacity 0 is a size query. No borrowed C++ layouts. */
MB_API mb_status mb_sonic_layer(mb_sonic * model, uint32_t encoder, uint32_t layer,
    float * data, uint64_t capacity, uint64_t * count, char * error, uint64_t error_capacity);
#ifdef __cplusplus
}
#endif
#endif
