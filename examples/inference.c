/* Minimal inference-only integration. Input arrays contain four canonical
   source frames followed by four already-placed target frames. No controller,
   style assets, physics or internal C++ headers are used. */
#include <motionbricks/inference.h>
#include <stddef.h>

mb_status infer_between(const mb_model * model, const float * roots_8x3,
                        const float * local_xyzw_8x34x4, uint64_t seed,
                        mb_motion ** output, char * error, uint64_t error_capacity) {
    mb_inference_request * request = NULL;
    mb_status status;
    /* Reuse the API's diagnostics for invalid pointers. */
    if (!output || !roots_8x3 || !local_xyzw_8x34x4)
        return mb_model_infer(model, NULL, output, error, error_capacity);
    *output = NULL;
    status = mb_inference_request_create(&request, error, error_capacity);
    if (status != MB_OK) return status;
    for (uint32_t boundary = 0; boundary < 2; ++boundary) {
        status = mb_inference_request_set_boundary_poses(request, model, boundary,
            roots_8x3 + boundary * 12, 12,
            local_xyzw_8x34x4 + boundary * 544, 544, error, error_capacity);
        if (status != MB_OK) goto done;
    }
    /* 40 frames = duration index 4 (24 + 4*4). */
    {
        const uint32_t durations[11] = {0,0,0,0,1,0,0,0,0,0,0};
        status = mb_inference_request_set_mask(request, MB_INFERENCE_DURATIONS,
            durations, 11, error, error_capacity);
    }
    if (status != MB_OK) goto done;
    status = mb_inference_request_set_seed(request, seed, error, error_capacity);
    if (status == MB_OK)
        status = mb_model_infer(model, request, output, error, error_capacity);
done:
    mb_inference_request_free(request);
    /* On success *output is owned by the caller; free with mb_motion_free. */
    return status;
}
