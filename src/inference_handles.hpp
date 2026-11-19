#pragma once
#include "handles.hpp"
#include "planner.hpp"
struct mb_inference_request {
    motionbricks::detail::transition_constraints constraints;
    std::array<std::array<bool,2>,3> initialized{};
    uint64_t seed=0;
    uint32_t argmax=0;
    uint32_t precision=0;
    float heading=0;
    bool sampling_set=false, feature_masks_set=false;
    std::shared_ptr<motionbricks::detail::soma_api_skeleton> skeleton;
    std::array<std::vector<float>,2> transforms;
};
namespace motionbricks::detail {
mb_status load_soma_model(const char *, const mb_runtime_options *, mb_model **, char *, uint64_t);
mb_status infer_soma(const mb_model *, const mb_inference_request *, mb_motion **, char *, uint64_t);
}
