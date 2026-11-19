#pragma once
#include "soma.hpp"
#include <motionbricks/motionbricks.h>
#include <array>
#include <span>
#include <string>
#include <vector>

namespace motionbricks::detail {
struct soma_request;
// Position XYZ followed by quaternion XYZW, matching upstream Warp transforms.
using soma_transform = std::array<float, 7>;
struct soma_skeleton {
    std::vector<int> parents, pose_to_model;
    std::vector<soma_transform> reference;
};
struct soma_playback {
    unsigned frames{};
    std::vector<soma_transform> decoded, corrected, world, start_deltas, end_deltas;
    std::vector<float> contacts;
};
mb_status encode_soma_poses(const soma_skeleton &, std::span<const int> model_to_pose,
    unsigned right_leg, unsigned left_leg, std::span<const soma_transform> context,
    std::span<const soma_transform> target, const soma_transform & runtime_to_model,
    unsigned allowed_token_bits, soma_request &, soma_transform & alignment,
    std::vector<soma_transform> & world, std::string & reason, float desired_heading_angle = 0);
mb_status decode_soma_playback(std::span<const float> features, unsigned tokens,
    const soma_skeleton &, const soma_transform & alignment,
    const soma_transform & model_to_runtime, std::span<const soma_transform> context,
    std::span<const soma_transform> target, soma_playback &, std::string & reason);

struct soma_motion_request {
    const soma_skeleton * skeleton{};
    std::span<const int> model_to_pose;
    unsigned right_leg{},left_leg{},allowed_token_bits=2047;
    std::span<const soma_transform> context,target;
    soma_transform runtime_to_model{0,0,0,0,0,0,1},model_to_runtime{0,0,0,0,0,0,1};
    bool mixed_fp16=false;
};
struct soma_motion_result {
    soma_request encoded;
    soma_transform alignment{};
    std::vector<soma_transform> input_world;
    soma_result prediction;
    soma_playback playback;
    soma_boundaries boundaries;
};
// Borrowed request spans must remain valid through the call. Neural execution
// batches compatible requests; encoding and playback retain each request's rig.
// Failures publish no partial results. Processing taps are optional.
mb_status run_soma_motion_batch(const neural_runtime &,std::span<const soma_motion_request>,
    std::vector<soma_motion_result> &,std::string &,bool capture_boundaries=false);
}
