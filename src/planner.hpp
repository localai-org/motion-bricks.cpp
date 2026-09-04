#pragma once

#include "motion_rep.hpp"

#include <motionbricks/motionbricks.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

struct mb_model;
struct mb_motion;

namespace motionbricks::detail {

struct transition_constraints {
    std::array<float, 8U * global_root_width> global_root{};
    std::array<float, 8U * local_root_width> local_root{};
    std::array<float, 8U * external_pose_width> poses{};
    std::array<std::uint8_t, 8U> has_global_root{1,1,1,1,1,1,1,1};
    std::array<std::uint8_t, 8U> has_local_root{1,1,1,0,1,1,1,1};
    std::array<std::uint8_t, 8U> has_poses{1,1,1,1,1,1,1,1};
    std::array<std::uint8_t, 11U> allowed_tokens{1,1,1,1,1,1,1,1,1,1,1};
    std::array<float, 4U * 3U> target_root_translations{};
    std::array<float, 4U * g1_joint_count * 4U> target_local_rotations_xyzw{};
};

// Internal diagnostic boundary used only by parity tooling. These values are
// copied after each stage and are not part of the installed C ABI.
struct transition_trace {
    transition_constraints constraints;
    std::vector<float> normalized_global_root;
    std::vector<float> normalized_local_root;
    std::vector<float> normalized_poses;
    std::vector<float> duration_logits;
    std::uint32_t selected_tokens{};
    std::vector<float> predicted_global_root;
    std::vector<float> predicted_local_root;
    std::vector<float> pose_root_condition;
    std::vector<float> pose_condition;
    std::vector<std::uint8_t> has_pose_condition;
    std::vector<float> pose_logits;
    std::vector<std::int32_t> pose_tokens;
    std::vector<float> decoder_quantized;
    std::vector<float> decoder_external_condition;
    std::vector<std::uint8_t> decoder_target_mask;
    std::vector<float> decoder_output;
};

mb_status run_transition(const mb_model & model,
                         const transition_constraints & constraints,
                         mb_motion & output,
                         std::uint32_t * selected_tokens,
                         transition_trace * trace,
                         std::string & reason);

} // namespace motionbricks::detail
