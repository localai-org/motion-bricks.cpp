#pragma once

#include <motionbricks/motionbricks.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

struct mb_model;
struct mb_motion;

namespace motionbricks::detail {

constexpr std::uint32_t g1_joint_count = 34U;
constexpr std::uint32_t boundary_frame_count = 4U;
constexpr std::uint32_t global_root_width = 5U;
constexpr std::uint32_t local_root_width = 4U;
constexpr std::uint32_t external_pose_width = 303U;
constexpr std::uint32_t internal_pose_width = 304U;
constexpr std::uint32_t local_motion_width = 413U;

struct encoded_frames {
    std::array<float, boundary_frame_count * global_root_width> global_root{};
    std::array<float, boundary_frame_count * local_root_width> local_root{};
    std::array<float, boundary_frame_count * external_pose_width> poses{};
};

/* Converts public root positions and local XYZW rotations into the exact raw
   sparse features used by the released demo. The last four input frames are
   used when a longer history is supplied. Input is already in the canonical
   model frame. Virtual hand/toe endpoints retain FK positions but use global
   identity orientation features, matching upstream's dummy-joint scheme. */
mb_status encode_context(const mb_model & model,
                         std::span<const float> root_xyz,
                         std::span<const float> local_rotation_xyzw,
                         std::uint64_t frames,
                         encoded_frames & output,
                         std::string & reason);

/* Encodes four already-global frames. Rotations are row-major 3x3 matrices;
   positions include the root and are in world space. */
mb_status encode_global_frames(std::span<const float> joint_positions,
                               std::span<const float> joint_rotations,
                               bool repeat_last_velocity,
                               encoded_frames & output,
                               std::string & reason);

float normalize_feature(const mb_model & model, float value, std::uint32_t dual_index);
float unnormalize_feature(const mb_model & model, float value, std::uint32_t dual_index);

/* Converts the decoder's normalized local representation to the public
   animation boundary. */
mb_status decode_motion(const mb_model & model,
                        std::span<const float> normalized_local_motion,
                        std::uint32_t frames,
                        float initial_x, float initial_z,
                        float initial_heading,
                        mb_motion & output,
                        std::string & reason);

} // namespace motionbricks::detail
