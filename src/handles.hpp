#pragma once

#include <motionbricks/motionbricks.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace motionbricks::detail {
class neural_runtime;
}

struct mb_runtime_options {
    mb_device device = MB_DEVICE_AUTO;
    std::uint32_t threads = 0;
    std::string backend_directory;
};

struct mb_model {
    std::string bundle_directory;
    std::uint64_t parameter_count = 0;
    std::vector<std::string> joint_names;
    std::vector<std::int32_t> joint_parents;
    std::vector<float> neutral_joints;
    std::vector<float> motion_mean;
    std::vector<float> motion_std;
    std::shared_ptr<motionbricks::detail::neural_runtime> runtime;
};

struct mb_style {
    std::string name;
    float speed = 0.0F;
    std::uint32_t frames = 0;
    std::vector<float> global_joint_positions;
    std::vector<float> global_joint_rotations;
    std::vector<float> global_root_positions;
    std::vector<float> global_headings;
    std::array<std::uint8_t, 11> allowed_tokens{};
};

struct mb_command {
    const mb_style * style = nullptr;
    std::array<float, 3> movement_direction{0.0F, 0.0F, 0.0F};
    std::array<float, 3> facing_direction{0.0F, 0.0F, 1.0F};
    float target_speed = -1.0F;
    std::array<float, 3> world_target{0.0F, 0.0F, 0.0F};
    float world_target_heading = 0.0F;
    std::uint32_t has_world_target = 0;
    // Internal diagnostic switch. The public controller always applies the
    // upstream four-frame playback seam; the parity runner disables it when
    // comparing the raw neural output stored in an open-loop fixture.
    std::uint32_t skip_context_blend = 0;
    std::uint64_t seed = 0;
    std::uint32_t sampling_argmax = 0;
};

struct mb_agent {
    const mb_model * model = nullptr;
    const mb_style * initial_style = nullptr;
    std::vector<float> context_root_xyz;
    std::vector<float> context_local_rotations_xyzw;
    std::uint64_t context_frames = 0;
    std::unique_ptr<mb_motion> current_motion;
    std::uint32_t current_frame = 0;
};

struct mb_motion {
    std::uint64_t frames = 0;
    std::uint64_t joints = 0;
    std::vector<float> root_translations;
    std::vector<float> local_rotations_xyzw;
    std::uint64_t target_frames = 0;
    std::vector<float> target_root_translations;
    std::vector<float> target_local_rotations_xyzw;
};
