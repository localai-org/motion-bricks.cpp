#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace motionbricks::replay {

inline constexpr std::uint32_t target_frames = 4;

struct session {
    std::uint32_t fps{};
    std::uint32_t frame_count{};
    std::uint32_t joint_count{};
    std::uint32_t qpos_count{};
    std::uint32_t plan_count{};
    std::vector<std::int32_t> parents;
    std::vector<std::int32_t> modes;
    std::vector<std::int32_t> frame_plans;
    std::vector<float> qpos;
    std::vector<float> joint_positions;
    std::vector<std::uint32_t> plan_frames;
    std::vector<std::int32_t> plan_modes;
    std::vector<std::uint32_t> plan_valid_lengths;
    std::vector<float> target_positions;

    [[nodiscard]] const float * frame_joints(std::uint32_t frame) const;
    [[nodiscard]] const float * plan_targets(std::uint32_t plan) const;
};

[[nodiscard]] session load(const std::filesystem::path & path);

} // namespace motionbricks::replay
