#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace motionbricks::parity {

inline constexpr std::uint32_t joint_count = 34;
inline constexpr std::uint32_t context_frames = 4;
inline constexpr std::uint32_t target_frames = 4;

struct plan {
    std::uint32_t command_frame{};
    std::uint32_t expected_frames{};
    std::int32_t mode{};
    std::uint64_t seed{};
    std::array<float, 3> movement{};
    std::array<float, 3> facing{};
    std::vector<float> context_roots;
    std::vector<float> context_local_rotations;
    std::vector<float> expected_roots;
    std::vector<float> expected_local_rotations;
    std::vector<float> expected_joint_positions;
    std::vector<float> target_roots;
    std::vector<float> target_local_rotations;
    std::vector<float> target_joint_positions;
};

struct fixture {
    std::uint32_t fps{};
    std::vector<std::int32_t> parents;
    std::vector<float> neutral_joints;
    std::vector<plan> plans;
};

[[nodiscard]] fixture load(const std::filesystem::path & path);

} // namespace motionbricks::parity
