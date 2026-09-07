#pragma once
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace motionbricks::detail {
// SplitMix64 with an explicitly defined 24-bit [0,1) float mapping. No
// process-global RNG or standard-library-dependent uniform distribution.
float sampling_uniform(std::uint64_t & state);
float gumbel_noise(float uniform);
bool sample_pose_tokens(std::span<const float> logits, std::span<const float> uniforms,
                        std::uint32_t choices, std::vector<std::int32_t> & tokens,
                        std::string & reason);
}
