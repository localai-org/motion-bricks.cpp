#include "sampling.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace motionbricks::detail {
float sampling_uniform(std::uint64_t & state) {
    auto z = (state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return static_cast<float>(z >> 40) * 0x1p-24F;
}

float gumbel_noise(float uniform) {
    // Matches NVIDIA motion_backbone/models/sampling.py (temperature=1):
    // log(x) = torch.log(x.clamp(min=1e-20)); -log(-log(uniform)).
    return -std::log(std::max(-std::log(std::max(uniform, 1e-20F)), 1e-20F));
}

bool sample_pose_tokens(std::span<const float> logits, std::span<const float> uniforms,
                        std::uint32_t choices, std::vector<std::int32_t> & tokens,
                        std::string & reason) {
    if (choices == 0 || choices > 1024 || logits.empty() || logits.size() % choices ||
        (!uniforms.empty() && uniforms.size() != logits.size())) {
        reason = "invalid sampling shape"; return false;
    }
    for (float value : logits) if (!std::isfinite(value)) {
        reason = "sampling logits are non-finite"; return false;
    }
    for (float value : uniforms) if (!std::isfinite(value) || value < 0 || value > 1) {
        reason = "sampling uniform must be finite and in [0,1]"; return false;
    }
    tokens.resize(logits.size() / choices);
    for (std::size_t row = 0; row < tokens.size(); ++row) {
        float best = -std::numeric_limits<float>::infinity();
        std::int32_t selected = 0;
        for (std::uint32_t choice = 0; choice < choices; ++choice) {
            const auto index = row * choices + choice;
            const float score = logits[index] + (uniforms.empty() ? 0.0F : gumbel_noise(uniforms[index]));
            if (score > best) { best = score; selected = static_cast<std::int32_t>(choice); }
        }
        tokens[row] = selected; // First maximum wins, as in torch.argmax.
    }
    return true;
}
}
