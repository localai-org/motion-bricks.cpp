#include "handles.hpp"
#include "model.hpp"
#include "pose.hpp"

#include <ggml.h>
#include <gguf.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef MOTIONBRICKS_REFERENCE_BUNDLE
#error MOTIONBRICKS_REFERENCE_BUNDLE is required
#endif
#ifndef MOTIONBRICKS_REFERENCE_FIXTURES
#error MOTIONBRICKS_REFERENCE_FIXTURES is required
#endif

namespace {

struct gguf_deleter { void operator()(gguf_context * value) const { gguf_free(value); } };
struct ggml_deleter { void operator()(ggml_context * value) const { ggml_free(value); } };

template<class T>
std::vector<T> tensor_values(ggml_context * context, const char * name) {
    auto * tensor = ggml_get_tensor(context, name);
    assert(tensor != nullptr);
    std::vector<T> values(static_cast<std::size_t>(ggml_nelements(tensor)));
    std::memcpy(values.data(), tensor->data, values.size() * sizeof(T));
    return values;
}

} // namespace

int main(int argc, char ** argv) {
    mb_model model;
    std::string reason;
    mb_runtime_options options;
    const bool vulkan = (argc >= 2 && std::string_view(argv[1]) == "vulkan") ||
                        (argc >= 3 && std::string_view(argv[2]) == "vulkan");
    const std::string_view bundle = argc >= 2 && std::string_view(argv[1]) != "vulkan"
        ? std::string_view(argv[1]) : std::string_view(MOTIONBRICKS_REFERENCE_BUNDLE);
    options.device = vulkan ? MB_DEVICE_VULKAN : MB_DEVICE_CPU;
    auto status = motionbricks::detail::load_model_bundle(
        bundle, &options, model, reason);
    if (status != MB_OK) {
        std::cerr << reason << '\n';
        return 1;
    }

    ggml_context * raw = nullptr;
    const std::string fixture_path = std::string(MOTIONBRICKS_REFERENCE_FIXTURES) + "/pose.gguf";
    std::unique_ptr<gguf_context, gguf_deleter> fixture(
        gguf_init_from_file(fixture_path.c_str(), {false, &raw}));
    std::unique_ptr<ggml_context, ggml_deleter> fixture_tensors(raw);
    assert(fixture && fixture_tensors);
    const auto tokens = tensor_values<std::int32_t>(fixture_tensors.get(), "input.pose_tokens");
    const auto root = tensor_values<float>(fixture_tensors.get(), "input.local_root_values");
    const auto condition = tensor_values<float>(fixture_tensors.get(), "input.pose_cond");
    const auto mask_i32 = tensor_values<std::int32_t>(fixture_tensors.get(), "input.has_pose_cond");
    const auto duration = tensor_values<std::int32_t>(fixture_tensors.get(), "input.num_tokens");
    const auto expected = tensor_values<float>(fixture_tensors.get(), "output.pose_logits");
    assert(tokens.size() % 8U == 0U);
    const auto positions = static_cast<std::uint32_t>(tokens.size() / 8U);
    assert(root.size() == static_cast<std::size_t>(positions) * 4U * 4U);
    assert(condition.size() == static_cast<std::size_t>(positions) * 4U * 304U);
    assert(mask_i32.size() == static_cast<std::size_t>(positions) * 4U);
    std::vector<std::uint8_t> mask(mask_i32.begin(), mask_i32.end());
    std::vector<float> actual;
    status = motionbricks::detail::run_pose_planner(
        *model.runtime, tokens, root, condition, mask, positions,
        static_cast<std::uint32_t>(duration[0]),
        actual, reason);
    if (status != MB_OK) {
        std::cerr << reason << '\n';
        return 1;
    }
    assert(actual.size() == expected.size());
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
    std::size_t token_mismatches = 0;
    const auto compared = std::min(actual.size(), static_cast<std::size_t>(duration[0]) * 80U);
    for (std::size_t index = 0; index < compared; ++index) {
        max_absolute = std::max(max_absolute, std::abs(actual[index] - expected[index]));
        max_relative = std::max(max_relative,
            std::abs(actual[index] - expected[index]) / std::max(1.0e-6F, std::abs(expected[index])));
    }
    for (std::size_t head = 0; head < compared / 10U; ++head) {
        const auto actual_begin = actual.begin() + static_cast<std::ptrdiff_t>(head * 10U);
        const auto expected_begin = expected.begin() + static_cast<std::ptrdiff_t>(head * 10U);
        if (std::max_element(actual_begin, actual_begin + 10) - actual_begin !=
            std::max_element(expected_begin, expected_begin + 10) - expected_begin)
            ++token_mismatches;
    }
    std::cout << "pose " << (vulkan ? "vulkan" : "cpu")
              << " max_abs=" << max_absolute << " max_rel=" << max_relative
              << " token_mismatches=" << token_mismatches << '\n';
    return max_absolute <= (vulkan ? 5.0e-2F : 3.0e-4F) && token_mismatches == 0U ? 0 : 1;
}
