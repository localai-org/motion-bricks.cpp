#include <motionbricks/motionbricks.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#ifndef MOTIONBRICKS_REFERENCE_BUNDLE
#error MOTIONBRICKS_REFERENCE_BUNDLE is required
#endif
#ifndef MOTIONBRICKS_REFERENCE_STYLES
#error MOTIONBRICKS_REFERENCE_STYLES is required
#endif

namespace {
struct snapshot { std::uint64_t frames{}; std::vector<float> roots, rotations; };

bool plan(const char * bundle, mb_device device, snapshot & output) {
    char error[1024]{};
    mb_runtime_options * options = nullptr; mb_model * model = nullptr;
    mb_style * style = nullptr; mb_agent * agent = nullptr;
    mb_command * command = nullptr; mb_motion * motion = nullptr;
    auto ok = [&](mb_status status) {
        if (status == MB_OK) return true;
        std::cerr << mb_status_string(status) << ": " << error << '\n'; return false;
    };
    bool valid = ok(mb_runtime_options_create(&options, error, sizeof error)) &&
        ok(mb_runtime_options_set_device(options, device, error, sizeof error)) &&
        ok(mb_model_load(bundle, options, &model, error, sizeof error)) &&
        ok(mb_style_load(model, MOTIONBRICKS_REFERENCE_STYLES "/walk.mbstyle", &style, error, sizeof error)) &&
        ok(mb_agent_create(model, &agent, error, sizeof error)) &&
        ok(mb_agent_reset(agent, style, error, sizeof error)) &&
        ok(mb_command_create(&command, error, sizeof error)) &&
        ok(mb_command_set_style(command, style, error, sizeof error)) &&
        ok(mb_command_set_movement_direction(command, 0.0F, 0.0F, 1.0F, error, sizeof error)) &&
        ok(mb_command_set_facing_direction(command, 0.0F, 0.0F, 1.0F, error, sizeof error)) &&
        ok(mb_command_set_seed(command, 23U, error, sizeof error)) &&
        ok(mb_agent_plan(agent, command, &motion, error, sizeof error));
    if (valid) {
        const float * roots = nullptr, * rotations = nullptr;
        std::uint64_t root_count = 0, rotation_count = 0;
        valid = ok(mb_motion_get_frame_count(motion, &output.frames, error, sizeof error)) &&
            ok(mb_motion_get_root_translations(motion, &roots, &root_count, error, sizeof error)) &&
            ok(mb_motion_get_local_rotations_xyzw(motion, &rotations, &rotation_count, error, sizeof error));
        if (valid) { output.roots.assign(roots, roots + root_count); output.rotations.assign(rotations, rotations + rotation_count); }
    }
    mb_motion_free(motion); mb_command_free(command); mb_agent_free(agent);
    mb_style_free(style); mb_model_free(model); mb_runtime_options_free(options);
    return valid;
}

float maximum_error(const std::vector<float> & left, const std::vector<float> & right) {
    if (left.size() != right.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index)
        result = std::max(result, std::abs(left[index] - right[index]));
    return result;
}
}

int main(int argc, char ** argv) {
    const bool compare_bundles = argc == 3;
    if (argc != 1 && !compare_bundles) {
        std::cerr << "usage: motionbricks-agent-backend-parity-test [REFERENCE_BUNDLE TEST_BUNDLE]\n";
        return 2;
    }
    const char * first_bundle = compare_bundles ? argv[1] : MOTIONBRICKS_REFERENCE_BUNDLE;
    const char * second_bundle = compare_bundles ? argv[2] : MOTIONBRICKS_REFERENCE_BUNDLE;
    const auto second_device = compare_bundles ? MB_DEVICE_CPU : MB_DEVICE_VULKAN;
    snapshot first, second;
    if (!plan(first_bundle, MB_DEVICE_CPU, first) ||
        !plan(second_bundle, second_device, second)) return 1;
    const float root_error = maximum_error(first.roots, second.roots);
    const float rotation_error = maximum_error(first.rotations, second.rotations);
    std::cout << "agent backend parity frames=" << first.frames
              << " root_max_abs=" << root_error
              << " rotation_max_abs=" << rotation_error << '\n';
    return first.frames == second.frames && root_error <= 2.0e-3F &&
        rotation_error <= 2.0e-3F ? 0 : 1;
}
