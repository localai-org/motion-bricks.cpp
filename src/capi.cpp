#include <motionbricks/motionbricks.h>

#include "error.hpp"
#include "agent.hpp"
#include "handles.hpp"
#include "model.hpp"
#include "style.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>

namespace {

using motionbricks::detail::fail;
using motionbricks::detail::guard;

bool valid_device(mb_device device) noexcept {
    return device == MB_DEVICE_AUTO || device == MB_DEVICE_CPU || device == MB_DEVICE_VULKAN;
}

bool finite3(float x, float y, float z) noexcept {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

} // namespace

extern "C" {

uint32_t mb_abi_version(void) { return MB_ABI_VERSION; }

const char * mb_status_string(mb_status status) {
    switch (status) {
    case MB_OK: return "ok";
    case MB_INVALID_ARGUMENT: return "invalid argument";
    case MB_OUT_OF_MEMORY: return "out of memory";
    case MB_IO_ERROR: return "I/O error";
    case MB_INVALID_FORMAT: return "invalid format";
    case MB_INCOMPATIBLE_MODEL: return "incompatible model";
    case MB_BACKEND_UNAVAILABLE: return "backend unavailable";
    case MB_COMPUTE_FAILED: return "compute failed";
    case MB_NOT_IMPLEMENTED: return "not implemented";
    case MB_INTERNAL_ERROR: return "internal error";
    default: return "unknown status";
    }
}

mb_status mb_runtime_options_create(mb_runtime_options ** output,
                                    char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        *output = new mb_runtime_options();
        return MB_OK;
    });
}

void mb_runtime_options_free(mb_runtime_options * value) { delete value; }

mb_status mb_runtime_options_set_device(mb_runtime_options * value, mb_device device,
                                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options are null");
        if (!valid_device(device)) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "invalid device");
        value->device = device;
        return MB_OK;
    });
}

mb_status mb_runtime_options_get_device(const mb_runtime_options * value, mb_device * output,
                                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options or output is null");
        *output = value->device;
        return MB_OK;
    });
}

mb_status mb_runtime_options_set_threads(mb_runtime_options * value, uint32_t threads,
                                         char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options are null");
        value->threads = threads;
        return MB_OK;
    });
}

mb_status mb_runtime_options_get_threads(const mb_runtime_options * value, uint32_t * output,
                                         char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options or output is null");
        *output = value->threads;
        return MB_OK;
    });
}

mb_status mb_runtime_options_set_backend_directory(mb_runtime_options * value,
                                                   const char * directory,
                                                   char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options are null");
        value->backend_directory = directory != nullptr ? directory : "";
        return MB_OK;
    });
}

mb_status mb_runtime_options_get_backend_directory(const mb_runtime_options * value,
                                                   const char ** output,
                                                   char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "options or output is null");
        *output = value->backend_directory.c_str();
        return MB_OK;
    });
}

mb_status mb_command_create(mb_command ** output, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        *output = new mb_command();
        return MB_OK;
    });
}

void mb_command_free(mb_command * value) { delete value; }

mb_status mb_command_set_style(mb_command * value, const mb_style * style,
                               char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        value->style = style;
        return MB_OK;
    });
}

mb_status mb_command_get_style(const mb_command * value, const mb_style ** output,
                               char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *output = value->style;
        return MB_OK;
    });
}

mb_status mb_command_set_movement_direction(mb_command * value, float x, float y, float z,
                                            char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        if (!finite3(x, y, z)) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "movement direction is not finite");
        value->movement_direction = {x, y, z};
        return MB_OK;
    });
}

mb_status mb_command_get_movement_direction(const mb_command * value,
                                            float * x, float * y, float * z,
                                            char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || x == nullptr || y == nullptr || z == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *x = value->movement_direction[0];
        *y = value->movement_direction[1];
        *z = value->movement_direction[2];
        return MB_OK;
    });
}

mb_status mb_command_set_facing_direction(mb_command * value, float x, float y, float z,
                                          char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        if (!finite3(x, y, z)) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "facing direction is not finite");
        if (x * x + y * y + z * z <= 1.0e-12F)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "facing direction is zero");
        value->facing_direction = {x, y, z};
        return MB_OK;
    });
}

mb_status mb_command_get_facing_direction(const mb_command * value,
                                          float * x, float * y, float * z,
                                          char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || x == nullptr || y == nullptr || z == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *x = value->facing_direction[0];
        *y = value->facing_direction[1];
        *z = value->facing_direction[2];
        return MB_OK;
    });
}

mb_status mb_command_set_target_speed(mb_command * value, float speed,
                                      char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        if (!std::isfinite(speed) || speed < -1.0F)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "target speed must be finite and at least -1");
        value->target_speed = speed;
        return MB_OK;
    });
}

mb_status mb_command_get_target_speed(const mb_command * value, float * output,
                                      char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *output = value->target_speed;
        return MB_OK;
    });
}

mb_status mb_command_set_world_target(mb_command * value, float x, float y, float z,
                                      float heading, uint32_t enabled,
                                      char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        if (!finite3(x, y, z) || !std::isfinite(heading))
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "world target is not finite");
        if (enabled > 1) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "enabled must be 0 or 1");
        value->world_target = {x, y, z};
        value->world_target_heading = heading;
        value->has_world_target = enabled;
        return MB_OK;
    });
}

mb_status mb_command_get_world_target(const mb_command * value,
                                      float * x, float * y, float * z,
                                      float * heading, uint32_t * enabled,
                                      char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || x == nullptr || y == nullptr || z == nullptr ||
            heading == nullptr || enabled == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *x = value->world_target[0];
        *y = value->world_target[1];
        *z = value->world_target[2];
        *heading = value->world_target_heading;
        *enabled = value->has_world_target;
        return MB_OK;
    });
}

mb_status mb_command_set_seed(mb_command * value, uint64_t seed,
                              char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command is null");
        value->seed = seed;
        return MB_OK;
    });
}

mb_status mb_command_set_sampling_argmax(mb_command * value, uint32_t enabled,
                                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (!value || enabled > 1)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "sampling argmax requires a command and 0 or 1");
        value->sampling_argmax = enabled;
        return MB_OK;
    });
}

mb_status mb_command_get_sampling_argmax(const mb_command * value, uint32_t * output,
                                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (!value || !output)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *output = value->sampling_argmax;
        return MB_OK;
    });
}

mb_status mb_command_get_seed(const mb_command * value, uint64_t * output,
                              char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "command or output is null");
        *output = value->seed;
        return MB_OK;
    });
}

mb_status mb_model_load(const char * bundle_directory, const mb_runtime_options * options, mb_model ** output,
                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        if (bundle_directory == nullptr || *bundle_directory == '\0')
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "bundle directory is empty");
        if (options != nullptr && !valid_device(options->device))
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "invalid device in runtime options");
#if !defined(MOTIONBRICKS_HAVE_VULKAN)
        if (options != nullptr && options->device == MB_DEVICE_VULKAN)
            return fail(MB_BACKEND_UNAVAILABLE, error, error_capacity,
                        "this build has no Vulkan backend");
#endif
        auto model = std::make_unique<mb_model>();
        std::string reason;
        const auto status = motionbricks::detail::load_model_bundle(bundle_directory, options, *model, reason);
        if (status != MB_OK) return fail(status, error, error_capacity, reason);
        *output = model.release();
        return MB_OK;
    });
}

void mb_model_free(mb_model * value) { delete value; }

mb_status mb_model_get_parameter_count(const mb_model * value, uint64_t * output,
                                       char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or output is null");
        *output = value->parameter_count;
        return MB_OK;
    });
}

mb_status mb_model_get_joint_count(const mb_model * value, uint32_t * output,
                                   char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or output is null");
        *output = static_cast<std::uint32_t>(value->joint_names.size());
        return MB_OK;
    });
}

mb_status mb_model_get_joint_name(const mb_model * value, uint32_t joint, const char ** output,
                                  char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or output is null");
        if (joint >= value->joint_names.size())
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "joint index is out of range");
        *output = value->joint_names[joint].c_str();
        return MB_OK;
    });
}

mb_status mb_model_get_joint_parent(const mb_model * value, uint32_t joint, int32_t * output,
                                    char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or output is null");
        if (joint >= value->joint_parents.size())
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "joint index is out of range");
        *output = value->joint_parents[joint];
        return MB_OK;
    });
}

mb_status mb_model_get_neutral_joint_position(const mb_model * value, uint32_t joint,
                                              float * x, float * y, float * z,
                                              char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || x == nullptr || y == nullptr || z == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or output is null");
        if (joint >= value->joint_names.size() || value->neutral_joints.size() != value->joint_names.size() * 3U)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "joint index is out of range");
        *x = value->neutral_joints[static_cast<std::size_t>(joint) * 3U];
        *y = value->neutral_joints[static_cast<std::size_t>(joint) * 3U + 1U];
        *z = value->neutral_joints[static_cast<std::size_t>(joint) * 3U + 2U];
        return MB_OK;
    });
}

mb_status mb_style_load(const mb_model * model, const char * style_path, mb_style ** output,
                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        if (model == nullptr || style_path == nullptr || *style_path == '\0')
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model or style path is null");
        auto style = std::make_unique<mb_style>();
        std::string reason;
        const auto status = motionbricks::detail::load_style_file(style_path, *style, reason);
        if (status != MB_OK) return fail(status, error, error_capacity, reason);
        *output = style.release();
        return MB_OK;
    });
}

void mb_style_free(mb_style * value) { delete value; }

mb_status mb_style_get_name(const mb_style * value, const char ** output, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "style or output is null");
        *output = value->name.c_str();
        return MB_OK;
    });
}

mb_status mb_style_set_speed(mb_style * value, float speed, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "style is null");
        if (!std::isfinite(speed) || speed < 0.0F)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "style speed must be finite and non-negative");
        value->speed = speed;
        return MB_OK;
    });
}

mb_status mb_style_get_speed(const mb_style * value, float * output, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "style or output is null");
        *output = value->speed;
        return MB_OK;
    });
}

mb_status mb_agent_create(const mb_model * model, mb_agent ** output,
                          char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        if (model == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "model is null");
        auto agent = std::make_unique<mb_agent>();
        agent->model = model;
        *output = agent.release();
        return MB_OK;
    });
}

void mb_agent_free(mb_agent * value) { delete value; }

mb_status mb_agent_reset(mb_agent * value, const mb_style * style, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || style == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "agent or style is null");
        std::string reason;
        const auto status = motionbricks::detail::seed_agent_from_style(*value, *style, reason);
        return status == MB_OK ? MB_OK : fail(status, error, error_capacity, reason);
    });
}

mb_status mb_agent_set_context(mb_agent * value, const float * roots, const float * rotations,
                               uint64_t frames, uint64_t joints,
                               char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || roots == nullptr || rotations == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "agent or context data is null");
        if (frames < 4U || joints != 34U || frames > 1000000U)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "context must contain at least four frames and 34 joints");
        const auto root_count = static_cast<std::size_t>(frames) * 3U;
        const auto rotation_count = static_cast<std::size_t>(frames) * 34U * 4U;
        if (!std::all_of(roots, roots + root_count, [](float item) { return std::isfinite(item); }) ||
            !std::all_of(rotations, rotations + rotation_count, [](float item) { return std::isfinite(item); }))
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "context contains non-finite values");
        value->context_root_xyz.assign(roots, roots + root_count);
        value->context_local_rotations_xyzw.assign(rotations, rotations + rotation_count);
        value->context_frames = frames;
        value->current_motion.reset();
        value->current_frame = 0U;
        return MB_OK;
    });
}

mb_status mb_agent_plan(mb_agent * value, const mb_command * command, mb_motion ** output,
                        char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (output == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "output is null");
        *output = nullptr;
        if (value == nullptr || command == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "agent or command is null");
        auto motion = std::make_unique<mb_motion>();
        std::string reason;
        const auto status = motionbricks::detail::plan_agent(*value, *command, *motion, reason);
        if (status != MB_OK) return fail(status, error, error_capacity, reason);
        value->current_motion = std::make_unique<mb_motion>(*motion);
        value->current_frame = 0U;
        *output = motion.release();
        return MB_OK;
    });
}

mb_status mb_agent_plan_batch(mb_agent * const * agents,
                              const mb_command * const * commands,
                              uint64_t count, mb_motion ** outputs,
                              char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (outputs == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "outputs is null");
        if (count > 64U)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity,
                        "batch count must be in the range 1..64");
        for (uint64_t index = 0; index < count; ++index) outputs[index] = nullptr;
        if (agents == nullptr || commands == nullptr || count == 0U)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity,
                        "agents, commands, or batch count is invalid");
        std::vector<std::unique_ptr<mb_motion>> owned(count);
        std::vector<mb_motion *> raw(count);
        std::vector<mb_agent *> agent_values(count);
        std::vector<const mb_command *> command_values(count);
        for (uint64_t index = 0; index < count; ++index) {
            owned[index] = std::make_unique<mb_motion>();
            raw[index] = owned[index].get();
            agent_values[index] = agents[index];
            command_values[index] = commands[index];
        }
        std::string reason;
        const auto status = motionbricks::detail::plan_agent_batch(
            agent_values, command_values, raw, reason);
        if (status != MB_OK) return fail(status, error, error_capacity, reason);
        for (uint64_t index = 0; index < count; ++index) {
            agents[index]->current_motion = std::make_unique<mb_motion>(*owned[index]);
            agents[index]->current_frame = 0U;
        }
        for (uint64_t index = 0; index < count; ++index)
            outputs[index] = owned[index].release();
        return MB_OK;
    });
}

mb_status mb_agent_advance(mb_agent * value, uint32_t frames, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr) return fail(MB_INVALID_ARGUMENT, error, error_capacity, "agent is null");
        std::string reason;
        const auto status = motionbricks::detail::advance_agent(*value, frames, reason);
        return status == MB_OK ? MB_OK : fail(status, error, error_capacity, reason);
    });
}

void mb_motion_free(mb_motion * value) { delete value; }

mb_status mb_motion_get_frame_count(const mb_motion * value, uint64_t * output, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->frames;
        return MB_OK;
    });
}

mb_status mb_motion_get_joint_count(const mb_motion * value, uint64_t * output, char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->joints;
        return MB_OK;
    });
}

mb_status mb_motion_get_root_translations(const mb_motion * value, const float ** output, uint64_t * values,
                                          char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr || values == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->root_translations.data();
        *values = value->root_translations.size();
        return MB_OK;
    });
}

mb_status mb_motion_get_local_rotations_xyzw(const mb_motion * value, const float ** output, uint64_t * values,
                                             char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr || values == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->local_rotations_xyzw.data();
        *values = value->local_rotations_xyzw.size();
        return MB_OK;
    });
}

mb_status mb_motion_get_target_frame_count(const mb_motion * value, uint64_t * output,
                                           char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->target_frames;
        return MB_OK;
    });
}

mb_status mb_motion_get_target_root_translations(const mb_motion * value, const float ** output, uint64_t * values,
                                                 char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr || values == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->target_root_translations.data();
        *values = value->target_root_translations.size();
        return MB_OK;
    });
}

mb_status mb_motion_get_target_local_rotations_xyzw(const mb_motion * value, const float ** output, uint64_t * values,
                                                    char * error, uint64_t error_capacity) {
    return guard(error, error_capacity, [&]() -> mb_status {
        if (value == nullptr || output == nullptr || values == nullptr)
            return fail(MB_INVALID_ARGUMENT, error, error_capacity, "motion or output is null");
        *output = value->target_local_rotations_xyzw.data();
        *values = value->target_local_rotations_xyzw.size();
        return MB_OK;
    });
}

} // extern "C"
