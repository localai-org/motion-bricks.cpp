#include "motion_rep.hpp"

#include "handles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace motionbricks::detail {
namespace {

constexpr float fps = 30.0F;
constexpr float stats_epsilon = 1.0e-5F;

using vec3 = std::array<float, 3>;
using quat = std::array<float, 4>; // wxyz internally
using mat3 = std::array<float, 9>;

float wrap_angle(float value) {
    constexpr float pi = 3.14159265358979323846F;
    return std::remainder(value, 2.0F * pi);
}

mat3 quaternion_matrix(quat q) {
    const float norm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if (!(norm > 1.0e-12F) || !std::isfinite(norm)) return {};
    const float s = 2.0F / norm;
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    return {
        1.0F - s * (y*y + z*z), s * (x*y - z*w),        s * (x*z + y*w),
        s * (x*y + z*w),        1.0F - s * (x*x + z*z), s * (y*z - x*w),
        s * (x*z - y*w),        s * (y*z + x*w),        1.0F - s * (x*x + y*y),
    };
}

mat3 multiply(const mat3 & a, const mat3 & b) {
    mat3 result{};
    for (unsigned row = 0; row < 3U; ++row)
        for (unsigned column = 0; column < 3U; ++column)
            for (unsigned k = 0; k < 3U; ++k)
                result[row * 3U + column] += a[row * 3U + k] * b[k * 3U + column];
    return result;
}

mat3 transpose(const mat3 & value) {
    return {value[0], value[3], value[6], value[1], value[4], value[7],
            value[2], value[5], value[8]};
}

vec3 transform(const mat3 & matrix, const vec3 & value) {
    return {
        matrix[0]*value[0] + matrix[1]*value[1] + matrix[2]*value[2],
        matrix[3]*value[0] + matrix[4]*value[1] + matrix[5]*value[2],
        matrix[6]*value[0] + matrix[7]*value[1] + matrix[8]*value[2],
    };
}

quat matrix_quaternion(const mat3 & matrix) {
    quat result{};
    const float trace = matrix[0] + matrix[4] + matrix[8];
    if (trace > 0.0F) {
        const float scale = 2.0F * std::sqrt(std::max(0.0F, trace + 1.0F));
        result = {0.25F * scale, (matrix[7] - matrix[5]) / scale,
                  (matrix[2] - matrix[6]) / scale, (matrix[3] - matrix[1]) / scale};
    } else if (matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
        const float scale = 2.0F * std::sqrt(std::max(0.0F, 1.0F + matrix[0] - matrix[4] - matrix[8]));
        result = {(matrix[7] - matrix[5]) / scale, 0.25F * scale,
                  (matrix[1] + matrix[3]) / scale, (matrix[2] + matrix[6]) / scale};
    } else if (matrix[4] > matrix[8]) {
        const float scale = 2.0F * std::sqrt(std::max(0.0F, 1.0F + matrix[4] - matrix[0] - matrix[8]));
        result = {(matrix[2] - matrix[6]) / scale, (matrix[1] + matrix[3]) / scale,
                  0.25F * scale, (matrix[5] + matrix[7]) / scale};
    } else {
        const float scale = 2.0F * std::sqrt(std::max(0.0F, 1.0F + matrix[8] - matrix[0] - matrix[4]));
        result = {(matrix[3] - matrix[1]) / scale, (matrix[2] + matrix[6]) / scale,
                  (matrix[5] + matrix[7]) / scale, 0.25F * scale};
    }
    const float norm = std::sqrt(result[0]*result[0] + result[1]*result[1] +
                                 result[2]*result[2] + result[3]*result[3]);
    if (norm > 0.0F) for (float & value : result) value /= norm;
    if (result[0] < 0.0F) for (float & value : result) value = -value;
    return result;
}

mat3 cont6d_matrix(const float * value) {
    vec3 x{value[0], value[1], value[2]};
    const vec3 y_raw{value[3], value[4], value[5]};
    const auto normalize = [](vec3 & vector) {
        const float length = std::sqrt(vector[0]*vector[0] + vector[1]*vector[1] + vector[2]*vector[2]);
        if (length > 1.0e-12F) for (float & item : vector) item /= length;
    };
    normalize(x);
    vec3 z{x[1]*y_raw[2] - x[2]*y_raw[1],
           x[2]*y_raw[0] - x[0]*y_raw[2],
           x[0]*y_raw[1] - x[1]*y_raw[0]};
    normalize(z);
    const vec3 y{z[1]*x[2] - z[2]*x[1], z[2]*x[0] - z[0]*x[2],
                 z[0]*x[1] - z[1]*x[0]};
    return {x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]};
}

bool finite_values(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

} // namespace

float normalize_feature(const mb_model & model, float value, std::uint32_t dual_index) {
    const float stddev = model.motion_std[dual_index];
    return (value - model.motion_mean[dual_index]) /
           std::sqrt(stddev * stddev + stats_epsilon);
}

float unnormalize_feature(const mb_model & model, float value, std::uint32_t dual_index) {
    const float stddev = model.motion_std[dual_index];
    return value * std::sqrt(stddev * stddev + stats_epsilon) + model.motion_mean[dual_index];
}

mb_status encode_global_frames(std::span<const float> joint_positions,
                               std::span<const float> joint_rotations,
                               bool repeat_last_velocity,
                               encoded_frames & output,
                               std::string & reason) {
    constexpr std::size_t position_count = boundary_frame_count * g1_joint_count * 3U;
    constexpr std::size_t rotation_count = boundary_frame_count * g1_joint_count * 9U;
    if (joint_positions.size() != position_count || joint_rotations.size() != rotation_count) {
        reason = "global frame input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    if (!finite_values(joint_positions) || !finite_values(joint_rotations)) {
        reason = "global frames contain non-finite values";
        return MB_INVALID_ARGUMENT;
    }
    std::array<float, boundary_frame_count> headings{};
    for (std::uint32_t frame = 0; frame < boundary_frame_count; ++frame) {
        const auto position_base = static_cast<std::size_t>(frame) * g1_joint_count * 3U;
        const auto rotation_base = static_cast<std::size_t>(frame) * g1_joint_count * 9U;
        const float * root = joint_positions.data() + position_base;
        const float * root_rotation = joint_rotations.data() + rotation_base;
        headings[frame] = std::atan2(root_rotation[2], root_rotation[8]);
        auto * global = output.global_root.data() + static_cast<std::size_t>(frame) * global_root_width;
        global[0] = root[0]; global[1] = root[1]; global[2] = root[2];
        global[3] = std::cos(headings[frame]); global[4] = std::sin(headings[frame]);

        auto * pose = output.poses.data() + static_cast<std::size_t>(frame) * external_pose_width;
        for (std::uint32_t joint = 1; joint < g1_joint_count; ++joint) {
            const float * position = joint_positions.data() + position_base + joint * 3U;
            const auto destination = static_cast<std::size_t>(joint - 1U) * 3U;
            pose[destination] = position[0] - root[0];
            pose[destination + 1U] = position[1];
            pose[destination + 2U] = position[2] - root[2];
        }
        constexpr std::size_t rotation_offset = (g1_joint_count - 1U) * 3U;
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            const float * matrix = joint_rotations.data() + rotation_base + joint * 9U;
            float * six = pose + rotation_offset + joint * 6U;
            six[0] = matrix[0]; six[1] = matrix[3]; six[2] = matrix[6];
            six[3] = matrix[1]; six[4] = matrix[4]; six[5] = matrix[7];
        }
    }
    for (std::uint32_t frame = 0; frame + 1U < boundary_frame_count; ++frame) {
        auto * local = output.local_root.data() + static_cast<std::size_t>(frame) * local_root_width;
        const auto * current = output.global_root.data() + static_cast<std::size_t>(frame) * global_root_width;
        const auto * next = current + global_root_width;
        local[0] = wrap_angle(headings[frame + 1U] - headings[frame]) * fps;
        local[1] = (next[0] - current[0]) * fps;
        local[2] = (next[2] - current[2]) * fps;
        local[3] = current[1];
    }
    if (repeat_last_velocity) {
        auto * last = output.local_root.data() + 3U * local_root_width;
        std::copy_n(last - local_root_width, local_root_width, last);
        last[3] = output.global_root[3U * global_root_width + 1U];
    }
    return MB_OK;
}

mb_status encode_context(const mb_model & model,
                         std::span<const float> root_xyz,
                         std::span<const float> local_rotation_xyzw,
                         std::uint64_t frames,
                         encoded_frames & output,
                         std::string & reason) {
    if (frames < boundary_frame_count || root_xyz.size() != frames * 3U ||
        local_rotation_xyzw.size() != frames * g1_joint_count * 4U) {
        reason = "context input shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    if (model.neutral_joints.size() != g1_joint_count * 3U ||
        model.joint_parents.size() != g1_joint_count) {
        reason = "model has incomplete skeleton support data";
        return MB_INCOMPATIBLE_MODEL;
    }
    if (!finite_values(root_xyz) || !finite_values(local_rotation_xyzw)) {
        reason = "context contains non-finite values";
        return MB_INVALID_ARGUMENT;
    }
    std::vector<float> positions(boundary_frame_count * g1_joint_count * 3U);
    std::vector<float> rotations(boundary_frame_count * g1_joint_count * 9U);
    const auto first = frames - boundary_frame_count;
    for (std::uint32_t frame = 0; frame < boundary_frame_count; ++frame) {
        const auto source_frame = first + frame;
        const float * root = root_xyz.data() + source_frame * 3U;
        std::array<mat3, g1_joint_count> global{};
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            const float * xyzw = local_rotation_xyzw.data() +
                (source_frame * g1_joint_count + joint) * 4U;
            const mat3 local = quaternion_matrix({xyzw[3], xyzw[0], xyzw[1], xyzw[2]});
            if (local == mat3{}) {
                reason = "context contains a zero rotation quaternion";
                return MB_INVALID_ARGUMENT;
            }
            const auto parent = model.joint_parents[joint];
            global[joint] = parent < 0 ? local : multiply(global[static_cast<std::size_t>(parent)], local);
            float * matrix = rotations.data() +
                (static_cast<std::size_t>(frame) * g1_joint_count + joint) * 9U;
            std::copy(global[joint].begin(), global[joint].end(), matrix);
            float * position = positions.data() +
                (static_cast<std::size_t>(frame) * g1_joint_count + joint) * 3U;
            if (parent < 0) {
                std::copy_n(root, 3, position);
            } else {
                const auto parent_index = static_cast<std::size_t>(parent);
                const float * parent_position = positions.data() +
                    (static_cast<std::size_t>(frame) * g1_joint_count + parent_index) * 3U;
                const vec3 rest{
                    model.neutral_joints[joint * 3U] - model.neutral_joints[parent_index * 3U],
                    model.neutral_joints[joint * 3U + 1U] - model.neutral_joints[parent_index * 3U + 1U],
                    model.neutral_joints[joint * 3U + 2U] - model.neutral_joints[parent_index * 3U + 2U],
                };
                const vec3 offset = transform(global[parent_index], rest);
                for (unsigned axis = 0; axis < 3U; ++axis) position[axis] = parent_position[axis] + offset[axis];
            }
        }
        // The caller has already canonicalized the context. Upstream's
        // mujoco_qpos_converter (dummy scheme) inserts global identity for
        // the four virtual endpoints absent from the physical G1 joints.
        // Override conditioning matrices only: retain parent-derived FK
        // positions and never modify the caller's authored local rotations.
        constexpr mat3 identity{1,0,0,0,1,0,0,0,1};
        for (const auto joint : {7U,14U,25U,33U}) {
            auto * matrix = rotations.data() +
                (static_cast<std::size_t>(frame) * g1_joint_count + joint) * 9U;
            std::copy(identity.begin(), identity.end(), matrix);
        }
    }
    return encode_global_frames(positions, rotations, false, output, reason);
}

mb_status decode_motion(const mb_model & model,
                        std::span<const float> normalized_local_motion,
                        std::uint32_t frames,
                        float initial_x, float initial_z,
                        float initial_heading,
                        mb_motion & output,
                        std::string & reason) {
    if (frames == 0U || normalized_local_motion.size() !=
        static_cast<std::size_t>(frames) * local_motion_width) {
        reason = "decoded local motion shape mismatch";
        return MB_INVALID_ARGUMENT;
    }
    std::vector<float> raw(normalized_local_motion.size());
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        for (std::uint32_t feature = 0; feature < local_motion_width; ++feature) {
            const std::uint32_t dual_index = feature < 4U ? 5U + feature : 9U + feature - 4U;
            raw[static_cast<std::size_t>(frame) * local_motion_width + feature] =
                unnormalize_feature(model,
                    normalized_local_motion[static_cast<std::size_t>(frame) * local_motion_width + feature],
                    dual_index);
        }
    }
    output.frames = frames;
    output.joints = g1_joint_count;
    output.root_translations.assign(static_cast<std::size_t>(frames) * 3U, 0.0F);
    output.local_rotations_xyzw.resize(static_cast<std::size_t>(frames) * g1_joint_count * 4U);
    float x = initial_x, z = initial_z;
    for (std::uint32_t frame = 0; frame < frames; ++frame) {
        const float * features = raw.data() + static_cast<std::size_t>(frame) * local_motion_width;
        float * root = output.root_translations.data() + static_cast<std::size_t>(frame) * 3U;
        root[0] = x; root[1] = features[3]; root[2] = z;
        if (frame + 1U < frames) {
            x += features[1] / fps;
            z += features[2] / fps;
        }
        std::array<mat3, g1_joint_count> global{};
        constexpr std::size_t rotation_offset = 4U + 99U;
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint)
            global[joint] = cont6d_matrix(features + rotation_offset + joint * 6U);
        for (std::uint32_t joint = 0; joint < g1_joint_count; ++joint) {
            const auto parent = model.joint_parents[joint];
            const mat3 local = parent < 0 ? global[joint]
                : multiply(transpose(global[static_cast<std::size_t>(parent)]), global[joint]);
            const quat q = matrix_quaternion(local);
            float * xyzw = output.local_rotations_xyzw.data() +
                (static_cast<std::size_t>(frame) * g1_joint_count + joint) * 4U;
            xyzw[0] = q[1]; xyzw[1] = q[2]; xyzw[2] = q[3]; xyzw[3] = q[0];
        }
    }
    (void)initial_heading; // Body rotations retain world heading in this released representation.
    return MB_OK;
}

} // namespace motionbricks::detail
