#include "handles.hpp"
#include "motion_rep.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace motionbricks::detail;
    // Analytic hierarchy: root yaw 90 degrees, endpoint parents roll 90
    // degrees, endpoint rest offsets along X. Thus offsets move along Y,
    // while parent orientations are neither local nor global identity.
    // Include two discarded history frames to test the last-four contract.
    constexpr unsigned frames = 6;
    constexpr std::array<unsigned,4> endpoints{7,14,25,33};
    constexpr std::array<unsigned,4> parents{6,13,24,32};
    mb_model model;
    model.joint_parents.assign(g1_joint_count, 0);
    model.joint_parents[0] = -1;
    model.neutral_joints.resize(g1_joint_count * 3U);
    for (unsigned i = 0; i < endpoints.size(); ++i) {
        model.joint_parents[endpoints[i]] = static_cast<int>(parents[i]);
        model.neutral_joints[parents[i] * 3U] = 2.0F;
        model.neutral_joints[endpoints[i] * 3U] = 3.0F;
    }
    std::vector<float> roots(frames * 3U), rotations(frames * g1_joint_count * 4U);
    const float s = std::sqrt(0.5F);
    for (unsigned frame = 0; frame < frames; ++frame) {
        roots[frame * 3U] = static_cast<float>(frame);
        roots[frame * 3U + 1U] = 0.8F + static_cast<float>(frame) * 0.01F;
        roots[frame * 3U + 2U] = 5.0F;
        for (unsigned joint = 0; joint < g1_joint_count; ++joint)
            rotations[(frame * g1_joint_count + joint) * 4U + 3U] = 1.0F;
        auto * root = rotations.data() + frame * g1_joint_count * 4U;
        root[1] = s; root[3] = s;
        for (unsigned i = 0; i < endpoints.size(); ++i) {
            auto * parent = root + parents[i] * 4U;
            parent[2] = s; parent[3] = s;
            auto * endpoint = root + endpoints[i] * 4U;
            const float angle = 0.2F + static_cast<float>(frame + i) * 0.1F;
            endpoint[0] = std::sin(angle); endpoint[3] = std::cos(angle);
        }
    }
    const auto original_roots = roots, original_rotations = rotations;
    encoded_frames encoded;
    std::string reason;
    if (encode_context(model, roots, rotations, frames, encoded, reason) != MB_OK) {
        std::cerr << reason << '\n'; return 1;
    }
    bool valid = roots == original_roots && rotations == original_rotations;
    const auto near = [&](float actual, float expected) {
        if (std::abs(actual - expected) <= 2e-6F) return;
        std::cerr << "expected " << expected << ", got " << actual << '\n';
        valid = false;
    };
    constexpr std::array<float,6> identity{1,0,0,0,1,0};
    constexpr std::array<float,6> parent_orientation{0,1,0,0,0,1};
    for (unsigned frame = 0; frame < boundary_frame_count; ++frame) {
        const auto * pose = encoded.poses.data() + frame * external_pose_width;
        for (unsigned i = 0; i < endpoints.size(); ++i) {
            const auto * position = pose + (endpoints[i] - 1U) * 3U;
            near(position[0], 0);
            near(position[1], roots[(frame + 2U) * 3U + 1U] + 1.0F);
            near(position[2], -2);
            for (unsigned axis = 0; axis < 6; ++axis) {
                near(pose[99U + endpoints[i] * 6U + axis], identity[axis]);
                near(pose[99U + parents[i] * 6U + axis], parent_orientation[axis]);
            }
        }
        near(encoded.global_root[frame * global_root_width], roots[(frame + 2U) * 3U]);
    }
    // Changing only virtual endpoint rotations cannot change conditioning.
    for (unsigned frame = 0; frame < frames; ++frame)
        for (auto joint : endpoints) {
            auto * q = rotations.data() + (frame * g1_joint_count + joint) * 4U;
            q[0] = 0; q[1] = s; q[2] = 0; q[3] = s;
        }
    encoded_frames changed;
    if (encode_context(model, roots, rotations, frames, changed, reason) != MB_OK) return 1;
    valid = valid && encoded.poses == changed.poses &&
        encoded.global_root == changed.global_root && encoded.local_root == changed.local_root;
    if (!valid) return 1;
    std::cout << "Endpoint conditioning: global identity, parent-derived positions, "
                 "physical rotations and caller buffers preserved\n";
    return 0;
}
