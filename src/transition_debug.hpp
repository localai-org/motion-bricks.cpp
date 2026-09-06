#pragma once

// Opt-in diagnostic capture for the actual agent path, including external
// context and post-model transforms. No inference tensors are changed.
#include "handles.hpp"
#include "planner.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string_view>

namespace motionbricks::detail {
inline void write_transition_debug(const char * directory, const mb_agent & agent,
                                  const mb_agent & canonical,
                                  const transition_trace & trace,
                                  const std::vector<float> & decoded_roots,
                                  const std::vector<float> & world_roots,
                                  const mb_motion & output) {
    static std::atomic<unsigned long long> sequence{0};
    const auto id=sequence.fetch_add(1);
    try {
        const auto path=std::filesystem::path(directory)/("transition-"+std::to_string(id)+".json");
        // The caller creates the diagnostic directory explicitly.
        std::ofstream out(path);
        if (!out) { std::fprintf(stderr,"Cannot write transition trace: %s\n",path.string().c_str());return; }
        out << std::setprecision(9) << "{\"format\":\"motionbricks-transition-debug-v1\",\"selected_tokens\":"
            << trace.selected_tokens << ",\"tensors\":{";
        bool first=true;
        auto item=[&](std::string_view name,const auto & values) {
            if (!first) out << ',';
            first=false;out << '\"' << name << "\":[";
            bool first_value=true;
            for (auto value:values) {
                if (!first_value) out << ',';
                first_value=false;
                if (std::isfinite(static_cast<double>(value))) out << +value;
                else out << "null";
            }
            out << ']';
        };
        item("stage_ms.root_pose_vq_decode",trace.stage_ms);
        item("stats.mean",agent.model->motion_mean);item("stats.std",agent.model->motion_std);
        item("context.roots",agent.context_root_xyz);
        item("context.rotations",agent.context_local_rotations_xyzw);
        item("canonical.roots",canonical.context_root_xyz);
        item("canonical.rotations",canonical.context_local_rotations_xyzw);
        item("input.global_root_values",trace.constraints.global_root);
        item("input.local_root_values",trace.constraints.local_root);
        item("input.local_poses",trace.constraints.poses);
        item("input.has_global_root_values",trace.constraints.has_global_root);
        item("input.has_local_root_values",trace.constraints.has_local_root);
        item("input.has_local_poses",trace.constraints.has_poses);
        item("input.allowed_pred_num_tokens",trace.constraints.allowed_tokens);
        item("root.global_root_values",trace.normalized_global_root);
        item("root.local_root_values",trace.normalized_local_root);
        item("root.poses",trace.normalized_poses);
        item("root.num_token_logits",trace.duration_logits);
        item("root.pred_global_root_values",trace.predicted_global_root);
        item("root.pred_local_root_values",trace.predicted_local_root);
        item("pose.root_condition",trace.pose_root_condition);
        item("pose.pose_condition",trace.pose_condition);
        item("pose.has_pose_condition",trace.has_pose_condition);
        item("pose.logits",trace.pose_logits);item("pose.tokens",trace.pose_tokens);
        item("decoder.quantized",trace.decoder_quantized);
        item("decoder.external_condition",trace.decoder_external_condition);
        item("decoder.has_target_condition",trace.decoder_target_mask);
        item("decoder.output",trace.decoder_output);
        item("decoded.roots",decoded_roots);item("world.roots",world_roots);
        item("blended.roots",output.root_translations);
        item("target.roots",output.target_root_translations);
        out << "}}\n";
        out.flush();
        if (!out) throw std::runtime_error("failed to finish diagnostic file");
        std::fprintf(stderr,"Transition trace: %s\n",path.string().c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr,"Transition trace failed: %s\n",error.what());
    }
}
}
