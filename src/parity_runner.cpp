#include <motionbricks/motionbricks.h>

#include "agent.hpp"
#include "handles.hpp"
#include "plan_parity.hpp"
#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

constexpr std::array<std::string_view, 15> mode_names{
    "idle", "slow_walk", "walk", "hand_crawling", "walk_boxing", "elbow_crawling",
    "stealth_walk", "injured_walk", "walk_stealth", "walk_happy_dance", "walk_zombie",
    "walk_gun", "walk_scared", "walk_left", "walk_right",
};

struct options_deleter { void operator()(mb_runtime_options * v) const { mb_runtime_options_free(v); } };
struct model_deleter { void operator()(mb_model * v) const { mb_model_free(v); } };
struct style_deleter { void operator()(mb_style * v) const { mb_style_free(v); } };
struct agent_deleter { void operator()(mb_agent * v) const { mb_agent_free(v); } };
struct command_deleter { void operator()(mb_command * v) const { mb_command_free(v); } };
struct motion_deleter { void operator()(mb_motion * v) const { mb_motion_free(v); } };

using mat3 = std::array<float, 9>;
using quat = std::array<float, 4>;

mat3 quaternion_matrix(const float * xyzw) {
    const float x=xyzw[0], y=xyzw[1], z=xyzw[2], w=xyzw[3];
    const float scale = 2.0F / (x*x+y*y+z*z+w*w);
    return {1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w),
            scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w),
            scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y)};
}

mat3 multiply(const mat3 & a, const mat3 & b) {
    mat3 output{};
    for (unsigned row=0; row<3; ++row) for (unsigned column=0; column<3; ++column)
        for (unsigned k=0; k<3; ++k) output[row*3+column] += a[row*3+k]*b[k*3+column];
    return output;
}

std::array<float,3> transform(const mat3 & m, const std::array<float,3> & v) {
    return {m[0]*v[0]+m[1]*v[1]+m[2]*v[2], m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
            m[6]*v[0]+m[7]*v[1]+m[8]*v[2]};
}

std::vector<float> forward_kinematics(std::span<const float> roots,
                                      std::span<const float> rotations,
                                      std::uint32_t frames,
                                      const motionbricks::parity::fixture & fixture) {
    constexpr auto joints = motionbricks::parity::joint_count;
    std::vector<float> output(static_cast<std::size_t>(frames)*joints*3U);
    std::vector<mat3> global(joints);
    for (std::uint32_t frame=0; frame<frames; ++frame) {
        for (std::uint32_t joint=0; joint<joints; ++joint) {
            const auto local = quaternion_matrix(rotations.data() +
                (static_cast<std::size_t>(frame)*joints+joint)*4U);
            const auto parent = fixture.parents[joint];
            global[joint] = parent < 0 ? local : multiply(global[static_cast<std::size_t>(parent)], local);
            float * position = output.data() + (static_cast<std::size_t>(frame)*joints+joint)*3U;
            if (parent < 0) {
                std::copy_n(roots.data()+static_cast<std::size_t>(frame)*3U, 3U, position);
            } else {
                const auto p = static_cast<std::size_t>(parent);
                const std::array<float,3> rest{
                    fixture.neutral_joints[joint*3U]-fixture.neutral_joints[p*3U],
                    fixture.neutral_joints[joint*3U+1U]-fixture.neutral_joints[p*3U+1U],
                    fixture.neutral_joints[joint*3U+2U]-fixture.neutral_joints[p*3U+2U]};
                const auto offset = transform(global[p], rest);
                const float * parent_position = output.data()+(static_cast<std::size_t>(frame)*joints+p)*3U;
                for (unsigned axis=0; axis<3; ++axis) position[axis]=parent_position[axis]+offset[axis];
            }
        }
    }
    return output;
}

struct metric { double maximum{}; double rmse{}; };

metric vector_metric(std::span<const float> expected, std::span<const float> actual, std::size_t width) {
    if (expected.size()!=actual.size() || expected.size()%width) return {INFINITY,INFINITY};
    double square=0.0, maximum=0.0;
    for (std::size_t row=0; row<expected.size()/width; ++row) {
        double norm=0.0;
        for (std::size_t axis=0; axis<width; ++axis) {
            const double difference=double(actual[row*width+axis])-double(expected[row*width+axis]);
            norm += difference*difference;
        }
        maximum=std::max(maximum,std::sqrt(norm)); square+=norm;
    }
    return {maximum,std::sqrt(square/double(expected.size()/width))};
}

metric quaternion_metric(std::span<const float> expected, std::span<const float> actual) {
    if (expected.size()!=actual.size() || expected.size()%4U) return {INFINITY,INFINITY};
    constexpr double radians_to_degrees=57.295779513082320876;
    double square=0.0, maximum=0.0;
    for (std::size_t row=0; row<expected.size()/4U; ++row) {
        double dot=0.0, left=0.0, right=0.0;
        for (unsigned axis=0; axis<4; ++axis) {
            const double a=expected[row*4U+axis], b=actual[row*4U+axis];
            dot+=a*b; left+=a*a; right+=b*b;
        }
        const double cosine=std::clamp(std::abs(dot)/std::sqrt(left*right),0.0,1.0);
        const double angle=2.0*std::acos(cosine)*radians_to_degrees;
        maximum=std::max(maximum,angle); square+=angle*angle;
    }
    return {maximum,std::sqrt(square/double(expected.size()/4U))};
}

struct result {
    std::uint32_t index{}, command_frame{}, expected_frames{}, actual_frames{};
    std::int32_t mode{};
    std::uint64_t seed{};
    bool duration_exact{}, passed{};
    metric roots, rotations, joints, target_roots, target_rotations, target_joints;
    std::vector<float> native_roots, native_rotations, native_joints;
    std::vector<float> native_target_roots, native_target_rotations, native_target_joints;
};

bool call(mb_status status, std::string_view operation, const char * error) {
    if (status==MB_OK) return true;
    std::cerr << operation << " failed (" << mb_status_string(status) << "): " << error << '\n';
    return false;
}

void json_number(std::ostream & out, double value) {
    if (std::isfinite(value)) out << value; else out << "null";
}
void json_metric(std::ostream & out, const metric & value) {
    out << "{\"max\":"; json_number(out,value.maximum); out << ",\"rmse\":";
    json_number(out,value.rmse); out << '}';
}
void json_floats(std::ostream & out, const std::vector<float> & values) {
    out << '[';
    for (std::size_t i=0;i<values.size();++i) { if(i) out << ','; out << values[i]; }
    out << ']';
}

template<class T>
void json_values(std::ostream & out, std::span<const T> values) {
    out << '[';
    for (std::size_t i=0;i<values.size();++i) {
        if(i) out << ',';
        if constexpr (std::is_same_v<T,std::uint8_t>) out << static_cast<unsigned>(values[i]);
        else out << values[i];
    }
    out << ']';
}

template<class T, std::size_t N>
void json_values(std::ostream & out, const std::array<T,N> & values) {
    json_values(out,std::span<const T>(values));
}

template<class T>
void json_values(std::ostream & out, const std::vector<T> & values) {
    json_values(out,std::span<const T>(values));
}

bool write_neural_trace(const std::filesystem::path & path, std::uint32_t plan,
                        const motionbricks::detail::transition_trace & trace) {
    std::ofstream out(path);
    if (!out) return false;
    out << std::setprecision(9)
        << "{\n  \"format\":\"motionbricks-native-neural-trace-v1\",\n"
        << "  \"plan\":" << plan << ",\n  \"selected_tokens\":" << trace.selected_tokens
        << ",\n  \"tensors\":{\n";
    auto item=[&]<class T>(std::string_view name, const T & values, bool last=false) {
        out << "    \"" << name << "\":"; json_values(out,values);
        out << (last?"\n":",\n");
    };
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
    item("pose.logits",trace.pose_logits);
    item("pose.tokens",trace.pose_tokens);
    item("decoder.quantized",trace.decoder_quantized);
    item("decoder.external_condition",trace.decoder_external_condition);
    item("decoder.has_target_condition",trace.decoder_target_mask);
    item("decoder.output",trace.decoder_output,true);
    out << "  }\n}\n";
    return bool(out);
}

bool write_report(const std::filesystem::path & path, std::string_view device,
                  const motionbricks::parity::fixture & fixture,
                  const std::vector<result> & results) {
    std::ofstream out(path);
    if (!out) return false;
    const bool passed=std::ranges::all_of(results,[](const result & value){return value.passed;});
    const auto worst=[&](auto member) {
        double maximum=0.0; std::uint32_t plan=0;
        for(const auto & value:results) if((value.*member).maximum>maximum) {
            maximum=(value.*member).maximum;plan=value.index;
        }
        return std::pair{maximum,plan};
    };
    const auto [root_max,root_plan]=worst(&result::roots);
    const auto [rotation_max,rotation_plan]=worst(&result::rotations);
    const auto [joint_max,joint_plan]=worst(&result::joints);
    const auto [target_root_max,target_root_plan]=worst(&result::target_roots);
    const auto [target_rotation_max,target_rotation_plan]=worst(&result::target_rotations);
    const auto [target_joint_max,target_joint_plan]=worst(&result::target_joints);
    const auto durations=std::ranges::count_if(results,[](const result & value){return value.duration_exact;});
    out << std::setprecision(9)
        << "{\n  \"format\":\"motionbricks-open-loop-report-v1\",\n"
        << "  \"device\":\"" << device << "\",\n"
        << "  \"fps\":" << fixture.fps << ",\n  \"joints\":" << motionbricks::parity::joint_count
        << ",\n  \"passed\":" << (passed?"true":"false") << ",\n"
        << "  \"comparison_layer\":\"raw_neural_output_before_playback_blend\",\n"
        << "  \"pose_token_ids\":\"unavailable at observational boundary\",\n"
        << "  \"tolerances\":{\"root_m\":0.01,\"rotation_deg\":2.0,\"joint_m\":0.02,"
           "\"target_root_m\":0.002,\"target_rotation_deg\":0.2,\"target_joint_m\":0.003},\n"
        << "  \"summary\":{\"duration_matches\":" << durations << ",\"duration_total\":" << results.size()
        << ",\"worst\":{\"root_m\":{\"max\":" << root_max << ",\"plan\":" << root_plan
        << "},\"rotation_deg\":{\"max\":" << rotation_max << ",\"plan\":" << rotation_plan
        << "},\"joint_m\":{\"max\":" << joint_max << ",\"plan\":" << joint_plan
        << "},\"target_root_m\":{\"max\":" << target_root_max << ",\"plan\":" << target_root_plan
        << "},\"target_rotation_deg\":{\"max\":" << target_rotation_max << ",\"plan\":" << target_rotation_plan
        << "},\"target_joint_m\":{\"max\":" << target_joint_max << ",\"plan\":" << target_joint_plan
        << "}}},\n"
        << "  \"parents\":";
    out << '['; for(std::size_t i=0;i<fixture.parents.size();++i){if(i)out<<',';out<<fixture.parents[i];} out << "],\n";
    out << "  \"neutral_joints\":"; json_floats(out,fixture.neutral_joints); out << ",\n  \"plans\":[\n";
    for (std::size_t i=0;i<results.size();++i) {
        const auto & value=results[i]; const auto & expected=fixture.plans[i];
        out << "    {\"index\":" << value.index << ",\"command_frame\":" << value.command_frame
            << ",\"mode\":" << value.mode << ",\"style\":\"" << mode_names[static_cast<std::size_t>(value.mode)]
            << "\",\"seed\":" << value.seed << ",\"expected_frames\":" << value.expected_frames
            << ",\"actual_frames\":" << value.actual_frames << ",\"duration_exact\":"
            << (value.duration_exact?"true":"false") << ",\"passed\":" << (value.passed?"true":"false")
            << ",\"movement\":[" << expected.movement[0] << ',' << expected.movement[1] << ',' << expected.movement[2]
            << "],\"facing\":[" << expected.facing[0] << ',' << expected.facing[1] << ',' << expected.facing[2]
            << "],\"metrics\":{\"root_m\":"; json_metric(out,value.roots);
        out << ",\"rotation_deg\":"; json_metric(out,value.rotations);
        out << ",\"joint_m\":"; json_metric(out,value.joints);
        out << ",\"target_root_m\":"; json_metric(out,value.target_roots);
        out << ",\"target_rotation_deg\":"; json_metric(out,value.target_rotations);
        out << ",\"target_joint_m\":"; json_metric(out,value.target_joints);
        out << "},\"expected_roots\":"; json_floats(out,expected.expected_roots);
        out << ",\"native_roots\":"; json_floats(out,value.native_roots);
        out << ",\"native_rotations\":"; json_floats(out,value.native_rotations);
        out << ",\"expected_joint_positions\":"; json_floats(out,expected.expected_joint_positions);
        out << ",\"native_joint_positions\":"; json_floats(out,value.native_joints);
        out << ",\"native_target_roots\":"; json_floats(out,value.native_target_roots);
        out << ",\"native_target_rotations\":"; json_floats(out,value.native_target_rotations);
        out << ",\"expected_target_joint_positions\":"; json_floats(out,expected.target_joint_positions);
        out << ",\"native_target_joint_positions\":"; json_floats(out,value.native_target_joints);
        out << '}' << (i+1==results.size()?"\n":",\n");
    }
    out << "  ]\n}\n";
    return bool(out);
}

int run(const std::filesystem::path & fixture_path, const std::filesystem::path & model_path,
        const std::filesystem::path & styles_path, const std::filesystem::path & report_path,
        mb_device device, std::string_view device_name,
        const std::optional<std::pair<std::uint32_t,std::filesystem::path>> & trace_request,
        const std::optional<std::filesystem::path> & trace_directory) {
    const auto fixture=motionbricks::parity::load(fixture_path);
    std::array<char,2048> error{};
    mb_runtime_options * raw_options=nullptr; mb_model * raw_model=nullptr;
    if (!call(mb_runtime_options_create(&raw_options,error.data(),error.size()),"create options",error.data())) return 1;
    std::unique_ptr<mb_runtime_options,options_deleter> options(raw_options);
    if (!call(mb_runtime_options_set_device(options.get(),device,error.data(),error.size()),"set device",error.data()) ||
        !call(mb_model_load(model_path.string().c_str(),options.get(),&raw_model,error.data(),error.size()),"load model",error.data())) return 1;
    std::unique_ptr<mb_model,model_deleter> model(raw_model);
    std::array<std::unique_ptr<mb_style,style_deleter>,mode_names.size()> styles;
    for (const auto & item : fixture.plans) if (!styles[static_cast<std::size_t>(item.mode)]) {
        mb_style * raw=nullptr;
        const auto path=styles_path/(std::string(mode_names[static_cast<std::size_t>(item.mode)])+".mbstyle");
        if (!call(mb_style_load(model.get(),path.string().c_str(),&raw,error.data(),error.size()),"load style",error.data())) return 1;
        styles[static_cast<std::size_t>(item.mode)].reset(raw);
    }
    // The fixture embeds the exact support skeleton used by upstream conversion.
    for (std::uint32_t joint=0;joint<motionbricks::parity::joint_count;++joint) {
        std::int32_t parent=-2; float x=0,y=0,z=0;
        if (!call(mb_model_get_joint_parent(model.get(),joint,&parent,error.data(),error.size()),"get parent",error.data()) ||
            !call(mb_model_get_neutral_joint_position(model.get(),joint,&x,&y,&z,error.data(),error.size()),"get neutral joint",error.data())) return 1;
        if (parent!=fixture.parents[joint] || std::max({std::abs(x-fixture.neutral_joints[joint*3U]),
            std::abs(y-fixture.neutral_joints[joint*3U+1U]),std::abs(z-fixture.neutral_joints[joint*3U+2U])})>1e-7F) {
            std::cerr << "fixture/model skeleton mismatch at joint " << joint << '\n'; return 1;
        }
    }
    std::vector<result> results; results.reserve(fixture.plans.size());
    bool wrote_trace=false;
    for (std::size_t index=0;index<fixture.plans.size();++index) {
        const auto & expected=fixture.plans[index];
        mb_agent * raw_agent=nullptr; mb_command * raw_command=nullptr; mb_motion * raw_motion=nullptr;
        if (!call(mb_agent_create(model.get(),&raw_agent,error.data(),error.size()),"create agent",error.data())) return 1;
        std::unique_ptr<mb_agent,agent_deleter> agent(raw_agent);
        if (!call(mb_agent_set_context(agent.get(),expected.context_roots.data(),expected.context_local_rotations.data(),
                motionbricks::parity::context_frames,motionbricks::parity::joint_count,error.data(),error.size()),"set context",error.data()) ||
            !call(mb_command_create(&raw_command,error.data(),error.size()),"create command",error.data())) return 1;
        std::unique_ptr<mb_command,command_deleter> command(raw_command);
        // open-loop.mbparity stores model_features, i.e. the output before
        // upstream's separate FILTER_QPOS playback seam. Keep this gate about
        // neural inference; agent_blend_test independently gates that seam
        // against captured raw/filtered upstream qpos.
        command->skip_context_blend=1U;
        command->sampling_argmax=1U; // These fixtures explicitly captured upstream argmax.
        auto * style=styles[static_cast<std::size_t>(expected.mode)].get();
        if (!call(mb_command_set_style(command.get(),style,error.data(),error.size()),"set style",error.data()) ||
            !call(mb_command_set_movement_direction(command.get(),expected.movement[0],expected.movement[1],expected.movement[2],error.data(),error.size()),"set movement",error.data()) ||
            !call(mb_command_set_facing_direction(command.get(),expected.facing[0],expected.facing[1],expected.facing[2],error.data(),error.size()),"set facing",error.data()) ||
            !call(mb_command_set_seed(command.get(),expected.seed,error.data(),error.size()),"set seed",error.data())) return 1;
        const bool trace_this = (trace_request && trace_request->first==index) || trace_directory;
        if (trace_this) {
            motionbricks::detail::transition_trace trace;
            mb_motion traced_motion;
            std::string trace_reason;
            if (!call(motionbricks::detail::plan_agent_trace(*agent,*command,traced_motion,trace,trace_reason),
                      "trace plan",trace_reason.c_str())) return 1;
            std::filesystem::path trace_path;
            if (trace_directory) {
                std::ostringstream name;
                name << "native-plan-" << std::setfill('0') << std::setw(3) << index << ".json";
                trace_path = *trace_directory / name.str();
            } else {
                trace_path = trace_request->second;
            }
            if (!write_neural_trace(trace_path,static_cast<std::uint32_t>(index),trace)) {
                std::cerr << "could not write neural trace: " << trace_path << '\n'; return 1;
            }
            wrote_trace=true;
        }
        if (!call(mb_agent_plan(agent.get(),command.get(),&raw_motion,error.data(),error.size()),"plan",error.data())) return 1;
        std::unique_ptr<mb_motion,motion_deleter> motion(raw_motion);
        std::uint64_t frames=0,root_count=0,rotation_count=0,target_count=0,target_root_count=0,target_rotation_count=0;
        const float * roots=nullptr,*rotations=nullptr,*target_roots=nullptr,*target_rotations=nullptr;
        if (!call(mb_motion_get_frame_count(motion.get(),&frames,error.data(),error.size()),"get frames",error.data()) ||
            !call(mb_motion_get_root_translations(motion.get(),&roots,&root_count,error.data(),error.size()),"get roots",error.data()) ||
            !call(mb_motion_get_local_rotations_xyzw(motion.get(),&rotations,&rotation_count,error.data(),error.size()),"get rotations",error.data()) ||
            !call(mb_motion_get_target_frame_count(motion.get(),&target_count,error.data(),error.size()),"get target frames",error.data()) ||
            !call(mb_motion_get_target_root_translations(motion.get(),&target_roots,&target_root_count,error.data(),error.size()),"get target roots",error.data()) ||
            !call(mb_motion_get_target_local_rotations_xyzw(motion.get(),&target_rotations,&target_rotation_count,error.data(),error.size()),"get target rotations",error.data())) return 1;
        result value{static_cast<std::uint32_t>(index),expected.command_frame,expected.expected_frames,
                     static_cast<std::uint32_t>(frames),expected.mode,expected.seed};
        value.duration_exact=frames==expected.expected_frames;
        value.native_roots.assign(roots,roots+root_count);
        value.native_rotations.assign(rotations,rotations+rotation_count);
        value.native_target_roots.assign(target_roots,target_roots+target_root_count);
        value.native_target_rotations.assign(target_rotations,target_rotations+target_rotation_count);
        value.native_joints=forward_kinematics(value.native_roots,value.native_rotations,static_cast<std::uint32_t>(frames),fixture);
        value.native_target_joints=forward_kinematics(value.native_target_roots,value.native_target_rotations,
                                                       static_cast<std::uint32_t>(target_count),fixture);
        value.roots=vector_metric(expected.expected_roots,value.native_roots,3U);
        value.rotations=quaternion_metric(expected.expected_local_rotations,value.native_rotations);
        value.joints=vector_metric(expected.expected_joint_positions,value.native_joints,3U);
        value.target_roots=vector_metric(expected.target_roots,value.native_target_roots,3U);
        value.target_rotations=quaternion_metric(expected.target_local_rotations,value.native_target_rotations);
        value.target_joints=vector_metric(expected.target_joint_positions,value.native_target_joints,3U);
        value.passed=value.duration_exact && value.roots.maximum<=0.01 && value.rotations.maximum<=2.0 &&
            value.joints.maximum<=0.02 && value.target_roots.maximum<=0.002 &&
            value.target_rotations.maximum<=0.2 && value.target_joints.maximum<=0.003;
        std::cout << "plan " << index << " style=" << mode_names[static_cast<std::size_t>(expected.mode)]
                  << " frames=" << frames << '/' << expected.expected_frames
                  << " root=" << value.roots.maximum << "m joint=" << value.joints.maximum
                  << "m rotation=" << value.rotations.maximum << "deg target=" << value.target_joints.maximum << "m\n";
        results.push_back(std::move(value));
    }
    if (!write_report(report_path,device_name,fixture,results)) {
        std::cerr << "could not write report: " << report_path << '\n'; return 1;
    }
    if (trace_request && !wrote_trace) {
        std::cerr << "trace plan index is outside the fixture\n"; return 1;
    }
    const bool passed=std::ranges::all_of(results,[](const result & value){return value.passed;});
    std::cout << "open-loop parity " << (passed?"passed":"failed") << ": " << report_path << '\n';
    return passed?0:1;
}

} // namespace

int main(int argc,char ** argv) {
    if (argc<6 || (std::string_view(argv[5])!="cpu" && std::string_view(argv[5])!="vulkan")) {
        std::cerr << "usage: motionbricks-parity FIXTURE MODEL_DIRECTORY STYLE_DIRECTORY REPORT.json cpu|vulkan [--report-only] [--trace-plan INDEX TRACE.json | --trace-directory DIRECTORY]\n";
        return 2;
    }
    try {
        bool report_only=false;
        std::optional<std::pair<std::uint32_t,std::filesystem::path>> trace_request;
        std::optional<std::filesystem::path> trace_directory;
        for (int index=6;index<argc;) {
            const std::string_view option=argv[index];
            if (option=="--report-only") { report_only=true; ++index; continue; }
            if (option=="--trace-plan" && index+2<argc) {
                const auto parsed=std::stoul(argv[index+1]);
                if (parsed>std::numeric_limits<std::uint32_t>::max()) throw std::out_of_range("trace plan");
                trace_request=std::pair{static_cast<std::uint32_t>(parsed),std::filesystem::path(argv[index+2])};
                index+=3; continue;
            }
            if (option=="--trace-directory" && index+1<argc) {
                trace_directory=std::filesystem::path(argv[index+1]);
                std::filesystem::create_directories(*trace_directory);
                index+=2; continue;
            }
            throw std::invalid_argument("invalid parity runner option");
        }
        const std::string_view device=argv[5];
        if (trace_request && trace_directory)
            throw std::invalid_argument("--trace-plan and --trace-directory are mutually exclusive");
        const int status=run(argv[1],argv[2],argv[3],argv[4],
            device=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,device,trace_request,trace_directory);
        return report_only && status==1 ? 0 : status;
    } catch (const std::exception & error) {
        std::cerr << "open-loop parity failed: " << error.what() << '\n'; return 1;
    }
}
