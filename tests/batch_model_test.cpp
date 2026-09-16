#include <motionbricks/inference.h>
#include <motionbricks/motionbricks.h>

#include "agent.hpp"
#include "handles.hpp"
#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
template <typename T, void (*Free)(T *)>
using handle = std::unique_ptr<T, decltype(Free)>;

bool same(const mb_motion & left, const mb_motion & right, float tolerance = 1.0e-5F) {
    if (left.frames != right.frames || left.joints != right.joints ||
        left.root_translations.size() != right.root_translations.size() ||
        left.local_rotations_xyzw.size() != right.local_rotations_xyzw.size()) return false;
    const auto close = [=](float a, float b) { return std::abs(a - b) <= tolerance; };
    return std::equal(left.root_translations.begin(), left.root_translations.end(),
                      right.root_translations.begin(), close) &&
           std::equal(left.local_rotations_xyzw.begin(), left.local_rotations_xyzw.end(),
                      right.local_rotations_xyzw.begin(), close);
}
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    std::array<char,1024> error{};
    const auto ok = [&](mb_status status, const char * operation) {
        if (status == MB_OK) return true;
        std::cerr << operation << ": " << error.data() << '\n'; return false;
    };
    mb_runtime_options * raw_options=nullptr; mb_model * raw_model=nullptr;
    mb_style * raw_style=nullptr;
    if (!ok(mb_runtime_options_create(&raw_options,error.data(),error.size()),"options")) return 1;
    handle<mb_runtime_options,mb_runtime_options_free> options(raw_options,mb_runtime_options_free);
    if (!ok(mb_runtime_options_set_device(options.get(),MB_DEVICE_CPU,error.data(),error.size()),"device") ||
        !ok(mb_runtime_options_set_threads(options.get(),1,error.data(),error.size()),"threads") ||
        !ok(mb_model_load(argv[1],options.get(),&raw_model,error.data(),error.size()),"model")) return 1;
    handle<mb_model,mb_model_free> model(raw_model,mb_model_free);
    if (!ok(mb_style_load(model.get(),argv[2],&raw_style,error.data(),error.size()),"style")) return 1;
    handle<mb_style,mb_style_free> style(raw_style,mb_style_free);

    constexpr std::size_t count=4;
    std::vector<handle<mb_agent,mb_agent_free>> agents;
    std::vector<handle<mb_command,mb_command_free>> commands;
    std::array<mb_agent *,count> agent_ptrs{};
    std::array<const mb_command *,count> command_ptrs{};
    for (std::size_t index=0;index<count;++index) {
        mb_agent * agent=nullptr;mb_command * command=nullptr;
        if (!ok(mb_agent_create(model.get(),&agent,error.data(),error.size()),"agent") ||
            !ok(mb_agent_reset(agent,style.get(),error.data(),error.size()),"reset") ||
            !ok(mb_command_create(&command,error.data(),error.size()),"command")) return 1;
        agents.emplace_back(agent,mb_agent_free);commands.emplace_back(command,mb_command_free);
        if (!ok(mb_command_set_style(command,style.get(),error.data(),error.size()),"set style") ||
            !ok(mb_command_set_movement_direction(command,0,0,1,error.data(),error.size()),"movement") ||
            !ok(mb_command_set_facing_direction(command,0,0,1,error.data(),error.size()),"facing") ||
            !ok(mb_command_set_seed(command,1009U+index,error.data(),error.size()),"seed")) return 1;
        agent_ptrs[index]=agent;command_ptrs[index]=command;
    }
    std::array<handle<mb_motion,mb_motion_free>,count> serial{
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free)};
    for(std::size_t index=0;index<count;++index) {
        mb_motion * value=nullptr;
        if(!ok(mb_agent_plan(agent_ptrs[index],command_ptrs[index],&value,error.data(),error.size()),"serial agent plan"))return 1;
        serial[index].reset(value);
    }
    std::array<mb_motion *,count> raw_batch{};
    if(!ok(mb_agent_plan_batch(agent_ptrs.data(),command_ptrs.data(),count,raw_batch.data(),
                               error.data(),error.size()),"agent batch"))return 1;
    std::array<handle<mb_motion,mb_motion_free>,count> batch{
        handle<mb_motion,mb_motion_free>(raw_batch[0],mb_motion_free),
        handle<mb_motion,mb_motion_free>(raw_batch[1],mb_motion_free),
        handle<mb_motion,mb_motion_free>(raw_batch[2],mb_motion_free),
        handle<mb_motion,mb_motion_free>(raw_batch[3],mb_motion_free)};
    for(std::size_t index=0;index<count;++index)if(!same(*serial[index],*batch[index])) {
        std::cerr<<"agent batch lane "<<index<<" differs\n";return 1;
    }
    std::array<mb_motion,count> native_values{};
    std::array<mb_motion *,count> native_outputs{
        &native_values[0],&native_values[1],&native_values[2],&native_values[3]};
    std::string native_reason;
    if(motionbricks::detail::plan_agent_batch(agent_ptrs,command_ptrs,native_outputs,
                                               native_reason,true)!=MB_OK) {
        std::cerr<<"native agent batch: "<<native_reason<<'\n';return 1;
    }
    for(std::size_t index=0;index<count;++index)if(!same(*serial[index],native_values[index])) {
        std::cerr<<"native agent batch lane "<<index<<" differs\n";return 1;
    }
    std::array<mb_agent *,2> duplicate_agents{agent_ptrs[0],agent_ptrs[0]};
    std::array<const mb_command *,2> duplicate_commands{command_ptrs[0],command_ptrs[1]};
    std::array<mb_motion *,2> rejected{reinterpret_cast<mb_motion *>(1),reinterpret_cast<mb_motion *>(1)};
    if(mb_agent_plan_batch(duplicate_agents.data(),duplicate_commands.data(),2,rejected.data(),
                           error.data(),error.size())!=MB_INVALID_ARGUMENT||rejected[0]||rejected[1]) {
        std::cerr<<"duplicate agent batch was not rejected atomically\n";return 1;
    }

    motionbricks::detail::transition_trace trace;mb_motion ignored;std::string reason;
    if(motionbricks::detail::plan_agent_trace(*agent_ptrs[0],*command_ptrs[0],ignored,trace,reason)!=MB_OK) {
        std::cerr<<"trace: "<<reason<<'\n';return 1;
    }
    constexpr std::array<std::uint32_t,count> durations{6,8,11,16};
    std::array<motionbricks::detail::transition_constraints,count> mixed_constraints{};
    std::array<mb_motion,count> mixed_serial{},mixed_native{};
    std::array<mb_motion *,count> mixed_native_outputs{
        &mixed_native[0],&mixed_native[1],&mixed_native[2],&mixed_native[3]};
    std::array<std::uint64_t,count> mixed_seeds{2003U,2004U,2005U,2006U};
    std::array<std::uint8_t,count> mixed_argmax{};
    std::array<std::uint32_t,count> selected{};
    for(std::size_t index=0;index<count;++index) {
        mixed_constraints[index]=trace.constraints;
        mixed_constraints[index].allowed_tokens.fill(0);
        mixed_constraints[index].allowed_tokens[durations[index]-6U]=1;
        if(motionbricks::detail::run_transition(*model,mixed_constraints[index],
            mixed_serial[index],nullptr,nullptr,reason,mixed_seeds[index],false)!=MB_OK) {
            std::cerr<<"mixed serial transition: "<<reason<<'\n';return 1;
        }
    }
    if(motionbricks::detail::run_transition_batch(*model,mixed_constraints,
        mixed_native_outputs,mixed_seeds,mixed_argmax,selected,reason,true)!=MB_OK) {
        std::cerr<<"mixed native transition: "<<reason<<'\n';return 1;
    }
    for(std::size_t index=0;index<count;++index) {
        if(selected[index]!=durations[index]||!same(mixed_serial[index],mixed_native[index])) {
            std::cerr<<"mixed native duration lane "<<index<<" differs\n";return 1;
        }
    }
    std::vector<handle<mb_inference_request,mb_inference_request_free>> requests;
    std::array<const mb_inference_request *,count> request_ptrs{};
    for(std::size_t index=0;index<count;++index) {
        mb_inference_request * request=nullptr;
        if(!ok(mb_inference_request_create(&request,error.data(),error.size()),"request"))return 1;
        requests.emplace_back(request,mb_inference_request_free);request_ptrs[index]=request;
        const std::array<std::pair<std::uint32_t,std::span<const float>>,3> fields{{
            {MB_INFERENCE_GLOBAL_ROOT,trace.constraints.global_root},
            {MB_INFERENCE_LOCAL_ROOT,trace.constraints.local_root},
            {MB_INFERENCE_POSE,trace.constraints.poses}}};
        for(const auto & [field,values]:fields)
            if(!ok(mb_inference_request_set_features(request,field,values.data(),values.size(),
                                                     error.data(),error.size()),"features"))return 1;
        const std::array<std::pair<std::uint32_t,std::span<const std::uint8_t>>,3> masks{{
            {MB_INFERENCE_GLOBAL_ROOT,trace.constraints.has_global_root},
            {MB_INFERENCE_LOCAL_ROOT,trace.constraints.has_local_root},
            {MB_INFERENCE_POSE,trace.constraints.has_poses}}};
        for(const auto & [field,values]:masks) {
            std::vector<std::uint32_t> converted(values.begin(),values.end());
            if(!ok(mb_inference_request_set_mask(request,field,converted.data(),converted.size(),
                                                 error.data(),error.size()),"mask"))return 1;
        }
        std::array<std::uint32_t,11> duration_mask{};duration_mask[durations[index]-6U]=1;
        if(!ok(mb_inference_request_set_mask(request,MB_INFERENCE_DURATIONS,duration_mask.data(),
                                             duration_mask.size(),error.data(),error.size()),"duration") ||
           !ok(mb_inference_request_set_seed(request,2003U+index,error.data(),error.size()),"request seed"))return 1;
    }
    std::array<handle<mb_motion,mb_motion_free>,count> stateless_serial{
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free),
        handle<mb_motion,mb_motion_free>(nullptr,mb_motion_free)};
    for(std::size_t index=0;index<count;++index) {
        mb_motion * value=nullptr;
        if(!ok(mb_model_infer(model.get(),request_ptrs[index],&value,error.data(),error.size()),"serial inference"))return 1;
        stateless_serial[index].reset(value);
    }
    std::array<mb_motion *,count> stateless_raw{};
    if(!ok(mb_model_infer_batch(model.get(),request_ptrs.data(),count,stateless_raw.data(),
                                error.data(),error.size()),"inference batch"))return 1;
    for(std::size_t index=0;index<count;++index) {
        handle<mb_motion,mb_motion_free> value(stateless_raw[index],mb_motion_free);
        if(!same(*stateless_serial[index],*value)) {
            std::cerr<<"stateless batch lane "<<index<<" differs\n";return 1;
        }
    }
    std::cout<<"batch parity passed for four agents and mixed durations\n";
    return 0;
}
