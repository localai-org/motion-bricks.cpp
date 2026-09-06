#include "agent.hpp"
#include "handles.hpp"
#include "motion_rep.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
std::array<float,4> z_rotation(float angle) {
    return {0.0F,0.0F,std::sin(angle/2.0F),std::cos(angle/2.0F)};
}

float angle(const float * quaternion) {
    return 2.0F*std::atan2(quaternion[2],quaternion[3]);
}

int recorded_parity(const char * path) {
    std::ifstream input(path,std::ios::binary);
    std::array<char,8> magic{};
    std::uint32_t records=0,joints=0;
    input.read(magic.data(),magic.size());
    input.read(reinterpret_cast<char *>(&records),sizeof records);
    input.read(reinterpret_cast<char *>(&joints),sizeof joints);
    if(!input || std::string_view(magic.data(),magic.size())!="MBBLEND1" ||
       records==0U || records>1000U || joints!=34U)return 2;
    constexpr std::size_t roots=4U*3U,rotations=4U*34U*4U;
    double maximum_root=0.0,maximum_angle=0.0,maximum_component=0.0;
    std::uint32_t worst_record=0,worst_frame=0,worst_joint=0;
    for(std::uint32_t record=0;record<records;++record) {
        std::vector<float> values((roots+rotations)*3U);
        input.read(reinterpret_cast<char *>(values.data()),
                   static_cast<std::streamsize>(values.size()*sizeof(float)));
        if(!input)return 2;
        mb_agent agent;
        agent.context_frames=4U;
        agent.context_root_xyz.assign(values.begin(),values.begin()+roots);
        agent.context_local_rotations_xyzw.assign(values.begin()+roots,
                                                   values.begin()+roots+rotations);
        const auto raw=roots+rotations;
        const auto expected=raw+roots+rotations;
        mb_motion motion;
        motion.frames=4U;motion.joints=34U;
        motion.root_translations.assign(values.begin()+raw,values.begin()+raw+roots);
        motion.local_rotations_xyzw.assign(values.begin()+raw+roots,values.begin()+expected);
        motionbricks::detail::apply_context_blend(agent,motion);
        for(std::size_t index=0;index<roots;++index)
            maximum_root=std::max(maximum_root,
                std::abs(double(motion.root_translations[index])-double(values[expected+index])));
        for(std::size_t index=0;index<rotations;index+=4U) {
            const auto joint=static_cast<std::uint32_t>((index/4U)%34U);
            if(joint==7U || joint==14U || joint==25U || joint==33U)continue;
            double dot=0,left=0,right=0;
            for(std::size_t axis=0;axis<4U;++axis) {
                const double a=motion.local_rotations_xyzw[index+axis];
                const double b=values[expected+roots+index+axis];
                dot+=a*b;left+=a*a;right+=b*b;
            }
            const double sign=dot<0.0?-1.0:1.0;
            for(std::size_t axis=0;axis<4U;++axis)
                maximum_component=std::max(maximum_component,std::abs(
                    double(motion.local_rotations_xyzw[index+axis])-sign*double(values[expected+roots+index+axis])));
            const double cosine=std::clamp(std::abs(dot)/std::sqrt(left*right),0.0,1.0);
            const double error=2.0*std::acos(cosine)*57.29577951308232;
            if(error>maximum_angle) {
                maximum_angle=error;worst_record=record;
                worst_frame=static_cast<std::uint32_t>(index/(34U*4U));
                worst_joint=static_cast<std::uint32_t>((index/4U)%34U);
            }
        }
    }
    char trailing{};
    if(input.read(&trailing,1))return 2;
    std::cout << "recorded upstream blend records=" << records
              << " root_max_abs=" << maximum_root
              << " rotation_component_max_abs=" << maximum_component
              << " rotation_max_degrees=" << maximum_angle
              << " worst=" << worst_record << ':' << worst_frame << ':' << worst_joint << '\n';
    return maximum_root<=2.0e-6 && maximum_component<=2.0e-6 ? 0 : 1;
}
}

int main(int argc,char ** argv) {
    if(argc==2)return recorded_parity(argv[1]);
    if(argc!=1)return 2;
    mb_agent agent;
    agent.context_frames=6U;
    agent.context_root_xyz.resize(6U*3U);
    agent.context_local_rotations_xyzw.resize(6U*motionbricks::detail::g1_joint_count*4U);
    mb_motion motion;
    motion.frames=8U;
    motion.joints=motionbricks::detail::g1_joint_count;
    motion.root_translations.resize(motion.frames*3U);
    motion.local_rotations_xyzw.resize(motion.frames*motion.joints*4U);
    for(std::uint32_t frame=0;frame<agent.context_frames;++frame) {
        for(std::uint32_t axis=0;axis<3U;++axis)
            agent.context_root_xyz[frame*3U+axis]=10.0F*frame+axis;
        for(std::uint32_t joint=0;joint<motion.joints;++joint) {
            const auto value=z_rotation(0.01F*static_cast<float>(frame+joint));
            std::copy(value.begin(),value.end(),agent.context_local_rotations_xyzw.begin()+
                static_cast<std::ptrdiff_t>((frame*motion.joints+joint)*4U));
        }
    }
    for(std::uint32_t frame=0;frame<motion.frames;++frame) {
        for(std::uint32_t axis=0;axis<3U;++axis)
            motion.root_translations[frame*3U+axis]=100.0F+10.0F*frame+axis;
        for(std::uint32_t joint=0;joint<motion.joints;++joint) {
            const auto value=z_rotation(0.8F+0.01F*static_cast<float>(frame+joint));
            std::copy(value.begin(),value.end(),motion.local_rotations_xyzw.begin()+
                static_cast<std::ptrdiff_t>((frame*motion.joints+joint)*4U));
        }
    }
    const auto generated_roots=motion.root_translations;
    const auto generated_rotations=motion.local_rotations_xyzw;
    motionbricks::detail::apply_context_blend(agent,motion);
    constexpr std::array<float,4> weights{0.3F,0.433333337F,0.566666663F,0.7F};
    for(std::uint32_t frame=0;frame<4U;++frame) {
        for(std::uint32_t axis=0;axis<3U;++axis) {
            const float context=agent.context_root_xyz[((agent.context_frames-4U)+frame)*3U+axis];
            const float expected=context*(1.0F-weights[frame])+generated_roots[frame*3U+axis]*weights[frame];
            assert(std::abs(motion.root_translations[frame*3U+axis]-expected)<1.0e-6F);
        }
        for(std::uint32_t joint=0;joint<motion.joints;++joint) {
            const float * actual=motion.local_rotations_xyzw.data()+(frame*motion.joints+joint)*4U;
            const float * raw=generated_rotations.data()+(frame*motion.joints+joint)*4U;
            const bool filtered=joint!=0U && joint!=7U && joint!=14U && joint!=25U && joint!=33U;
            if(!filtered) {
                for(std::uint32_t axis=0;axis<4U;++axis)assert(actual[axis]==raw[axis]);
            } else {
                const float * context=agent.context_local_rotations_xyzw.data()+
                    (((agent.context_frames-4U)+frame)*motion.joints+joint)*4U;
                const float expected=angle(context)*(1.0F-weights[frame])+angle(raw)*weights[frame];
                assert(std::abs(angle(actual)-expected)<2.0e-6F);
            }
        }
    }
    for(std::uint32_t frame=4U;frame<motion.frames;++frame) {
        for(std::uint32_t axis=0;axis<3U;++axis)
            assert(motion.root_translations[frame*3U+axis]==generated_roots[frame*3U+axis]);
    }
}
