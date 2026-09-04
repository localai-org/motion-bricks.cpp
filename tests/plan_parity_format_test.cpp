#include "plan_parity.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace {
void u32(std::ofstream & out, std::uint32_t value) {
    const std::array<unsigned char,4> bytes{static_cast<unsigned char>(value),
        static_cast<unsigned char>(value>>8U),static_cast<unsigned char>(value>>16U),
        static_cast<unsigned char>(value>>24U)};
    out.write(reinterpret_cast<const char *>(bytes.data()),4);
}
void u64(std::ofstream & out, std::uint64_t value) {u32(out,static_cast<std::uint32_t>(value));u32(out,static_cast<std::uint32_t>(value>>32U));}
void f32(std::ofstream & out,float value){u32(out,std::bit_cast<std::uint32_t>(value));}
}

int main() {
    const auto path=std::filesystem::temp_directory_path()/"motionbricks-plan-parity-format-test.mbparity";
    {
        std::ofstream out(path,std::ios::binary|std::ios::trunc);
        out.write("MBPARI1\0",8); u32(out,1);u32(out,30);u32(out,1);u32(out,34);u32(out,4);u32(out,4);u32(out,0);
        for(int joint=0;joint<34;++joint)u32(out,std::bit_cast<std::uint32_t>(joint==0?-1:joint-1));
        for(int i=0;i<34*3;++i)f32(out,float(i)/100.0F);
        u32(out,17);u32(out,24);u32(out,2);u32(out,0);u64(out,1234);
        for(int i=0;i<6;++i)f32(out,float(i));
        const auto zeros=[&](std::size_t count){for(std::size_t i=0;i<count;++i)f32(out,0.0F);};
        const auto identities=[&](std::size_t count){for(std::size_t i=0;i<count;++i){f32(out,0);f32(out,0);f32(out,0);f32(out,1);}};
        zeros(4*3); identities(4*34); zeros(24*3); identities(24*34); zeros(24*34*3);
        zeros(4*3); identities(4*34); zeros(4*34*3);
    }
    try {
        const auto fixture=motionbricks::parity::load(path);
        std::filesystem::remove(path);
        const auto & plan=fixture.plans.at(0);
        if(fixture.fps!=30 || fixture.parents.size()!=34 || fixture.neutral_joints.size()!=102 ||
           plan.command_frame!=17 || plan.expected_frames!=24 || plan.mode!=2 || plan.seed!=1234 ||
           plan.expected_joint_positions.size()!=24*34*3) return 1;
    } catch(const std::exception & error) {
        std::filesystem::remove(path); std::cerr<<error.what()<<'\n'; return 1;
    }
    return 0;
}
