#include "sampling.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

int main(int argc, char ** argv) {
    using namespace motionbricks::detail;
    std::string reason;
    std::vector<std::int32_t> selected;
    if (argc == 2) {
        std::ifstream file(argv[1], std::ios::binary);
        std::array<char,8> magic{}; std::uint32_t rows=0, choices=0;
        file.read(magic.data(),8);
        file.read(reinterpret_cast<char *>(&rows),4);file.read(reinterpret_cast<char *>(&choices),4);
        if (!file || std::string(magic.data(),8)!="MBGSAMP1" || rows==0 || rows>10000 || choices!=10) return 1;
        const auto count=static_cast<std::size_t>(rows)*choices;
        std::vector<float> logits(count),uniforms(count),noise(count),scores(count);
        std::vector<std::int32_t> expected(rows);
        for (auto * values : {&logits,&uniforms,&noise,&scores})
            file.read(reinterpret_cast<char *>(values->data()),static_cast<std::streamsize>(count*4));
        file.read(reinterpret_cast<char *>(expected.data()),rows*4);
        if (!file || file.peek()!=std::char_traits<char>::eof()) return 1;
        if (!sample_pose_tokens(logits,uniforms,choices,selected,reason)) return 1;
        float noise_error=0,score_error=0;
        for(std::size_t i=0;i<count;++i) {
            noise_error=std::max(noise_error,std::abs(gumbel_noise(uniforms[i])-noise[i]));
            score_error=std::max(score_error,std::abs(logits[i]+gumbel_noise(uniforms[i])-scores[i]));
        }
        std::cout<<"stored-uniform upstream parity rows="<<rows<<" noise_max_abs="<<noise_error
                 <<" score_max_abs="<<score_error<<" exact_tokens="<<(selected==expected)<<'\n';
        return selected==expected && noise_error<2e-5F && score_error<2e-5F ? 0 : 1;
    }
    const std::array<float,3> logits{std::log(.1F),std::log(.2F),std::log(.7F)};
    std::array<std::uint32_t,3> frequencies{};
    std::uint64_t rng=1234,repeat=1234,other=5678;
    bool different=false;
    for(unsigned i=0;i<50000;++i) {
        std::array<float,3> uniforms{};
        for(float & value:uniforms) {
            value=sampling_uniform(rng);
            if(value!=sampling_uniform(repeat) || value<0 || value>=1) return 1;
            different |= value!=sampling_uniform(other);
        }
        if(!sample_pose_tokens(logits,uniforms,3,selected,reason))return 1;
        ++frequencies[static_cast<std::size_t>(selected[0])];
    }
    if(!different)return 1;
    for(unsigned i=0;i<3;++i) {
        const float probability=static_cast<float>(frequencies[i])/50000;
        if(std::abs(probability-std::exp(logits[i]))>.01F)return 1;
    }
    const std::array<float,3> ties{1,1,1},ones{1,1,1};
    if(!sample_pose_tokens(ties,{},3,selected,reason)||selected[0]!=0)return 1;
    if(!sample_pose_tokens(ties,ones,3,selected,reason)||selected[0]!=0)return 1;
    if(!std::isfinite(gumbel_noise(0)) || !std::isfinite(gumbel_noise(1)))return 1;
    const std::array<float,3> bad{0,-.1F,1};
    if(sample_pose_tokens(logits,bad,3,selected,reason)||sample_pose_tokens(logits,{},0,selected,reason))return 1;
    auto nonfinite=logits;nonfinite[0]=std::numeric_limits<float>::quiet_NaN();
    if(sample_pose_tokens(nonfinite,{},3,selected,reason))return 1;
    std::cout<<"seed reproducibility, endpoint/tie/rejection, categorical frequencies passed\n";
    return 0;
}
