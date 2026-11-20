#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include "soma_rsqrt_table.hpp"

namespace motionbricks::detail {
// CPU diagnostic of the pinned ORT reduction/output order. The default uses
// a double-derived reciprocal square root; an explicit calibration can emulate
// the measured CUDA instruction without invoking CUDA during inference.
struct soma_norm_statistics { float mean=0, variance=0; };
inline soma_norm_statistics soma_welford_statistics(const float * input,std::size_t width) {
    if(width!=512 && width!=1024)throw std::invalid_argument("unsupported SOMA norm width");
    std::array<soma_norm_statistics,256> states{};
    const auto threads=width/4;
    for(std::size_t thread=0;thread<threads;++thread) {
        auto & state=states[thread];
        for(std::size_t j=0;j<4;++j) {
            const float value=input[4*thread+j],delta=value-state.mean;
            state.mean+=delta/float(j+1);
            state.variance=std::fma(delta,value-state.mean,state.variance);
        }
    }
    const auto combine=[](soma_norm_statistics a,soma_norm_statistics b,float count) {
        const float delta=b.mean-a.mean;
        return soma_norm_statistics{std::fma(0.5F,a.mean,0.5F*b.mean),
            std::fma((delta*delta)*0.25F,2*count,a.variance+b.variance)};
    };
    float count=4;
    for(std::size_t stride=16;stride;stride/=2,count*=2)
        for(std::size_t warp=0;warp<threads;warp+=32)
            for(std::size_t lane=0;lane<stride;++lane)
                states[warp+lane]=combine(states[warp+lane],states[warp+lane+stride],count);
    for(std::size_t stride=threads/64;stride;stride/=2,count*=2)
        for(std::size_t warp=0;warp<stride;++warp)
            states[warp*32]=combine(states[warp*32],states[(warp+stride)*32],count);
    return {states[0].mean,states[0].variance/float(width)};
}
inline void soma_welford_normalize(const float * input,const float * scale,const float * bias,
                                   float * output,std::size_t width,float epsilon) {
    const auto statistics=soma_welford_statistics(input,width);
    const float inverse=soma_rsqrt_corrections.empty()?
        static_cast<float>(1.0/std::sqrt(double(statistics.variance+epsilon))):
        soma_calibrated_rsqrt(statistics.variance+epsilon);
    for(std::size_t i=0;i<width;++i) {
        const float factor=scale[i]*inverse;
        output[i]=std::fma(factor,input[i]-statistics.mean,bias[i]);
    }
}
}
