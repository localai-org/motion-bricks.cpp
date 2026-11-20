#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace motionbricks::detail {
// Diagnostic model validated against pinned CUDA operands, not a general GPU
// emulator. Inputs must be finite, exactly half-representable FP32 values.
// See reference/audit_soma_matmul.py and https://arxiv.org/abs/2512.07004.
inline float soma_aligned_dot(const float * a,const float * b,std::size_t count,bool residue_first=true,float initial=0) {
    const auto exponent=[](float x) {
        // Every nonzero half value is normal after conversion to FP32.
        return int((std::bit_cast<std::uint32_t>(x)>>23)&255U)-127;
    };
    float accumulator=initial;
    for(std::size_t offset=0;offset<count;) {
        // The pinned narrow projection processes its short K remainder first.
        const auto length=offset==0&&residue_first&&count%16?count%16:
            std::min<std::size_t>(16,count-offset);
        std::array<double,16> products{};
        int largest=accumulator==0?-126:exponent(accumulator);
        for(std::size_t j=0;j<length;++j) {
            const float left=a[offset+j],right=b[offset+j];
            products[j]=double(left)*double(right);
            if(products[j]!=0)largest=std::max(largest,exponent(left)+exponent(right));
        }
        // The product is not normalized before selecting the alignment scale.
        const double scale=std::ldexp(1.0,25-largest);
        double aligned=std::trunc(double(accumulator)*scale);
        for(std::size_t j=0;j<length;++j)aligned+=std::trunc(products[j]*scale);
        const double exact=aligned/scale;
        accumulator=static_cast<float>(exact);
        if(std::abs(double(accumulator))>std::abs(exact))
            accumulator=std::bit_cast<float>(std::bit_cast<std::uint32_t>(accumulator)-1U);
        offset+=length;
    }
    return accumulator;
}
inline float soma_nvjet_dot(const float * a,const float * b,std::size_t count,std::size_t accumulators) {
    if((accumulators!=3 && accumulators!=8) || (count%128!=0 && !(accumulators==8 && count==64)))
        throw std::invalid_argument("unsupported NVJET reduction shape");
    // Observed sm_120 QKV/output kernels: three/eight persistent HMMA
    // accumulators; block assignment restarts at each 128-wide K tile.
    std::array<float,8> partial{};
    for(std::size_t offset=0;offset<count;offset+=16) {
        auto & accumulator=partial[((offset%128)/16)%accumulators];
        accumulator=soma_aligned_dot(a+offset,b+offset,std::min<std::size_t>(16,count-offset),false,accumulator);
    }
    float result=partial[0];
    for(std::size_t i=1;i<accumulators;++i)result+=partial[i];
    return result;
}
inline float soma_nvjet_split4_dot(const float * a,const float * b,std::size_t count) {
    if(count%512!=0)throw std::invalid_argument("unsupported NVJET split reduction shape");
    std::array<std::array<float,3>,4> partial{};
    for(std::size_t offset=0;offset<count;offset+=16) {
        auto & accumulator=partial[(offset/128)%4][((offset%128)/16)%3];
        accumulator=soma_aligned_dot(a+offset,b+offset,16,false,accumulator);
    }
    float result=0;
    for(const auto & group:partial)result+=(group[0]+group[1])+group[2];
    return result;
}
}
