#include <motionbricks/inference.h>
#include "error.hpp"
#include "handles.hpp"
#include "planner.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <type_traits>

struct mb_inference_request {
    motionbricks::detail::transition_constraints constraints;
    std::array<std::array<bool,2>,3> initialized{};
    uint64_t seed=0;
    uint32_t argmax=0;
};
namespace {
using motionbricks::detail::guard;
using motionbricks::detail::fail;
template<class Request> auto features(Request & r, uint32_t field) {
    using T=std::conditional_t<std::is_const_v<Request>,const float,float>;
    switch(field) {
        case MB_INFERENCE_GLOBAL_ROOT:return std::span<T>(r.constraints.global_root);
        case MB_INFERENCE_LOCAL_ROOT:return std::span<T>(r.constraints.local_root);
        case MB_INFERENCE_POSE:return std::span<T>(r.constraints.poses);
        default:return std::span<T>{};
    }
}
template<class Request> auto mask(Request & r, uint32_t field) {
    using T=std::conditional_t<std::is_const_v<Request>,const uint8_t,uint8_t>;
    switch(field) {
        case MB_INFERENCE_GLOBAL_ROOT:return std::span<T>(r.constraints.has_global_root);
        case MB_INFERENCE_LOCAL_ROOT:return std::span<T>(r.constraints.has_local_root);
        case MB_INFERENCE_POSE:return std::span<T>(r.constraints.has_poses);
        case MB_INFERENCE_DURATIONS:return std::span<T>(r.constraints.allowed_tokens);
        default:return std::span<T>{};
    }
}
bool safe(std::span<const float> data) {
    return std::all_of(data.begin(),data.end(),[](float v){return std::isfinite(v)&&std::abs(v)<=1e4F;});
}
}
extern "C" {
mb_status mb_inference_request_create(mb_inference_request ** out,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!out)return fail(MB_INVALID_ARGUMENT,e,n,"output is null");
        *out=nullptr;*out=new mb_inference_request;return MB_OK;
    });
}
void mb_inference_request_free(mb_inference_request * r) {delete r;}
mb_status mb_inference_request_set_features(mb_inference_request * r,uint32_t f,const float * data,uint64_t count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!data)return fail(MB_INVALID_ARGUMENT,e,n,"request or features are null");
        auto dest=features(*r,f);
        if(dest.empty()||count!=dest.size())return fail(MB_INVALID_ARGUMENT,e,n,"feature field/count mismatch");
        if(!safe({data,dest.size()}))return fail(MB_INVALID_ARGUMENT,e,n,"features must be finite and within +/-1e4");
        std::copy_n(data,dest.size(),dest.begin());r->initialized[f]={true,true};return MB_OK;
    });
}
mb_status mb_inference_request_get_features(const mb_inference_request * r,uint32_t f,float * data,uint64_t capacity,uint64_t * count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!count)return fail(MB_INVALID_ARGUMENT,e,n,"request or count is null");
        *count=0;auto source=features(*r,f);
        if(source.empty())return fail(MB_INVALID_ARGUMENT,e,n,"invalid feature field");
        *count=source.size();if(!data&&capacity==0)return MB_OK;
        if(!data||capacity<source.size())return fail(MB_INVALID_ARGUMENT,e,n,"feature buffer too small");
        std::copy(source.begin(),source.end(),data);return MB_OK;
    });
}
mb_status mb_inference_request_set_mask(mb_inference_request * r,uint32_t f,const uint32_t * data,uint64_t count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!data)return fail(MB_INVALID_ARGUMENT,e,n,"request or mask is null");
        auto dest=mask(*r,f);
        if(dest.empty()||count!=dest.size())return fail(MB_INVALID_ARGUMENT,e,n,"mask field/count mismatch");
        if(!std::all_of(data,data+count,[](uint32_t v){return v<=1;}))return fail(MB_INVALID_ARGUMENT,e,n,"mask values must be 0 or 1");
        if(f==MB_INFERENCE_DURATIONS&&std::none_of(data,data+count,[](uint32_t v){return v!=0;}))return fail(MB_INVALID_ARGUMENT,e,n,"at least one duration must be enabled");
        if(f==MB_INFERENCE_GLOBAL_ROOT&&!data[0])return fail(MB_INVALID_ARGUMENT,e,n,"global root slot zero is required");
        std::transform(data,data+count,dest.begin(),[](uint32_t v){return static_cast<uint8_t>(v);});return MB_OK;
    });
}
mb_status mb_inference_request_get_mask(const mb_inference_request * r,uint32_t f,uint32_t * data,uint64_t capacity,uint64_t * count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!count)return fail(MB_INVALID_ARGUMENT,e,n,"request or count is null");
        *count=0;auto source=mask(*r,f);
        if(source.empty())return fail(MB_INVALID_ARGUMENT,e,n,"invalid mask field");
        *count=source.size();if(!data&&capacity==0)return MB_OK;
        if(!data||capacity<source.size())return fail(MB_INVALID_ARGUMENT,e,n,"mask buffer too small");
        std::copy(source.begin(),source.end(),data);return MB_OK;
    });
}
mb_status mb_inference_request_set_seed(mb_inference_request * r,uint64_t seed,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r)return fail(MB_INVALID_ARGUMENT,e,n,"request is null");r->seed=seed;return MB_OK;});
}
mb_status mb_inference_request_get_seed(const mb_inference_request * r,uint64_t * seed,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||!seed)return fail(MB_INVALID_ARGUMENT,e,n,"request or seed is null");*seed=r->seed;return MB_OK;});
}
mb_status mb_inference_request_set_sampling_argmax(mb_inference_request * r,uint32_t enabled,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||enabled>1)return fail(MB_INVALID_ARGUMENT,e,n,"request is null or argmax is not 0/1");r->argmax=enabled;return MB_OK;});
}
mb_status mb_inference_request_get_sampling_argmax(const mb_inference_request * r,uint32_t * enabled,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||!enabled)return fail(MB_INVALID_ARGUMENT,e,n,"request or argmax output is null");*enabled=r->argmax;return MB_OK;});
}
mb_status mb_inference_request_set_boundary_poses(mb_inference_request * r,const mb_model * model,uint32_t boundary,const float * roots,uint64_t nr,const float * rotations,uint64_t nq,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!model||!roots||!rotations||boundary>1||nr!=12||nq!=544)return fail(MB_INVALID_ARGUMENT,e,n,"boundary requires model, request, index 0/1, roots[4,3], rotations[4,34,4]");
        if(!safe({roots,12})||!safe({rotations,544}))return fail(MB_INVALID_ARGUMENT,e,n,"boundary values must be finite and within +/-1e4");
        for(size_t i=0;i<544;i+=4) {
            double norm=0;for(size_t j=0;j<4;++j)norm+=double(rotations[i+j])*rotations[i+j];
            if(norm<.9801||norm>1.0201)return fail(MB_INVALID_ARGUMENT,e,n,"boundary quaternions must have unit norm within .01");
        }
        motionbricks::detail::encoded_frames encoded;std::string reason;
        const auto status=motionbricks::detail::encode_context(*model,{roots,12},{rotations,544},4,encoded,reason);
        if(status!=MB_OK)return fail(status,e,n,reason);
        if(boundary==1) {
            std::copy_n(encoded.local_root.data()+8,4,encoded.local_root.data()+12);
            encoded.local_root[15]=encoded.global_root[16];
        }
        if(!safe(encoded.global_root)||!safe(encoded.local_root)||!safe(encoded.poses))return fail(MB_INVALID_ARGUMENT,e,n,"encoded boundary exceeds feature bounds");
        const std::array<std::span<const float>,3> values{encoded.global_root,encoded.local_root,encoded.poses};
        for(uint32_t f=0;f<3;++f) {
            auto dest=features(*r,f).subspan(boundary*values[f].size());
            std::copy(values[f].begin(),values[f].end(),dest.begin());
            auto flags=mask(*r,f);std::fill_n(flags.begin()+boundary*4,4,1);
            r->initialized[f][boundary]=true;
        }
        if(boundary==0)r->constraints.has_local_root[3]=0;
        return MB_OK;
    });
}
mb_status mb_model_infer(const mb_model * model,const mb_inference_request * r,mb_motion ** out,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!out)return fail(MB_INVALID_ARGUMENT,e,n,"output is null");
        *out=nullptr;
        if(!model||!r)return fail(MB_INVALID_ARGUMENT,e,n,"model or request is null");
        for(const auto & halves:r->initialized)if(!halves[0]||!halves[1])return fail(MB_INVALID_ARGUMENT,e,n,"all feature fields or both boundaries must be supplied");
        const auto & g=r->constraints.global_root;
        for(size_t i=0;i<8;++i)if(r->constraints.has_global_root[i]) {
            const double norm=double(g[i*5+3])*g[i*5+3]+double(g[i*5+4])*g[i*5+4];
            if(std::abs(norm-1)>.02)return fail(MB_INVALID_ARGUMENT,e,n,"enabled global headings require unit cosine/sine pairs");
        }
        auto result=std::make_unique<mb_motion>();std::string reason;
        const auto status=motionbricks::detail::run_transition(*model,r->constraints,*result,nullptr,nullptr,reason,r->seed,r->argmax!=0);
        if(status!=MB_OK)return fail(status,e,n,reason);
        const auto finite=[](float v){return std::isfinite(v);};
        if(!std::all_of(result->root_translations.begin(),result->root_translations.end(),finite)||!std::all_of(result->local_rotations_xyzw.begin(),result->local_rotations_xyzw.end(),finite))return fail(MB_COMPUTE_FAILED,e,n,"inference produced non-finite motion");
        *out=result.release();return MB_OK;
    });
}
}
