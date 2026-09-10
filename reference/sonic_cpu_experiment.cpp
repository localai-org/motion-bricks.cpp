// Linux/AVX512 diagnostic interposer, explicitly loaded with LD_PRELOAD.
// Single model per process. Not installed or linked into the normal runtime.
// SONIC_CPU_EXPERIMENT=f32|f16; SONIC_CPU_EXPERIMENT_SCOPE=decoder|all.
// F16 caches round weights only; inputs and accumulation remain F32.
#include <ggml.h>
#include <ggml-backend.h>
#include <immintrin.h>
#include <dlfcn.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace {
struct Weights {
    const float * f32;
    int width, rows;
    bool half;
    std::vector<ggml_fp16_t> f16;
};
std::vector<std::unique_ptr<Weights>> weights;
template<class T> T next(const char * name) {
    auto p=dlsym(RTLD_NEXT,name);
    if(!p) { std::fprintf(stderr,"missing %s: %s\n",name,dlerror()); std::abort(); }
    return reinterpret_cast<T>(p);
}
template<bool Half> __m512 load(const Weights & w, int row, int col) {
    const auto offset=static_cast<size_t>(row)*static_cast<size_t>(w.width)+static_cast<size_t>(col);
    if constexpr(Half) return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(w.f16.data()+offset)));
    else return _mm512_loadu_ps(w.f32+offset);
}
template<bool Half, int Rows> void tile(const Weights & w, const float * x, float * y, int row) {
        __m512 sums[Rows][4];
        for(int r=0;r<Rows;++r) for(int k=0;k<4;++k) sums[r][k]=_mm512_setzero_ps();
        const int stop=w.width & ~63;
        for(int col=0;col<stop;col+=64) {
            for(int k=0;k<4;++k) {
                const auto v=_mm512_loadu_ps(x+col+k*16);
                for(int r=0;r<Rows;++r)
                    sums[r][k]=_mm512_fmadd_ps(load<Half>(w,row+r,col+k*16),v,sums[r][k]);
            }
        }
        for(int r=0;r<Rows;++r) {
            // Same four-accumulator reduction tree as GGML's AVX512 F32 dot.
            float value=_mm512_reduce_add_ps(_mm512_add_ps(
                _mm512_add_ps(sums[r][0],sums[r][2]),_mm512_add_ps(sums[r][1],sums[r][3])));
            for(int col=stop;col<w.width;++col) {
                const auto offset=static_cast<size_t>(row+r)*static_cast<size_t>(w.width)+static_cast<size_t>(col);
                const float v=Half?ggml_fp16_to_fp32(w.f16[offset]):w.f32[offset];
                value+=v*x[col];
            }
            y[row+r]=value;
        }
}
template<bool Half> void compute(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & w=*static_cast<const Weights *>(userdata);
    const auto * x=static_cast<const float *>(dst->src[1]->data);
    auto * y=static_cast<float *>(dst->data);
    const int tiles=(w.rows+3)/4;
    const int begin=tiles*ith/nth, end=tiles*(ith+1)/nth;
    for(int t=begin;t<end;++t) {
        const int row=t*4;
        if(row+4<=w.rows) tile<Half,4>(w,x,y,row);
        else for(int r=row;r<w.rows;++r) tile<Half,1>(w,x,y,r);
    }
}
}

extern "C" ggml_tensor * ggml_mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) {
    static auto original=next<decltype(&ggml_mul_mat)>("ggml_mul_mat");
    const char * mode=std::getenv("SONIC_CPU_EXPERIMENT");
    if(!mode || (std::strcmp(mode,"f32") && std::strcmp(mode,"f16"))) {
        std::fprintf(stderr,"SONIC_CPU_EXPERIMENT must be f32 or f16\n"); std::abort();
    }
    const char * scope=std::getenv("SONIC_CPU_EXPERIMENT_SCOPE");
    if(!scope) scope="decoder";
    if(std::strcmp(scope,"decoder") && std::strcmp(scope,"all")) {
        std::fprintf(stderr,"SONIC_CPU_EXPERIMENT_SCOPE must be decoder or all\n"); std::abort();
    }
    const std::string_view name(w->name);
    const bool selected=name.starts_with("decoder.") || (!std::strcmp(scope,"all") && name.starts_with("encoder."));
    if(!selected || !name.ends_with(".weight")) return original(ctx,w,x);
    if(w->type!=GGML_TYPE_F32 || x->type!=GGML_TYPE_F32 || !ggml_is_contiguous(w) ||
       !ggml_is_contiguous(x) || !ggml_backend_buffer_is_host(w->buffer) ||
       w->ne[0]!=x->ne[0] || x->ne[1]!=1 || x->ne[2]!=1 || x->ne[3]!=1 ||
       w->ne[2]!=1 || w->ne[3]!=1 || w->ne[0]>2048 || w->ne[1]>2048) {
        std::fprintf(stderr,"experiment requires native CPU SONIC batch-one F32 matrices\n"); std::abort();
    }
    auto item=std::make_unique<Weights>();
    item->f32=static_cast<const float *>(w->data);
    item->width=static_cast<int>(w->ne[0]); item->rows=static_cast<int>(w->ne[1]);
    item->half=!std::strcmp(mode,"f16");
    if(item->half) {
        item->f16.resize(static_cast<size_t>(ggml_nelements(w)));
        ggml_fp32_to_fp16_row(item->f32,item->f16.data(),ggml_nelements(w));
    }
    std::fprintf(stderr,"SONIC experiment: %s %s, %d x %d\n",mode,w->name,item->rows,item->width);
    ggml_tensor * args[]{w,x};
    auto * result=ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],1,1,1,args,2,
        item->half?compute<true>:compute<false>,GGML_N_TASKS_MAX,item.get());
    weights.push_back(std::move(item));
    return result;
}

extern "C" void ggml_mul_mat_set_prec(ggml_tensor * tensor, ggml_prec prec) {
    static auto original=next<decltype(&ggml_mul_mat_set_prec)>("ggml_mul_mat_set_prec");
    if(tensor->op==GGML_OP_CUSTOM) {
        if(prec!=GGML_PREC_F32) std::abort();
        return;
    }
    original(tensor,prec);
}
