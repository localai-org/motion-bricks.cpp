// G1 MLP/FSQ graph structure follows the pinned official NVIDIA SONIC ONNX
// export. No upstream inference code is copied; inference uses GGML only.
#include <motionbricks/sonic.h>
#include "error.hpp"
#include "handles.hpp"
#include "neural_runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>
#if defined(MOTIONBRICKS_HAVE_GGML)
#include <ggml.h>
#include <ggml-backend.h>
#include <gguf.h>
#endif

using motionbricks::detail::guard;
using motionbricks::detail::fail;
using namespace motionbricks::detail;

#if defined(MOTIONBRICKS_HAVE_GGML)
namespace {
struct ContextDelete { void operator()(ggml_context * p) const { ggml_free(p); } };
struct BufferDelete { void operator()(ggml_backend_buffer * p) const { ggml_backend_buffer_free(p); } };
struct GgufDelete { void operator()(gguf_context * p) const { gguf_free(p); } };
struct Graph {
    std::unique_ptr<ggml_context, ContextDelete> context;
    std::unique_ptr<ggml_backend_buffer, BufferDelete> buffer;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    std::vector<ggml_tensor *> layers;
    bool ready = false;
};
constexpr std::array<int64_t,6> encoder_dims{640,2048,1024,512,512,64};
constexpr std::array<int64_t,8> decoder_dims{994,2048,2048,1024,1024,512,512,29};
void identity(gguf_context * c, const char * key, const char * expected) {
    auto i=gguf_find_key(c,key);
    if(i<0 || gguf_get_kv_type(c,i)!=GGUF_TYPE_STRING || std::string(gguf_get_val_str(c,i))!=expected)
        throw std::invalid_argument(std::string("incompatible SONIC metadata: ")+key);
}
void validate(const char * file) {
    const auto bytes=std::filesystem::file_size(file);
    if(bytes<1024 || bytes>100U*1024U*1024U) throw std::invalid_argument("invalid SONIC GGUF size");
    std::unique_ptr<gguf_context,GgufDelete> c(gguf_init_from_file(file,{true,nullptr}));
    if(!c) throw std::invalid_argument("invalid SONIC GGUF");
    identity(c.get(),"general.architecture","motionbricks");
    identity(c.get(),"motionbricks.component","sonic");
    identity(c.get(),"motionbricks.skeleton","g1skel34");
    identity(c.get(),"motionbricks.source_sha256","013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3");
    identity(c.get(),"sonic.architecture","g1-mode0-mlp-fsq32-v1");
    identity(c.get(),"sonic.encoder_sha256","013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3");
    identity(c.get(),"sonic.decoder_sha256","c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed");
    if(gguf_get_n_tensors(c.get())!=24) throw std::invalid_argument("unexpected SONIC tensor count");
    for(int component=0;component<2;++component) {
        const auto dims = component==0 ? std::span<const int64_t>(encoder_dims) : std::span<const int64_t>(decoder_dims);
        for(size_t layer=0;layer+1<dims.size();++layer) for(bool bias:{false,true}) {
            const std::string name=std::string(component==0?"encoder.":"decoder.")+std::to_string(layer)+(bias?".bias":".weight");
            const auto index=gguf_find_tensor(c.get(),name.c_str());
            if(index<0 || gguf_get_tensor_type(c.get(),index)!=GGML_TYPE_F32) throw std::invalid_argument("missing/non-F32 SONIC tensor: "+name);
            const auto * ne=gguf_get_tensor_ne(c.get(),index);
            if(ne[0]!=(bias?dims[layer+1]:dims[layer]) || ne[1]!=(bias?1:dims[layer+1]) || ne[2]!=1 || ne[3]!=1)
                throw std::invalid_argument("invalid SONIC tensor shape: "+name);
            const auto offset=gguf_get_data_offset(c.get())+gguf_get_tensor_offset(c.get(),index);
            const auto amount=static_cast<uint64_t>(ne[0]*ne[1])*sizeof(float);
            if(offset>bytes || amount>bytes-offset) throw std::invalid_argument("truncated SONIC tensor: "+name);
        }
    }
}
void build(Graph & g, neural_runtime & runtime, bool encoder) {
    g.context.reset(ggml_init({2U*1024U*1024U,nullptr,true}));
    if(!g.context) throw std::bad_alloc();
    auto * c=g.context.get();
    const auto dims=encoder?std::span<const int64_t>(encoder_dims):std::span<const int64_t>(decoder_dims);
    g.input=ggml_new_tensor_1d(c,GGML_TYPE_F32,dims[0]);
    auto * x=g.input;
    for(size_t layer=0;layer+1<dims.size();++layer) {
        const auto prefix=std::string(encoder?"encoder.":"decoder.")+std::to_string(layer);
        auto * w=neural_weight(runtime,"sonic",prefix+".weight");
        auto * b=neural_weight(runtime,"sonic",prefix+".bias");
        auto * product=ggml_mul_mat(c,w,x);
        ggml_mul_mat_set_prec(product,GGML_PREC_F32);
        x=ggml_add(c,product,b);
        g.layers.push_back(x);
        if(layer+2<dims.size()) x=ggml_silu(c,x);
    }
    g.output=x;
    g.graph=ggml_new_graph_custom(c,128,false);
    ggml_build_forward_expand(g.graph,x);
    g.buffer.reset(ggml_backend_alloc_ctx_tensors(c,neural_backend(runtime)));
    if(!g.buffer) throw std::bad_alloc();
}
bool finite(const float * p,uint64_t n) { return std::all_of(p,p+n,[](float v){ return std::isfinite(v); }); }
// Far beyond physical observations, but bounded before GEMM: finite FLT_MAX
// inputs can overflow and make GGML's debug SiLU assertion abort the process.
bool safe_input(const float * p,uint64_t n) {
    return std::all_of(p,p+n,[](float v){ return std::isfinite(v) && std::abs(v)<=1.0e6F; });
}
void execute(Graph & g,neural_runtime & runtime,const float * input,float * output) {
    ggml_backend_tensor_set(g.input,input,0,ggml_nbytes(g.input));
    if(ggml_backend_graph_compute(neural_backend(runtime),g.graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("SONIC GGML graph compute failed");
    ggml_backend_tensor_get(g.output,output,0,ggml_nbytes(g.output));
    if(!finite(output,static_cast<uint64_t>(ggml_nelements(g.output)))) throw std::runtime_error("non-finite SONIC output");
    g.ready=true;
}
// ONNX Round is ties-to-even, unlike std::round/GGML's round-away-from-zero.
// Preserve the exported straight-through Sub/Add sequence in F32 as well.
float fsq(float x) {
    const float bounded=std::tanh(x+0.03223699703812599F)*15.515501022338867F-0.5F;
    const float low=std::floor(bounded), fraction=bounded-low;
    const float rounded=fraction<0.5F?low:fraction>0.5F?low+1.0F:(std::fmod(low,2.0F)==0.0F?low:low+1.0F);
    const float difference=rounded-bounded;
    return (bounded+difference)/16.0F;
}
}
struct mb_sonic {
    std::mutex mutex;
    std::shared_ptr<neural_runtime> runtime;
    Graph encoder,decoder;
};
#else
struct mb_sonic {};
#endif

extern "C" {
mb_status mb_sonic_load(const char * file,const mb_runtime_options * options,mb_sonic ** output,char * error,uint64_t cap) {
    if(output) *output=nullptr;
    return guard(error,cap,[&]() -> mb_status {
        if(!output || !file || !*file) return fail(MB_INVALID_ARGUMENT,error,cap,"SONIC file/output required");
#if defined(MOTIONBRICKS_HAVE_GGML)
        if(options && options->device>MB_DEVICE_VULKAN) return fail(MB_INVALID_ARGUMENT,error,cap,"invalid device");
        try { validate(file); } catch(const std::exception & e) { return fail(MB_INVALID_FORMAT,error,cap,e.what()); }
        auto model=std::make_unique<mb_sonic>();
        std::string reason;
        auto status=create_sonic_runtime(file,options?options->device:MB_DEVICE_CPU,options?options->threads:4,
            options?options->backend_directory:"",model->runtime,reason);
        if(status!=MB_OK) return fail(status,error,cap,reason);
        build(model->encoder,*model->runtime,true); build(model->decoder,*model->runtime,false);
        *output=model.release(); return MB_OK;
#else
        (void)options; return fail(MB_BACKEND_UNAVAILABLE,error,cap,"GGML unavailable");
#endif
    });
}
void mb_sonic_free(mb_sonic * model) { delete model; }
mb_status mb_sonic_encode(mb_sonic * m,const float * obs,uint64_t n,float * out,uint64_t count,char * error,uint64_t cap) {
    return guard(error,cap,[&]() -> mb_status {
        if(!m || !obs || !out || n!=1762 || count!=64) return fail(MB_INVALID_ARGUMENT,error,cap,"encode requires 1762 input/64 output floats");
#if defined(MOTIONBRICKS_HAVE_GGML)
        if(!safe_input(obs,n) || obs[0]!=0) return fail(MB_INVALID_ARGUMENT,error,cap,"G1 mode-0 observations must be finite and within +/-1e6");
        std::scoped_lock lock(m->mutex);
        std::array<float,640> packed{};
        for(size_t t=0;t<10;++t) {
            std::copy_n(obs+4+t*58,58,packed.data()+t*64);
            std::copy_n(obs+601+t*6,6,packed.data()+t*64+58);
        }
        std::array<float,64> raw{};
        execute(m->encoder,*m->runtime,packed.data(),raw.data());
        std::transform(raw.begin(),raw.end(),out,fsq); return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"GGML unavailable");
#endif
    });
}
mb_status mb_sonic_decode(mb_sonic * m,const float * obs,uint64_t n,float * out,uint64_t count,char * error,uint64_t cap) {
    return guard(error,cap,[&]() -> mb_status {
        if(!m || !obs || !out || n!=994 || count!=29) return fail(MB_INVALID_ARGUMENT,error,cap,"decode requires 994 input/29 output floats");
#if defined(MOTIONBRICKS_HAVE_GGML)
        if(!safe_input(obs,n)) return fail(MB_INVALID_ARGUMENT,error,cap,"observations must be finite and within +/-1e6");
        std::scoped_lock lock(m->mutex); execute(m->decoder,*m->runtime,obs,out); return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"GGML unavailable");
#endif
    });
}
mb_status mb_sonic_layer(mb_sonic * m,uint32_t enc,uint32_t layer,float * data,uint64_t capacity,uint64_t * count,char * error,uint64_t cap) {
    if(count) *count=0;
    return guard(error,cap,[&]() -> mb_status {
        if(!m || !count || enc>1 || (!data && capacity)) return fail(MB_INVALID_ARGUMENT,error,cap,"invalid trace request");
#if defined(MOTIONBRICKS_HAVE_GGML)
        std::scoped_lock lock(m->mutex); const auto & g=enc?m->encoder:m->decoder;
        if(layer>=g.layers.size() || !g.ready) return fail(MB_INVALID_ARGUMENT,error,cap,"trace unavailable");
        *count=static_cast<uint64_t>(ggml_nelements(g.layers[layer]));
        if(!data && !capacity) return MB_OK;
        if(capacity<*count) return fail(MB_INVALID_ARGUMENT,error,cap,"trace capacity too small");
        ggml_backend_tensor_get(g.layers[layer],data,0,*count*sizeof(float)); return MB_OK;
#else
        (void)layer; (void)data; return fail(MB_BACKEND_UNAVAILABLE,error,cap,"GGML unavailable");
#endif
    });
}
}
