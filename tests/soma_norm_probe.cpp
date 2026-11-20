// Isolate the production GGML normalization expression with exact CUDA inputs.
#include "neural_runtime.hpp"
#include "soma_welford.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace motionbricks::detail;
int main(int argc,char ** argv) try {
    if(argc!=5&&argc!=6)throw std::runtime_error("usage: soma-norm-probe BUNDLE FIXTURE OUTPUT.json cpu|vulkan [welford|vulkan-packed]");
    const std::string backend=argv[4];
    const bool welford=argc==6&&std::string(argv[5])=="welford";
    const bool packed=argc==6&&std::string(argv[5])=="vulkan-packed";
    if(argc==6&&!welford&&!packed)throw std::runtime_error("unknown diagnostic mode");
    if(packed&&backend!="vulkan")throw std::runtime_error("packed diagnostic requires experimental Vulkan norm shader");
    if(welford&&(std::string(argv[5])!="welford"||backend!="cpu"))throw std::runtime_error("welford diagnostic requires CPU");
    if(const char * table=std::getenv("MOTIONBRICKS_SOMA_RSQRT_TABLE")) {
        if(!welford)throw std::runtime_error("rsqrt calibration requires Welford diagnostic");
        load_soma_rsqrt_table(table);
    }
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> fixture(gguf_init_from_file(argv[2],{false,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(raw,ggml_free);
    if(!fixture||!raw)throw std::runtime_error("invalid fixture");
    const auto metadata=[&](const char * key){
        const auto index=gguf_find_key(fixture.get(),key);
        if(index<0)throw std::runtime_error(key);
        return std::string(gguf_get_val_str(fixture.get(),index));
    };
    if(metadata("fixture.precision")!="mixed-fp16")throw std::runtime_error("wrong precision");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    std::ofstream report(argv[3]);
    report<<"{\"format\":\"motionbricks-soma-isolated-norm-v1\",\"backend\":\""<<backend
          <<"\",\"capture_sha256\":\""<<metadata("motionbricks.source_sha256")
          <<"\",\"arithmetic\":\""<<(welford?soma_welford_policy():packed?"vulkan-packed-welford-diagnostic":"backend")
          <<"\",\"rsqrt_table_sha256\":\""<<(soma_rsqrt_corrections.empty()?"":soma_rsqrt_table_sha256)
          <<"\",\"status\":\"isolated diagnostic; not full model parity\",\"names\":"
          <<metadata("fixture.norm_names")<<",\"nodes\":[";
    size_t total=0,half_mismatches=0,float_mismatches=0,nodes=0;double maximum=0;
    for(size_t index=0;;++index){
        const auto stem="norm."+std::to_string(index)+".";
        auto * input=ggml_get_tensor(raw,(stem+"input").c_str());
        if(!input)break;
        const auto tensor=[&](const char * key){
            auto * value=ggml_get_tensor(raw,(stem+key).c_str());
            if(!value||value->type!=GGML_TYPE_F32)throw std::runtime_error("invalid norm operand");
            return value;
        };
        auto * scale=tensor("scale"),*bias=tensor("bias"),*expected=tensor("expected"),*epsilon=tensor("epsilon");
        if(input->type!=GGML_TYPE_F32||!ggml_are_same_shape(input,expected)||
           ggml_nelements(scale)!=input->ne[0]||ggml_nelements(bias)!=input->ne[0]||ggml_nelements(epsilon)!=1)
            throw std::runtime_error("invalid norm shape");
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
        if(!ctx)throw std::runtime_error("cannot allocate graph");
        auto * x=ggml_dup_tensor(ctx.get(),input),*s=ggml_dup_tensor(ctx.get(),scale),*b=ggml_dup_tensor(ctx.get(),bias);
        auto * output=ggml_add(ctx.get(),ggml_mul(ctx.get(),ggml_norm(ctx.get(),x,*static_cast<float *>(epsilon->data)),s),b);
        if(packed) {
            // Diagnostic-only protocol understood by the accompanying shader patch.
            const auto width=input->ne[0], rows=ggml_nelements(input)/width;
            if(width!=512&&width!=1024)throw std::runtime_error("unsupported packed norm width");
            auto * flat=ggml_reshape_2d(ctx.get(),x,width,rows);
            auto * affine=ggml_concat(ctx.get(),ggml_reshape_2d(ctx.get(),s,width,1),
                                      ggml_reshape_2d(ctx.get(),b,width,1),1);
            auto * joined=ggml_concat(ctx.get(),flat,affine,1);
            auto * normalized=ggml_norm(ctx.get(),joined,-*static_cast<float *>(epsilon->data));
            output=ggml_cont(ctx.get(),ggml_view_2d(ctx.get(),normalized,width,rows,width*sizeof(float),0));
        }
        auto * graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,output);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(),neural_backend(*runtime)),ggml_backend_buffer_free);
        if(!buffer)throw std::runtime_error("cannot allocate norm buffer");
        ggml_backend_tensor_set(x,input->data,0,ggml_nbytes(input));
        ggml_backend_tensor_set(s,scale->data,0,ggml_nbytes(scale));
        ggml_backend_tensor_set(b,bias->data,0,ggml_nbytes(bias));
        if(neural_graph_compute(*runtime,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("norm compute failed");
        std::vector<float> actual(ggml_nelements(input));
        ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
        if(welford)for(size_t offset=0;offset<actual.size();offset+=input->ne[0])
            soma_welford_normalize(static_cast<float *>(input->data)+offset,
                static_cast<float *>(scale->data),static_cast<float *>(bias->data),
                actual.data()+offset,static_cast<size_t>(input->ne[0]),*static_cast<float *>(epsilon->data));
        const auto * target=static_cast<float *>(expected->data);
        double local_max=0,squared=0;size_t local_mismatches=0,local_float_mismatches=0;
        for(size_t j=0;j<actual.size();++j){
            if(!std::isfinite(actual[j])||!std::isfinite(target[j]))throw std::runtime_error("nonfinite norm result");
            const auto error=std::abs(double(actual[j])-target[j]);local_max=std::max(local_max,error);squared+=error*error;
            local_float_mismatches+=std::memcmp(&actual[j],&target[j],sizeof(float))!=0;
            local_mismatches+=ggml_fp32_to_fp16(actual[j])!=ggml_fp32_to_fp16(target[j]);
        }
        if(index)report<<",";
        report<<"{\"index\":"<<index<<",\"values\":"<<actual.size()<<",\"max_absolute\":"<<local_max
              <<",\"float_bit_mismatches\":"<<local_float_mismatches<<",\"rms\":"<<std::sqrt(squared/actual.size())<<",\"half_rounded_mismatches\":"<<local_mismatches<<"}";
        float_mismatches+=local_float_mismatches;total+=actual.size();half_mismatches+=local_mismatches;maximum=std::max(maximum,local_max);++nodes;
    }
    if(!nodes)throw std::runtime_error("empty norm fixture");
    report<<"],\"summary\":{\"nodes\":"<<nodes<<",\"values\":"<<total<<",\"max_absolute\":"<<maximum
          <<",\"float_bit_mismatches\":"<<float_mismatches<<",\"half_rounded_mismatches\":"<<half_mismatches<<"}}\n";
    report.close();if(!report)throw std::runtime_error("cannot write report");
    std::cout<<"nodes="<<nodes<<" values="<<total<<" max="<<maximum<<" half_mismatches="<<half_mismatches<<'\n';
    return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
