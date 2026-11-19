// Isolate production half matrix arithmetic with identical official CUDA operands.
#include "neural_runtime.hpp"
#include "neural_precision.hpp"
#include "soma_aligned_dot.hpp"
#include "soma_aligned_ggml.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace motionbricks::detail;
int main(int argc,char ** argv) try {
    if(argc!=5&&argc!=6)throw std::runtime_error("usage: soma-matmul-probe BUNDLE FIXTURE OUTPUT.json cpu|vulkan [backend-prefix-bias|backend-split8-matrix|backend-nvjet-scores|backend-conv|backend-conv-plain|backend-nvjet-split4|backend-nvjet8-bias|backend-bias|backend-nvjet3|backend-split8|backend-pad-tail|backend-residue-first|backend-residue-first-spaced|backend-prefix-pad|backend-prefix-spaced|backend-prefix-matrix|aligned16|aligned16-split8|aligned16-graph|aligned16-tail-graph|aligned16-split8-graph|nvjet-qkv-graph|nvjet-outproj-graph|nvjet-split4-graph|nvjet-scores-graph|conv-profiled-graph]");
    const std::string backend=argv[4];
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    const std::string arithmetic=argc==6?argv[5]:"backend";
    const bool aligned_graph=arithmetic=="aligned16-tail-graph"||arithmetic=="aligned16-split8-graph"||arithmetic=="aligned16-graph"||arithmetic=="nvjet-qkv-graph"||arithmetic=="nvjet-outproj-graph"||arithmetic=="nvjet-split4-graph"||arithmetic=="nvjet-scores-graph"||arithmetic=="conv-profiled-graph";
    const bool reorder_backend=arithmetic=="backend-pad-tail"||arithmetic=="backend-residue-first"||arithmetic=="backend-residue-first-spaced"||arithmetic=="backend-prefix-pad"||arithmetic=="backend-prefix-spaced";
    const bool backend_graph=arithmetic=="backend-prefix-bias"||arithmetic=="backend-split8-matrix"||arithmetic=="backend-nvjet-scores"||arithmetic=="backend-conv"||arithmetic=="backend-conv-plain"||arithmetic=="backend-nvjet-split4"||arithmetic=="backend-nvjet8-bias"||arithmetic=="backend-bias"||arithmetic=="backend-nvjet3"||arithmetic=="backend"||arithmetic=="backend-split8"||arithmetic=="backend-prefix-matrix"||reorder_backend;
    if(!backend_graph&&arithmetic!="aligned16"&&arithmetic!="aligned16-split8"&&!aligned_graph)throw std::runtime_error("invalid arithmetic mode");
    if(!backend_graph&&backend!="cpu")throw std::runtime_error("aligned diagnostic is CPU only");
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
    const auto tensor=[&](const char * key){
        auto * value=ggml_get_tensor(raw,key);
        if(!value||value->type!=GGML_TYPE_F32)
            throw std::runtime_error("invalid matrix operand");
        return value;
    };
    auto * input=tensor("input"),*weight=tensor("weight"),*expected=tensor("expected");
    auto * bias=ggml_get_tensor(raw,"bias");
    if((arithmetic=="nvjet-outproj-graph" || arithmetic=="conv-profiled-graph" || arithmetic=="backend-nvjet8-bias" || arithmetic=="backend-prefix-bias" || arithmetic=="backend-bias" || arithmetic=="backend-conv" || arithmetic=="backend-conv-plain")!=(bias!=nullptr))
        throw std::runtime_error("bias fixture requires an output-projection or convolution mode");
    if(bias && (bias->type!=GGML_TYPE_F32 || ggml_nelements(bias)!=expected->ne[0]))throw std::runtime_error("invalid bias shape");
    auto * high_precision=ggml_get_tensor(raw,"fp64_rounded_diagnostic");
    if(high_precision&&(high_precision->type!=GGML_TYPE_F32||!ggml_are_same_shape(expected,high_precision)))
        throw std::runtime_error("invalid high-precision diagnostic");
    if(input->ne[0]!=weight->ne[0]||expected->ne[0]!=weight->ne[1]||expected->ne[1]!=input->ne[1]||
       expected->ne[2]!=input->ne[2]||expected->ne[3]!=input->ne[3]||
       input->ne[2]%weight->ne[2]||input->ne[3]%weight->ne[3])
        throw std::runtime_error("invalid matrix dimensions");
    std::vector<float> actual(ggml_nelements(expected));
    if(!backend_graph&&!aligned_graph) {
        if(input->ne[2]!=1||input->ne[3]!=1||weight->ne[2]!=1||weight->ne[3]!=1)
            throw std::runtime_error("scalar diagnostic requires unbatched matrices");
        const auto round=[](float x){return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));};
        for(auto * operand:{input,weight}) {
            const auto * values=static_cast<const float *>(operand->data);
            for(int64_t i=0;i<ggml_nelements(operand);++i)
                if(!std::isfinite(values[i])||round(values[i])!=values[i])throw std::runtime_error("expected exact finite half operands");
        }
        const auto k=static_cast<std::size_t>(input->ne[0]);
        const std::size_t parts=arithmetic=="aligned16-split8"?8:1;
        if(k%parts)throw std::runtime_error("invalid split width");
        const auto width=k/parts;
        for(int64_t m=0;m<input->ne[1];++m)for(int64_t n=0;n<weight->ne[1];++n) {
            float total=0;
            for(std::size_t part=0;part<parts;++part) {
                const auto * a=static_cast<const float *>(input->data)+m*k+part*width;
                const auto * b=static_cast<const float *>(weight->data)+n*k+part*width;
                const float partial=soma_aligned_dot(a,b,width);
                total+=parts==1?partial:round(partial);
            }
            actual[m*weight->ne[1]+n]=round(total);
        }
    } else {
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    neural_precision_scope half(true);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
    if(!ctx)throw std::runtime_error("cannot allocate graph");
    auto * x=ggml_dup_tensor(ctx.get(),input),*w=ggml_dup_tensor(ctx.get(),weight);
    auto * product_x=x,*product_w=w;
    if(reorder_backend) {
        if(input->ne[0]%16==0)throw std::runtime_error("residue probe requires a nonmultiple-of-16 K");
        const auto reorder=[&](ggml_tensor * value) {
            const int64_t remainder=value->ne[0]%16;
            if(arithmetic=="backend-pad-tail")return ggml_pad(ctx.get(),value,16-remainder,0,0,0);
            if(arithmetic=="backend-prefix-pad" || arithmetic=="backend-prefix-spaced") {
                // CUDA's short first group is the FIRST K%16 inputs, not the
                // final tail rotated to the front. Preserve operand order.
                const int64_t stride=arithmetic=="backend-prefix-spaced"?32:16;
                const auto block=[&](int64_t offset,int64_t width) {
                    auto * view=ggml_view_4d(ctx.get(),value,width,value->ne[1],value->ne[2],value->ne[3],
                        value->nb[1],value->nb[2],value->nb[3],offset*value->nb[0]);
                    return ggml_pad(ctx.get(),ggml_cont(ctx.get(),view),stride-width,0,0,0);
                };
                auto * result=block(0,remainder);
                for(int64_t offset=remainder;offset<value->ne[0];offset+=16)
                    result=ggml_concat(ctx.get(),result,block(offset,16),0);
                return result;
            }
            if(arithmetic=="backend-residue-first-spaced") {
                const auto block=[&](int64_t offset,int64_t width) {
                    auto * view=ggml_view_4d(ctx.get(),value,width,value->ne[1],value->ne[2],value->ne[3],
                        value->nb[1],value->nb[2],value->nb[3],offset*value->nb[0]);
                    return ggml_pad(ctx.get(),ggml_cont(ctx.get(),view),32-width,0,0,0);
                };
                auto * result=block(value->ne[0]-remainder,remainder);
                for(int64_t offset=0;offset<value->ne[0]-remainder;offset+=16)
                    result=ggml_concat(ctx.get(),result,block(offset,16),0);
                return result;
            }
            auto * tail=ggml_view_4d(ctx.get(),value,remainder,value->ne[1],value->ne[2],value->ne[3],
                value->nb[1],value->nb[2],value->nb[3],(value->ne[0]-remainder)*value->nb[0]);
            auto * head=ggml_view_4d(ctx.get(),value,value->ne[0]-remainder,value->ne[1],value->ne[2],value->ne[3],
                value->nb[1],value->nb[2],value->nb[3],0);
            return ggml_concat(ctx.get(),ggml_pad(ctx.get(),ggml_cont(ctx.get(),tail),16-remainder,0,0,0),
                ggml_cont(ctx.get(),head),0);
        };
        product_x=reorder(x);product_w=reorder(w);
    }
    auto * output=aligned_graph?
        half_round(ctx.get(),soma_aligned_product(ctx.get(),half_round(ctx.get(),w),half_round(ctx.get(),x),arithmetic=="aligned16-split8-graph",arithmetic=="nvjet-qkv-graph"?3:arithmetic=="nvjet-split4-graph"?4:arithmetic=="nvjet-scores-graph"?8:0,arithmetic!="aligned16-tail-graph")):
        arithmetic=="backend-nvjet-split4"?half_round(ctx.get(),precision_backend_nvjet_split4(ctx.get(),w,x)):
        arithmetic=="backend-nvjet-scores"?half_round(ctx.get(),precision_backend_nvjet(ctx.get(),w,x,8)):
        arithmetic=="backend-nvjet3"?half_round(ctx.get(),precision_backend_nvjet(ctx.get(),w,x)):
        arithmetic=="backend-split8-matrix"?half_round(ctx.get(),precision_backend_split8(ctx.get(),w,x,true)):
        arithmetic=="backend-split8"?half_round(ctx.get(),precision_backend_split8(ctx.get(),w,x)):
        arithmetic=="backend-prefix-matrix"?half_round(ctx.get(),precision_backend_prefix_product(ctx.get(),w,x)):
        precision_matmul(ctx.get(),product_w,product_x);
    ggml_tensor * graph_bias=nullptr;
    if(bias) {
        graph_bias=ggml_dup_tensor(ctx.get(),bias);
        ggml_tensor * product=nullptr;
        if(arithmetic=="conv-profiled-graph" || arithmetic=="backend-conv" || arithmetic=="backend-conv-plain") {
            const auto spatial=std::stoll(metadata("fixture.conv_spatial"));
            const auto channels=std::stoll(metadata("fixture.conv_channels"));
            const auto dilation=std::stoi(metadata("fixture.conv_dilation"));
            if(dilation<1 || spatial*channels!=w->ne[0] ||
               !((spatial==3&&(channels==160||channels==512)) || (spatial==1&&channels==512)))
                throw std::runtime_error("unsupported convolution fixture");
            product=half_round(ctx.get(),arithmetic=="backend-conv-plain"?precision_product(ctx.get(),w,x):arithmetic=="backend-conv"?precision_backend_conv(ctx.get(),w,x,spatial,channels,dilation):soma_profiled_conv_product(ctx.get(),
                half_round(ctx.get(),w),half_round(ctx.get(),x),spatial,channels,dilation));
        } else if(arithmetic=="backend-nvjet8-bias")product=precision_backend_nvjet(ctx.get(),w,x,8);
        else if(arithmetic=="backend-prefix-bias")product=precision_backend_prefix_product(ctx.get(),w,x);
        else if(arithmetic=="backend-bias")product=precision_product(ctx.get(),w,x);
        else product=soma_aligned_product(ctx.get(),half_round(ctx.get(),w),half_round(ctx.get(),x),false,8);
        output=half_round(ctx.get(),ggml_add(ctx.get(),product,half_round(ctx.get(),graph_bias)));
    }
    auto * graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,output);
    std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(),neural_backend(*runtime)),ggml_backend_buffer_free);
    if(!buffer)throw std::runtime_error("cannot allocate matrix buffer");
    ggml_backend_tensor_set(x,input->data,0,ggml_nbytes(input));
    ggml_backend_tensor_set(w,weight->data,0,ggml_nbytes(weight));
    if(bias)ggml_backend_tensor_set(graph_bias,bias->data,0,ggml_nbytes(bias));
    if(neural_graph_compute(*runtime,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("matrix compute failed");
    ggml_backend_tensor_get(output,actual.data(),0,actual.size()*sizeof(float));
    }
    const auto * target=static_cast<float *>(expected->data);
    double maximum=0,squared=0;size_t mismatches=0,native_fp64_mismatches=0,cuda_fp64_mismatches=0;
    for(size_t j=0;j<actual.size();++j){
        if(!std::isfinite(actual[j])||!std::isfinite(target[j]))throw std::runtime_error("nonfinite matrix result");
        const auto error=std::abs(double(actual[j])-target[j]);maximum=std::max(maximum,error);squared+=error*error;
        mismatches+=actual[j]!=target[j];
        if(high_precision){
            const auto diagnostic=static_cast<float *>(high_precision->data)[j];
            if(!std::isfinite(diagnostic))throw std::runtime_error("nonfinite high-precision diagnostic");
            native_fp64_mismatches+=actual[j]!=diagnostic;
            cuda_fp64_mismatches+=target[j]!=diagnostic;
        }
    }
    std::ofstream report(argv[3]);
    report<<"{\"format\":\"motionbricks-soma-isolated-native-matmul-v1\",\"backend\":\""<<backend
          <<"\",\"capture_sha256\":\""<<metadata("motionbricks.source_sha256")
          <<"\",\"node\":\""<<metadata("fixture.node")
          <<"\",\"arithmetic\":\""<<arithmetic
          <<"\",\"aligned_block_order\":\""<<(!backend_graph?"residue-first-v1":"backend")
          <<"\",\"operands\":\""<<(!backend_graph?"f32-exact-half":(native_half_operands_requested()?"f16":"f32-rounded-to-f16"))
          <<"\",\"status\":\"isolated diagnostic; not full model parity\",\"values\":"<<actual.size()
          <<",\"mismatches\":"<<mismatches<<",\"max_absolute\":"<<maximum
          <<",\"rms\":"<<std::sqrt(squared/actual.size());
    if(high_precision)report<<",\"fp64_rounded_diagnostic\":{\"native_mismatches\":"<<native_fp64_mismatches
                           <<",\"cuda_mismatches\":"<<cuda_fp64_mismatches<<"}";
    report<<",\"differences\":[";bool comma=false;
    for(size_t j=0;j<actual.size();++j)if(actual[j]!=target[j]){
        if(comma)report<<",";comma=true;
        report<<"{\"index\":"<<j<<",\"native\":"<<actual[j]<<",\"reference\":"<<target[j]<<"}";
    }
    report<<"]";
    report<<"}\n";
    report.close();if(!report)throw std::runtime_error("cannot write report");
    std::cout<<"values="<<actual.size()<<" max="<<maximum<<" mismatches="<<mismatches<<'\n';
    return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
