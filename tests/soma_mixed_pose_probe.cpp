// Isolate pose arithmetic using independent CUDA-captured pose inputs.
#include "neural_runtime.hpp"
#include "neural_precision.hpp"
#include "pose.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>
using namespace motionbricks::detail;
int main(int argc,char ** argv) try {
    if(argc!=5&&argc!=6)throw std::runtime_error("usage: soma-mixed-pose-probe BUNDLE FIXTURE OUTPUT.json cpu|vulkan [SECOND_FIXTURE]");
    const std::string backend=argv[4];
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> fixture(gguf_init_from_file(argv[2],{false,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(raw,ggml_free);
    if(!fixture||!raw)throw std::runtime_error("invalid fixture");
    const auto precision_key=gguf_find_key(fixture.get(),"fixture.precision");
    if(precision_key<0||std::string(gguf_get_val_str(fixture.get(),precision_key))!="mixed-fp16")
        throw std::runtime_error("expected mixed-FP16 fixture");
    const auto read=[&]<class T>(const char * name,ggml_context * source=nullptr){
        auto * t=ggml_get_tensor(source?source:raw,name);
        const auto type=std::is_same_v<T,float>?GGML_TYPE_F32:GGML_TYPE_I32;
        if(!t||t->type!=type)throw std::runtime_error(name);
        std::vector<T> result(ggml_nelements(t));std::memcpy(result.data(),t->data,ggml_nbytes(t));return result;
    };
    auto tokens=read.operator()<int32_t>("input.pose_tokens");
    auto roots=read.operator()<float>("input.root_values");
    auto poses=read.operator()<float>("input.pose_cond");
    auto flags=read.operator()<int32_t>("input.has_pose_cond");
    auto duration=read.operator()<int32_t>("input.num_tokens");
    auto expected=read.operator()<float>("output.pose_logits");
    if(duration.size()!=1||duration[0]<6||duration[0]>16||expected.size()!=16*80)
        throw std::runtime_error("invalid fixture shape");
    std::vector<uint8_t> mask(flags.begin(),flags.end());
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    neural_precision_scope precision(true);
    {
        // Two exactly representable half products sum to a half-way value.
        // Rounding that product before bias gives 1; fusing bias gives 1+2^-10.
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
        if(!ctx)throw std::runtime_error("cannot allocate convolution rounding test");
        auto * kernel=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,1);
        auto * input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,1);
        auto * bias=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,1);
        auto * product=precision_product(ctx.get(),kernel,input);
        auto * separate=precision_convolution_bias(ctx.get(),product,bias);
        auto * fused=half_round(ctx.get(),ggml_add(ctx.get(),product,half_round(ctx.get(),bias)));
        auto * graph=ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph,separate);ggml_build_forward_expand(graph,fused);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(),neural_backend(*runtime)),ggml_backend_buffer_free);
        if(!buffer)throw std::runtime_error("cannot allocate convolution rounding buffer");
        const std::array<float,2> weights{.5F,.5F+0x1p-11F},inputs{1,1};
        const float bias_value=0x1p-11F;
        ggml_backend_tensor_set(kernel,weights.data(),0,sizeof(weights));
        ggml_backend_tensor_set(input,inputs.data(),0,sizeof(inputs));
        ggml_backend_tensor_set(bias,&bias_value,0,sizeof(bias_value));
        if(neural_graph_compute(*runtime,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("convolution rounding compute failed");
        float actual_separate=0,actual_fused=0;
        ggml_backend_tensor_get(separate,&actual_separate,0,sizeof(float));
        ggml_backend_tensor_get(fused,&actual_fused,0,sizeof(float));
        if(actual_separate!=1.F||actual_fused!=1.F+0x1p-10F)
            throw std::runtime_error("convolution product was not rounded before bias");
    }
    {
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
        if(!ctx)throw std::runtime_error("cannot allocate selection test");
        auto * fallback=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,1);
        auto * selected=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2,2);
        auto * select_mask=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1,2);
        auto * result=precision_blend(ctx.get(),fallback,selected,select_mask);
        auto * graph=ggml_new_graph(ctx.get());ggml_build_forward_expand(graph,result);
        std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(),neural_backend(*runtime)),ggml_backend_buffer_free);
        if(!buffer)throw std::runtime_error("cannot allocate selection test buffer");
        const std::array<float,2> fallback_values{32,-32},mask_values{1,0};
        const std::array<float,4> selected_values{0x1p-20F,-0x1p-19F,0x1p-20F,-0x1p-19F};
        ggml_backend_tensor_set(fallback,fallback_values.data(),0,sizeof(fallback_values));
        ggml_backend_tensor_set(selected,selected_values.data(),0,sizeof(selected_values));
        ggml_backend_tensor_set(select_mask,mask_values.data(),0,sizeof(mask_values));
        if(neural_graph_compute(*runtime,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("selection test failed");
        std::array<float,4> got{};ggml_backend_tensor_get(result,got.data(),0,sizeof(got));
        const std::array<float,4> expected_values{0x1p-20F,-0x1p-19F,32,-32};
        if(got!=expected_values)throw std::runtime_error("binary selection lost an exactly representable half value");
    }
    std::vector<float> actual;std::vector<std::vector<float>> batch;
    std::vector<neural_tensor_trace> traces;
    {
        neural_trace_scope trace(*runtime,traces,"pose");
        if(run_pose_planner(*runtime,tokens,roots,poses,mask,16,duration[0],actual,reason)!=MB_OK)
            throw std::runtime_error(reason);
    }
    // Check the source graph's F32 positional addition independently of the
    // accumulated upstream/native error in the input projection.
    const auto snapshot=[&](const char * suffix){
        const auto it=std::find_if(traces.begin(),traces.end(),[&](const auto & t){return t.name.ends_with(suffix);});
        if(it==traces.end()||it->type!=GGML_TYPE_F32||it->data.size()!=16*1024*sizeof(float))
            throw std::runtime_error("missing positional boundary tap");
        std::vector<float> values(16*1024);std::memcpy(values.data(),it->data.data(),it->data.size());return values;
    };
    const auto before=snapshot("/_proj_input.1"),after=snapshot("/_positioned_input");
    std::vector<float> positions;
    if(!neural_copy_f32(*runtime,"pose","_position_emb.embed",positions,reason)||positions.size()!=before.size())
        throw std::runtime_error("invalid position embedding");
    for(size_t i=0;i<before.size();++i){
        const float expected_sum=before[i]+positions[i];
        if(after[i]!=expected_sum)throw std::runtime_error("pose positional addition changed the upstream FP32 contract");
    }
    traces.clear();
    if(run_pose_planner_batch(*runtime,tokens,roots,poses,mask,16,duration[0],1,batch,reason)!=MB_OK)
        throw std::runtime_error(reason);
    if(batch.size()!=1||actual.size()!=expected.size()||batch[0].size()!=expected.size())
        throw std::runtime_error("unexpected native output shape");
    double maximum=0,squared=0,batch_max=0;size_t mismatches=0,batch_mismatches=0;
    const size_t valid=static_cast<size_t>(duration[0])*80;
    for(size_t i=0;i<valid;++i){
        if(!std::isfinite(actual[i])||!std::isfinite(expected[i])||!std::isfinite(batch[0][i]))
            throw std::runtime_error("nonfinite pose output");
        const double error=std::abs(double(actual[i])-expected[i]);
        maximum=std::max(maximum,error);squared+=error*error;
        batch_max=std::max(batch_max,std::abs(double(batch[0][i])-expected[i]));
    }
    for(size_t i=0;i<valid;i+=16){
        const auto winner=[&](const auto & v){return std::max_element(v.begin()+i,v.begin()+i+16)-(v.begin()+i);};
        mismatches+=winner(actual)!=winner(expected);batch_mismatches+=winner(batch[0])!=winner(expected);
    }
    std::ostringstream batch_two_report;
    if(argc==6){
        ggml_context * second_raw=nullptr;
        std::unique_ptr<gguf_context,decltype(&gguf_free)> second(gguf_init_from_file(argv[5],{false,&second_raw}),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> second_tensors(second_raw,ggml_free);
        if(!second||!second_raw)throw std::runtime_error("invalid second fixture");
        const auto second_hash=gguf_find_key(second.get(),"motionbricks.source_sha256");
        const auto second_precision=gguf_find_key(second.get(),"fixture.precision");
        if(second_hash<0||second_precision<0||std::string(gguf_get_val_str(second.get(),second_precision))!="mixed-fp16")
            throw std::runtime_error("second fixture provenance missing");
        auto tokens2=read.operator()<int32_t>("input.pose_tokens",second_raw);
        auto roots2=read.operator()<float>("input.root_values",second_raw);
        auto poses2=read.operator()<float>("input.pose_cond",second_raw);
        auto flags2=read.operator()<int32_t>("input.has_pose_cond",second_raw);
        auto duration2=read.operator()<int32_t>("input.num_tokens",second_raw);
        auto expected2=read.operator()<float>("output.pose_logits",second_raw);
        if(duration2!=duration||expected2.size()!=expected.size()||tokens2.size()!=tokens.size()||
           roots2.size()!=roots.size()||poses2.size()!=poses.size()||flags2.size()!=flags.size())
            throw std::runtime_error("batch fixtures require equal shapes and durations");
        if(tokens2==tokens&&roots2==roots&&poses2==poses&&flags2==flags)
            throw std::runtime_error("batch fixtures must have different inputs");
        std::vector<uint8_t> mask2(flags2.begin(),flags2.end());
        std::vector<float> single2;
        if(run_pose_planner(*runtime,tokens2,roots2,poses2,mask2,16,duration[0],single2,reason)!=MB_OK)
            throw std::runtime_error(reason);
        const auto join=[](auto a,const auto & b){a.insert(a.end(),b.begin(),b.end());return a;};
        std::vector<std::vector<float>> forward,reverse;
        if(run_pose_planner_batch(*runtime,join(tokens,tokens2),join(roots,roots2),join(poses,poses2),join(mask,mask2),
                                 16,duration[0],2,forward,reason)!=MB_OK||
           run_pose_planner_batch(*runtime,join(tokens2,tokens),join(roots2,roots),join(poses2,poses),join(mask2,mask),
                                 16,duration[0],2,reverse,reason)!=MB_OK)
            throw std::runtime_error(reason);
        if(forward.size()!=2||reverse.size()!=2)throw std::runtime_error("invalid batch-two output");
        batch_two_report<<",\n  \"batch_two\": {\"second_capture_sha256\": \""
                        <<gguf_get_val_str(second.get(),second_hash)<<"\", \"requests\": [";
        for(size_t request=0;request<2;++request){
            const auto & ref=request?expected2:expected;
            const auto & single=request?single2:actual;
            const auto & got=forward[request];const auto & reversed=reverse[1-request];
            if(got.size()!=ref.size()||reversed.size()!=ref.size()||single.size()!=ref.size())
                throw std::runtime_error("invalid batch-two logit shape");
            double maximum2=0,single_error=0,order_error=0;size_t mismatches2=0;
            for(size_t i=0;i<valid;++i){
                if(!std::isfinite(got[i])||!std::isfinite(reversed[i])||!std::isfinite(ref[i])||!std::isfinite(single[i]))
                    throw std::runtime_error("nonfinite batch-two result");
                maximum2=std::max(maximum2,std::abs(double(got[i])-ref[i]));
                single_error=std::max(single_error,std::abs(double(got[i])-single[i]));
                order_error=std::max(order_error,std::abs(double(got[i])-reversed[i]));
            }
            for(size_t i=0;i<valid;i+=16){
                const auto winner=[&](const auto & v){return std::max_element(v.begin()+i,v.begin()+i+16)-(v.begin()+i);};
                mismatches2+=winner(got)!=winner(ref);
            }
            if(request)batch_two_report<<",";
            batch_two_report<<"{\"max_absolute\":"<<maximum2<<",\"pose_token_mismatches\":"<<mismatches2
                            <<",\"single_vs_batch_max\":"<<single_error<<",\"reversed_order_max\":"<<order_error<<"}";
        }
        batch_two_report<<"]}";
    }
    const auto hash_key=gguf_find_key(fixture.get(),"motionbricks.source_sha256");
    if(hash_key<0)throw std::runtime_error("fixture provenance missing");
    std::ofstream output(argv[3]);
    output<<"{\n  \"format\": \"motionbricks-soma-isolated-mixed-pose-v1\",\n  \"backend\": \""<<backend
          <<"\",\n  \"position_add_fp32_exact\": true,\n  \"binary_selection_exact\": true,\n  \"convolution_bias_rounding_exact\": true,\n  \"status\": \"diagnostic; not full model parity\",\n  \"capture_sha256\": \""
          <<gguf_get_val_str(fixture.get(),hash_key)<<"\",\n  \"matmul_operands\": \""
          <<(native_half_operands_requested()?"f16":"f32-rounded-to-f16")<<"\",\n  \"max_absolute\": "<<maximum
          <<",\n  \"rms\": "<<std::sqrt(squared/valid)<<",\n  \"pose_token_mismatches\": "<<mismatches
          <<",\n  \"batch_max_absolute\": "<<batch_max<<",\n  \"batch_pose_token_mismatches\": "<<batch_mismatches
          <<batch_two_report.str()<<"\n}\n";
    output.close();if(!output)throw std::runtime_error("cannot write diagnostic");
    std::cout<<"max_abs="<<maximum<<" token_mismatches="<<mismatches<<" batch_max="<<batch_max
             <<" batch_token_mismatches="<<batch_mismatches<<'\n';
    return 0;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
