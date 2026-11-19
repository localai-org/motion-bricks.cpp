#include "soma_trial.hpp"
// Native SOMA pose-to-pose runner used by the independent chain parity harness.
#include "soma.hpp"
#include "neural_precision.hpp"
#include "soma_playback.hpp"
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace motionbricks::detail;
namespace {
template<class T> std::vector<T> read(ggml_context * context,const char * name) {
    auto * tensor=ggml_get_tensor(context,name);
    if(!tensor||ggml_nbytes(tensor)%sizeof(T))throw std::runtime_error(std::string("invalid tensor: ")+name);
    std::vector<T> values(ggml_nbytes(tensor)/sizeof(T));
    std::memcpy(values.data(),tensor->data,ggml_nbytes(tensor));return values;
}
}
int main(int argc,char ** argv) try {
    if(argc<4||argc>7)throw std::runtime_error("usage: motionbricks-soma-run BUNDLE INPUT.gguf OUTPUT.gguf [cpu|vulkan] [fp32|mixed-fp16|mixed-fp16-sm120] [reference-features]");
    const std::string backend=argc>=5?argv[4]:"cpu";
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    std::shared_ptr<neural_runtime> runtime;std::string reason;
    if(create_neural_runtime(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,"",runtime,reason)!=MB_OK)
        throw std::runtime_error(reason);
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> input(gguf_init_from_file(argv[2],{false,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(raw,ggml_free);
    if(!input||!tensors)throw std::runtime_error("cannot read input GGUF");
    const auto copy_capture_provenance=[&](gguf_context * destination) {
        const auto key=gguf_find_key(input.get(),"fixture.capture_sha256");
        if(key<0)return; // Legacy and closed-loop inputs need not name a capture.
        if(gguf_get_kv_type(input.get(),key)!=GGUF_TYPE_STRING)
            throw std::runtime_error("invalid fixture capture provenance");
        gguf_set_val_str(destination,"trace.capture_sha256",gguf_get_val_str(input.get(),key));
    };
    soma_skeleton skeleton{read<int>(raw,"parents"),read<int>(raw,"pose_to_model"),read<soma_transform>(raw,"reference")};
    const auto context=read<soma_transform>(raw,"context");
    auto target=read<soma_transform>(raw,"target");
    const auto count=read<unsigned>(raw,"target_count").at(0);
    if(count<1||count>4||target.size()<count*skeleton.parents.size())throw std::runtime_error("invalid target count");
    target.resize(count*skeleton.parents.size());
    const auto hips=read<unsigned>(raw,"heading_joints");
    const auto heading_key=gguf_find_key(input.get(),"fixture.heading_angle");
    const float heading_angle=heading_key>=0?gguf_get_val_f32(input.get(),heading_key):0.f;
    const auto allowed_key=gguf_find_key(input.get(),"fixture.allowed_token_bits");
    const uint32_t allowed_bits_input=allowed_key>=0?gguf_get_val_u32(input.get(),allowed_key):2047;
    soma_request request;soma_transform alignment;std::vector<soma_transform> world;
    if(encode_soma_poses(skeleton,read<int>(raw,"model_to_pose"),hips.at(0),hips.at(1),context,target,
        {0,0,0,0,0,0,1},allowed_bits_input,request,alignment,world,reason,heading_angle)!=MB_OK)throw std::runtime_error(reason);
    if(argc>=6) {
        if(std::string(argv[5])!="mixed-fp16" && std::string(argv[5])!="fp32" && std::string(argv[5])!="mixed-fp16-sm120")throw std::runtime_error("invalid precision");
        request.mixed_fp16=std::string(argv[5])!="fp32";
    }
    const bool reference_features=argc==7;
    configure_soma_trial(backend,request.mixed_fp16);
    std::unique_ptr<soma_public_arithmetic_scope> public_arithmetic;
    if(argc>=6 && std::string(argv[5])=="mixed-fp16-sm120") {
        if(backend!="vulkan" || !neural_soma_vulkan_sm120(*runtime,reason)) throw std::runtime_error("SM120 qualification: "+reason);
        public_arithmetic=std::make_unique<soma_public_arithmetic_scope>(false,true);
    }
    const char * root_policy=neural_backend_split_root_trial?"backend-f16-split8-trial":
        neural_aligned_root_trial?"aligned16-split8-trial":"backend";
    if(reference_features) {
        if(std::string(argv[6])!="reference-features")throw std::runtime_error("invalid diagnostic mode");
        const auto copy_values=[&](const char * name,auto & destination){
            const auto values=read<float>(raw,name);
            if(values.size()!=destination.size())throw std::runtime_error("invalid reference feature shape");
            std::copy(values.begin(),values.end(),destination.begin());
        };
        const auto copy_mask=[&](const char * name,auto & destination){
            const auto values=read<int32_t>(raw,name);
            if(values.size()!=destination.size())throw std::runtime_error("invalid reference mask shape");
            for(size_t i=0;i<values.size();++i){
                if(values[i]!=0&&values[i]!=1)throw std::runtime_error("invalid reference mask");
                destination[i]=static_cast<uint8_t>(values[i]);
            }
        };
        copy_values("input.global_root_values",request.global_root);
        copy_values("input.local_root_values",request.local_root);
        copy_values("input.local_poses",request.poses);
        copy_mask("input.has_global_root_values",request.has_global);
        copy_mask("input.has_local_root_values",request.has_local);
        copy_mask("input.has_local_poses",request.has_pose);
        copy_mask("input.allowed_pred_num_tokens",request.allowed);
    }
    soma_result prediction;soma_boundaries boundaries;
    std::vector<neural_tensor_trace> layer_traces;
    const char * trace_layers=std::getenv("MOTIONBRICKS_SOMA_TRACE_LAYERS");
    if(request.mixed_fp16 || (trace_layers && std::strcmp(trace_layers,"1")==0)) {
        neural_trace_scope trace(*runtime,layer_traces,"soma");
        if(run_soma_features(*runtime,request,prediction,reason,&boundaries)!=MB_OK)throw std::runtime_error(reason);
    } else if(run_soma_features(*runtime,request,prediction,reason,&boundaries)!=MB_OK)throw std::runtime_error(reason);
    soma_playback playback;
    if(decode_soma_playback(prediction.features,prediction.tokens,skeleton,alignment,{0,0,0,0,0,0,1},
        context,target,playback,reason)!=MB_OK)throw std::runtime_error(reason);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead()*(boundaries.size()+20),nullptr,true}),ggml_free);
    std::unique_ptr<gguf_context,decltype(&gguf_free)> output(gguf_init_empty(),gguf_free);
    if(!soma_rsqrt_corrections.empty())gguf_set_val_str(output.get(),"arithmetic.rsqrt_table_sha256",soma_rsqrt_table_sha256);
    gguf_set_val_str(output.get(),"arithmetic.root_projection_policy",neural_backend_prefix_trial?"backend-f16-prefix-matrix-mlp-trial":"default");
    gguf_set_val_str(output.get(),"trace.format","motionbricks-soma-playback-v1");
    copy_capture_provenance(output.get());
    gguf_set_val_str(output.get(),"backend",backend.c_str());
    std::string qualification_reason;
    const bool sm120_backend = neural_soma_vulkan_sm120(*runtime, qualification_reason);
    gguf_set_val_bool(output.get(), "backend.soma_sm120_qualified", sm120_backend);
    gguf_set_val_str(output.get(), "backend.soma_sm120_qualification_reason", qualification_reason.c_str());
    gguf_set_val_str(output.get(),"input.feature_source",reference_features?"captured-reference":"native-encoder");
    gguf_set_val_str(output.get(),"precision",request.mixed_fp16?"mixed-fp16-experimental":"fp32");
    const char * operands=!request.mixed_fp16?"f32":native_half_operands_requested()?"f16":"f32-rounded-to-f16";
    gguf_set_val_str(output.get(),"arithmetic.matmul_operands",operands);
    gguf_set_val_str(output.get(),"arithmetic.matmul_accumulation","f32");
    gguf_set_val_str(output.get(),"arithmetic.root_softmax_policy",neural_softmax_trial?"warp-sm120-exp2-table-with-fallback-trial":neural_backend_softmax_trial?"vulkan-scoped-sm120":"backend");
    gguf_set_val_str(output.get(),"arithmetic.pose_softmax_policy",neural_softmax_trial?"warp-sm120-exp2-table-with-fallback-trial":neural_backend_softmax_trial?"vulkan-scoped-sm120":"backend");
    if(neural_softmax_trial)gguf_set_val_str(output.get(),"arithmetic.exp2_table_sha256",soma_exp2_table_sha256);
    gguf_set_val_str(output.get(),"arithmetic.nvjet3_policy",neural_backend_nvjet3_trial?"backend-f16-nvjet3-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.nvjet8_policy",neural_backend_nvjet8_trial?"backend-f16-nvjet8-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.pose_scores_policy",neural_backend_pose_scores_trial?"backend-f16-pose_scores-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.root_conv_policy",neural_backend_root_conv_trial?"backend-f16-root_conv-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.conditioning_policy",neural_backend_conditioning_trial?"backend-f16-conditioning-target-pose-matrix-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.decoder_conv_policy",neural_backend_decoder_conv_trial?"backend-f16-decoder_conv-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.nvjet_split4_policy",neural_backend_nvjet_split4_trial?"backend-f16-nvjet_split4-trial":"default");
    gguf_set_val_str(output.get(),"arithmetic.norm_policy",neural_welford_trial?soma_welford_policy():neural_backend_norm_trial?"vulkan-packed-welford-diagnostic":"backend");
    gguf_set_val_str(output.get(),"arithmetic.root_policy",root_policy);
    gguf_set_val_str(output.get(),"arithmetic.matmul_policy",neural_aligned_matmul_trial?"aligned16-nvjet-soma-attention-tail-trial":"backend");
    gguf_set_val_str(output.get(),"arithmetic.aligned_block_order",neural_aligned_root_trial?"residue-first-v1":"not-used");
    gguf_set_val_u32(output.get(),"soma.tokens",prediction.tokens);
    uint32_t allowed_bits=0;
    for(size_t i=0;i<request.allowed.size();++i)if(request.allowed[i])allowed_bits|=uint32_t(1)<<i;
    gguf_set_val_u32(output.get(),"soma.allowed_token_bits",allowed_bits);
    gguf_set_val_u32(output.get(),"soma.frames",playback.frames);
    gguf_set_val_u32(output.get(),"soma.joints",static_cast<unsigned>(skeleton.parents.size()));
    const auto add=[&](const char * name,const float * values,std::size_t count_values) {
        auto * t=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,static_cast<int64_t>(count_values));
        t->data=const_cast<float *>(values);ggml_set_name(t,name);gguf_add_tensor(output.get(),t);
    };
    for(const auto & [name,values]:boundaries)add(name.c_str(),values.data(),values.size());
    add("input.global_root_values",request.global_root.data(),request.global_root.size());
    add("input.local_root_values",request.local_root.data(),request.local_root.size());
    add("input.local_poses",request.poses.data(),request.poses.size());
    add("alignment",alignment.data(),7);
    for(const auto & [name,values]:{std::pair{"decoded",&playback.decoded},{"corrected",&playback.corrected},
        {"world",&playback.world},{"start_deltas",&playback.start_deltas},{"end_deltas",&playback.end_deltas}})
        add(name,values->front().data(),values->size()*7);
    add("contacts",playback.contacts.data(),playback.contacts.size());
    if(!gguf_write_to_file(output.get(),argv[3],false))throw std::runtime_error("cannot write output");
    if(!layer_traces.empty()) {
        std::unique_ptr<ggml_context,decltype(&ggml_free)> layers_ctx(
            ggml_init({ggml_tensor_overhead()*(layer_traces.size()+1),nullptr,true}),ggml_free);
        std::unique_ptr<gguf_context,decltype(&gguf_free)> layers_file(gguf_init_empty(),gguf_free);
        if(!soma_rsqrt_corrections.empty())gguf_set_val_str(layers_file.get(),"arithmetic.rsqrt_table_sha256",soma_rsqrt_table_sha256);
        gguf_set_val_str(layers_file.get(),"trace.format","motionbricks-neural-trace-v1");
        copy_capture_provenance(layers_file.get());
        gguf_set_val_str(layers_file.get(),"precision",request.mixed_fp16?"mixed-fp16-experimental":"fp32");
        gguf_set_val_str(layers_file.get(),"arithmetic.matmul_operands",operands);
        gguf_set_val_str(layers_file.get(),"arithmetic.root_softmax_policy",neural_softmax_trial?"warp-sm120-exp2-table-with-fallback-trial":neural_backend_softmax_trial?"vulkan-scoped-sm120":"backend");
        gguf_set_val_str(layers_file.get(),"arithmetic.pose_softmax_policy",neural_softmax_trial?"warp-sm120-exp2-table-with-fallback-trial":neural_backend_softmax_trial?"vulkan-scoped-sm120":"backend");
        if(neural_softmax_trial)gguf_set_val_str(layers_file.get(),"arithmetic.exp2_table_sha256",soma_exp2_table_sha256);
        gguf_set_val_str(layers_file.get(),"arithmetic.nvjet3_policy",neural_backend_nvjet3_trial?"backend-f16-nvjet3-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.nvjet8_policy",neural_backend_nvjet8_trial?"backend-f16-nvjet8-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.pose_scores_policy",neural_backend_pose_scores_trial?"backend-f16-pose_scores-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.root_conv_policy",neural_backend_root_conv_trial?"backend-f16-root_conv-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.conditioning_policy",neural_backend_conditioning_trial?"backend-f16-conditioning-target-pose-matrix-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.decoder_conv_policy",neural_backend_decoder_conv_trial?"backend-f16-decoder_conv-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.nvjet_split4_policy",neural_backend_nvjet_split4_trial?"backend-f16-nvjet_split4-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.norm_policy",neural_welford_trial?soma_welford_policy():neural_backend_norm_trial?"vulkan-packed-welford-diagnostic":"backend");
        gguf_set_val_str(layers_file.get(),"arithmetic.root_projection_policy",neural_backend_prefix_trial?"backend-f16-prefix-matrix-mlp-trial":"default");
        gguf_set_val_str(layers_file.get(),"arithmetic.root_policy",root_policy);
        gguf_set_val_str(layers_file.get(),"arithmetic.matmul_policy",neural_aligned_matmul_trial?"aligned16-nvjet-soma-attention-tail-trial":"backend");
        gguf_set_val_str(layers_file.get(),"arithmetic.aligned_block_order",neural_aligned_root_trial?"residue-first-v1":"not-used");
        for(std::size_t i=0;i<layer_traces.size();++i) {
            auto & layer=layer_traces[i];
            const auto stem=layer.name.substr(layer.name.find_last_of('/')+1);
            if(stem.empty()||stem.find(' ')!=std::string::npos||stem.starts_with("node_")||layer.data.size()>4*1024*1024)continue;
            const auto key="tap."+std::to_string(i);
            auto * tensor=ggml_new_tensor(layers_ctx.get(),static_cast<ggml_type>(layer.type),4,layer.shape.data());
            tensor->data=layer.data.data();ggml_set_name(tensor,key.c_str());gguf_add_tensor(layers_file.get(),tensor);
            gguf_set_val_str(layers_file.get(),(key+".name").c_str(),layer.name.c_str());
            gguf_set_val_str(layers_file.get(),(key+".operation").c_str(),layer.operation.c_str());
        }
        const std::string path=std::string(argv[3])+".layers.gguf";
        if(!gguf_write_to_file(layers_file.get(),path.c_str(),false))throw std::runtime_error("cannot write layer diagnostics");
    }
    std::cout<<"SOMA "<<playback.frames<<" frames, "<<skeleton.parents.size()<<" joints -> "<<argv[3]<<'\n';
    if(reference_features)std::cout<<"Diagnostic captured-feature input; does not validate native pose encoding.\n";
    if(request.mixed_fp16)std::cout<<"Experimental half rounding with FP32 accumulation; parity is not established.\n";
    return 0;
} catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
