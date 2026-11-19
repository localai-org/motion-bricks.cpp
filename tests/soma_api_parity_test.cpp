#include <motionbricks/inference.h>
#include <ggml.h>
#include <gguf.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
template<class T>std::vector<T> read(ggml_context * ctx,const char * name) {
    auto * t=ggml_get_tensor(ctx,name);
    if(!t||ggml_element_size(t)!=sizeof(T))throw std::runtime_error(name);
    std::vector<T> out(ggml_nelements(t));std::memcpy(out.data(),t->data,ggml_nbytes(t));return out;
}
mb_status load_model(const char * path,mb_device device,uint32_t threads,mb_model ** out,char * e,uint64_t n) {
    mb_runtime_options * o=nullptr;auto status=mb_runtime_options_create(&o,e,n);
    if(status!=MB_OK)return status;
    mb_runtime_options_set_model_kind(o,MB_MODEL_SOMA,e,n);
    mb_runtime_options_set_device(o,device,e,n);mb_runtime_options_set_threads(o,threads,e,n);
    status=mb_model_load(path,o,out,e,n);mb_runtime_options_free(o);return status;
}
mb_status generate(const mb_model * m,mb_inference_request * r,const float * c,uint64_t nc,const float * t,uint64_t nt,uint32_t allowed,uint32_t precision,mb_motion ** out,char * e,uint64_t n) {
    *out=nullptr;
    auto status=mb_inference_request_set_boundary_transforms(r,0,c,nc,e,n);if(status!=MB_OK)return status;
    status=mb_inference_request_set_boundary_transforms(r,1,t,nt,e,n);if(status!=MB_OK)return status;
    uint32_t mask[11];for(unsigned i=0;i<11;++i)mask[i]=(allowed>>i)&1;
    status=mb_inference_request_set_mask(r,MB_INFERENCE_DURATIONS,mask,11,e,n);if(status!=MB_OK)return status;
    status=mb_inference_request_set_precision(r,precision,e,n);if(status!=MB_OK)return status;
    return mb_model_infer(m,r,out,e,n);
}
void reject_incomplete_bundle() {
    struct temporary_directory {
        std::filesystem::path path=std::filesystem::temp_directory_path()/
            ("soma-incomplete-bundle-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        temporary_directory(){if(!std::filesystem::create_directory(path))throw std::runtime_error("cannot create test directory");}
        ~temporary_directory(){std::error_code error;std::filesystem::remove_all(path,error);}
    } directory;
    struct component {const char * name;const char * tensor;int64_t size;};
    for(const auto & item:{component{"pose","placeholder",1},component{"root","placeholder",1},
                          component{"support","motion_mean",370},component{"vq-decoder","quantizer.vq._codebook.embed",2560}}){
        std::unique_ptr<gguf_context,decltype(&gguf_free)> header(gguf_init_empty(),gguf_free);
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({ggml_tensor_overhead(),nullptr,true}),ggml_free);
        if(!header||!ctx)throw std::runtime_error("cannot create incomplete fixture");
        std::vector<float> values(item.size,0);
        auto * tensor=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,item.size);
        tensor->data=values.data();ggml_set_name(tensor,item.tensor);gguf_add_tensor(header.get(),tensor);
        gguf_set_val_str(header.get(),"general.architecture","motionbricks");
        gguf_set_val_str(header.get(),"motionbricks.component",item.name);
        gguf_set_val_str(header.get(),"motionbricks.skeleton","soma30");
        gguf_set_val_str(header.get(),"motionbricks.upstream_revision","54fc563aeaaf819a753d72cc9fdc6713989cdd43");
        gguf_set_val_u32(header.get(),"motionbricks.format_version",1);
        gguf_set_val_u64(header.get(),"motionbricks.parameter_count",item.size);
        if(!gguf_write_to_file(header.get(),(directory.path/(std::string(item.name)+".gguf")).string().c_str(),false))
            throw std::runtime_error("cannot write incomplete fixture");
    }
    mb_model * model=nullptr;char message[512];
    const auto status=load_model(directory.path.string().c_str(),MB_DEVICE_CPU,1,&model,message,sizeof(message));
    if(model)mb_model_free(model);
    if(status!=MB_INCOMPATIBLE_MODEL||model||!std::strstr(message,"inventory"))
        throw std::runtime_error("incomplete neural bundle was not rejected during model load");
}
int main(int argc,char ** argv) try {
    if(argc!=3&&argc!=4&&argc!=6)throw std::runtime_error("usage: soma-api-parity BUNDLE PLAYBACK_FIXTURE [cpu|vulkan [RSQRT_TABLE EXP2_TABLE]]");
    const bool calibrated=argc==6;
    const std::string backend=argc>=4?argv[3]:"cpu";
    if(backend!="cpu"&&backend!="vulkan")throw std::runtime_error("invalid backend");
    char message[1024];
    reject_incomplete_bundle();
    const auto check=[&](mb_status s){if(s!=MB_OK)throw std::runtime_error(message);};
    ggml_context * raw=nullptr;
    std::unique_ptr<gguf_context,decltype(&gguf_free)> file(gguf_init_from_file(argv[2],{false,&raw}),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> tensors(raw,ggml_free);
    if(!file||!tensors)throw std::runtime_error("invalid fixture");
    auto parents=read<int32_t>(raw,"parents");auto reference=read<float>(raw,"reference");
    auto mapping=read<uint32_t>(raw,"model_to_pose");auto heading=read<uint32_t>(raw,"heading_joints");
    mb_inference_request * skeleton_raw=nullptr;
    check(mb_inference_request_create(&skeleton_raw,message,sizeof(message)));
    std::unique_ptr<mb_inference_request,decltype(&mb_inference_request_free)> skeleton(skeleton_raw,mb_inference_request_free);
    check(mb_inference_request_set_skeleton(skeleton.get(),parents.size(),parents.data(),reference.data(),reference.size(),mapping.data(),mapping.size(),heading[0],heading[1],message,sizeof(message)));
    auto invalid=mapping;invalid[1]=invalid[0];
    if(mb_inference_request_set_skeleton(skeleton.get(),parents.size(),parents.data(),reference.data(),reference.size(),invalid.data(),invalid.size(),heading[0],heading[1],message,sizeof(message))!=MB_INVALID_ARGUMENT)
        throw std::runtime_error("duplicate mapping was not rejected");
    mb_model * model_raw=nullptr;
    check(load_model(argv[1],backend=="cpu"?MB_DEVICE_CPU:MB_DEVICE_VULKAN,1,&model_raw,message,sizeof(message)));
    std::unique_ptr<mb_model,decltype(&mb_model_free)> model(model_raw,mb_model_free);
    auto context=read<float>(raw,"context"),target=read<float>(raw,"target");
    target.resize(read<uint32_t>(raw,"target_count")[0]*parents.size()*7);
    mb_motion * motion_raw=nullptr;
    if(generate(model.get(),skeleton.get(),context.data(),context.size()-1,target.data(),target.size(),
        2047,MB_PRECISION_FP32,&motion_raw,message,sizeof(message))!=MB_INVALID_ARGUMENT||motion_raw)
        throw std::runtime_error("incorrect context count was not rejected");
    std::vector<float> fp32_before;
    const auto fp32_features=[&]() {
        mb_motion * result=nullptr;
        check(generate(model.get(),skeleton.get(),context.data(),context.size(),target.data(),target.size(),
            2047,MB_PRECISION_FP32,&result,message,sizeof(message)));
        std::unique_ptr<mb_motion,decltype(&mb_motion_free)> owner(result,mb_motion_free);
        uint64_t count=0;check(mb_motion_copy_field(result,MB_MOTION_FEATURES,nullptr,0,&count,message,sizeof(message)));
        std::vector<float> values(count);
        check(mb_motion_copy_field(result,MB_MOTION_FEATURES,values.data(),count,&count,message,sizeof(message)));
        return values;
    };
    if(calibrated) {
        if(generate(model.get(),skeleton.get(),context.data(),context.size(),target.data(),target.size(),
            2047,MB_PRECISION_MIXED_FP16_SM120_EMULATED,&motion_raw,message,sizeof(message))!=MB_INVALID_ARGUMENT||motion_raw)
            throw std::runtime_error("uncalibrated model accepted calibrated inference");
        fp32_before=fp32_features();
        if(mb_model_set_arithmetic_calibration(model.get(),"/nonexistent/soma-table",argv[5],message,sizeof(message))!=MB_INVALID_FORMAT)
            throw std::runtime_error("missing calibration was not rejected");
        check(mb_model_set_arithmetic_calibration(model.get(),argv[4],argv[5],message,sizeof(message)));
        check(mb_model_set_arithmetic_calibration(model.get(),argv[4],argv[5],message,sizeof(message)));
    }
    check(generate(model.get(),skeleton.get(),context.data(),context.size(),target.data(),target.size(),
        2047,calibrated?MB_PRECISION_MIXED_FP16_SM120_EMULATED:MB_PRECISION_FP32,&motion_raw,message,sizeof(message)));
    if(calibrated&&fp32_before!=fp32_features())throw std::runtime_error("calibrated request changed FP32 output");
    std::unique_ptr<mb_motion,decltype(&mb_motion_free)> motion(motion_raw,mb_motion_free);
    check(mb_inference_request_set_precision(skeleton.get(),calibrated?MB_PRECISION_MIXED_FP16_SM120_EMULATED:MB_PRECISION_FP32,message,sizeof(message)));
    // The generic batch API preserves single-request results and clears all
    // outputs if any request is invalid, including after a successful item.
    const mb_inference_request * requests[2]={skeleton.get(),skeleton.get()};
    mb_motion * batch[2]={};
    check(mb_model_infer_batch(model.get(),requests,2,batch,message,sizeof(message)));
    uint64_t expected_count=0;
    check(mb_motion_copy_field(motion.get(),MB_MOTION_FEATURES,nullptr,0,&expected_count,message,sizeof(message)));
    std::vector<float> single(expected_count);
    check(mb_motion_copy_field(motion.get(),MB_MOTION_FEATURES,single.data(),single.size(),&expected_count,message,sizeof(message)));
    for(auto * item:batch) {
        std::vector<float> values(single.size());uint64_t count=0;
        check(mb_motion_copy_field(item,MB_MOTION_FEATURES,values.data(),values.size(),&count,message,sizeof(message)));
        if(count!=single.size()||values!=single)throw std::runtime_error("generic batch differs from single inference");
        mb_motion_free(item);
    }
    requests[1]=nullptr;
    if(mb_model_infer_batch(model.get(),requests,2,batch,message,sizeof(message))!=MB_INVALID_ARGUMENT||batch[0]||batch[1]||!message[0])
        throw std::runtime_error("generic batch failure did not clear outputs with a useful error");
    // The animation must own its data independently of the model and skeleton.
    model.reset();skeleton.reset();
    uint64_t frames=0,joints=0;uint32_t tokens=0;
    check(mb_motion_get_frame_count(motion.get(),&frames,message,sizeof(message)));
    check(mb_motion_get_joint_count(motion.get(),&joints,message,sizeof(message)));
    check(mb_motion_get_token_count(motion.get(),&tokens,message,sizeof(message)));
    if(joints!=parents.size()||tokens!=read<uint32_t>(raw,"tokens")[0]||frames!=tokens*4-6)
        throw std::runtime_error("motion shape mismatch");
    double maximum=0;
    for(const auto & [field,name]:{std::pair{MB_MOTION_LOCAL_TRANSFORMS,"corrected"},
        {MB_MOTION_WORLD_TRANSFORMS,"world"},{MB_MOTION_FEATURES,"features"},{MB_MOTION_CONTACTS,"contacts"}}) {
        uint64_t count=0;check(mb_motion_copy_field(motion.get(),field,nullptr,0,&count,message,sizeof(message)));
        std::vector<float> values(count),expected=read<float>(raw,name);
        if(count!=expected.size())throw std::runtime_error("copy size mismatch");
        float sentinel=123;
        if(mb_motion_copy_field(motion.get(),field,&sentinel,1,&count,message,sizeof(message))!=MB_INVALID_ARGUMENT||sentinel!=123)
            throw std::runtime_error("short buffer was not preserved");
        check(mb_motion_copy_field(motion.get(),field,values.data(),values.size(),&count,message,sizeof(message)));
        for(std::size_t i=0;i<values.size();++i) {
            if(!std::isfinite(values[i]))throw std::runtime_error("nonfinite API output");
            // Quaternions may use either sign; align each transform before comparison.
            if(field<2&&i%7==3) {
                float dot=0;for(unsigned q=0;q<4;++q)dot+=values[i+q]*expected[i+q];
                if(dot<0)for(unsigned q=0;q<4;++q)expected[i+q]=-expected[i+q];
            }
            maximum=std::max(maximum,std::abs(double(values[i])-expected[i]));
        }
    }
    std::cout<<"SOMA C API "<<backend<<" max_abs="<<maximum<<" frames="<<frames<<" joints="<<joints<<'\n';
    return maximum<(calibrated?1e-6:1e-3)?0:1;
} catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
