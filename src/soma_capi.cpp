#include <motionbricks/inference.h>
#include "inference_handles.hpp"
#include "soma.hpp"
#include "soma_playback.hpp"
#include "error.hpp"
#include "neural_precision.hpp"
#include "soma_rsqrt_table.hpp"
#include "soma_softmax.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#if defined(MOTIONBRICKS_HAVE_GGML)
#include <gguf.h>
#endif

using namespace motionbricks::detail;
namespace motionbricks::detail {
struct soma_api_skeleton {
    soma_skeleton data;
    std::vector<int> mapping;
    unsigned right_leg{},left_leg{};
};
}
namespace {
mb_status validate_soma_bundle(const std::filesystem::path & bundle,std::string & reason) {
#if defined(MOTIONBRICKS_HAVE_GGML)
    struct component {const char * name;int64_t tensors;uint64_t parameters;};
    for(auto spec:std::array{component{"pose",209,136520976},component{"root",174,40418385},
                                    component{"vq-decoder",51,12202221},component{"support",4,1280}}) {
        const auto path=bundle/(std::string(spec.name)+".gguf");
        if(!std::filesystem::is_regular_file(path)){reason="missing SOMA component: "+path.string();return MB_IO_ERROR;}
        std::unique_ptr<gguf_context,decltype(&gguf_free)> header(gguf_init_from_file(path.string().c_str(),{true,nullptr}),gguf_free);
        if(!header){reason="cannot read SOMA component: "+path.string();return MB_INVALID_FORMAT;}
        if(std::string(spec.name)=="support" && gguf_get_n_tensors(header.get())==5 &&
           gguf_find_tensor(header.get(),"mixed_motion_scale")>=0) {
            spec.tensors=5;spec.parameters=1650;
        }
        const auto string_is=[&](const char * key,const char * expected){
            const auto i=gguf_find_key(header.get(),key);
            return i>=0&&gguf_get_kv_type(header.get(),i)==GGUF_TYPE_STRING&&
                std::string(gguf_get_val_str(header.get(),i))==expected;
        };
        const auto version=gguf_find_key(header.get(),"motionbricks.format_version");
        const auto parameters=gguf_find_key(header.get(),"motionbricks.parameter_count");
        if(gguf_get_version(header.get())!=3||
           !string_is("general.architecture","motionbricks")||!string_is("motionbricks.component",spec.name)||
           !string_is("motionbricks.skeleton","soma30")||
           !string_is("motionbricks.upstream_revision","54fc563aeaaf819a753d72cc9fdc6713989cdd43")||
           version<0||gguf_get_kv_type(header.get(),version)!=GGUF_TYPE_UINT32||gguf_get_val_u32(header.get(),version)!=1||
           parameters<0||gguf_get_kv_type(header.get(),parameters)!=GGUF_TYPE_UINT64||
           gguf_get_val_u64(header.get(),parameters)!=spec.parameters||gguf_get_n_tensors(header.get())!=spec.tensors) {
            reason="incompatible SOMA component metadata or tensor inventory: "+path.string();return MB_INCOMPATIBLE_MODEL;
        }
        uint64_t total=0;
        for(int64_t i=0;i<spec.tensors;++i){
            if(gguf_get_tensor_type(header.get(),i)!=GGML_TYPE_F32){
                reason="SOMA bundles require FP32 stored tensors: "+path.string();return MB_INCOMPATIBLE_MODEL;
            }
            const auto * dimensions=gguf_get_tensor_ne(header.get(),i);uint64_t count=1;
            for(int d=0;d<4;++d){
                if(dimensions[d]<=0||static_cast<uint64_t>(dimensions[d])>spec.parameters/count){
                    reason="invalid SOMA tensor dimensions: "+path.string();return MB_INCOMPATIBLE_MODEL;
                }
                count*=static_cast<uint64_t>(dimensions[d]);
            }
            if(count>spec.parameters-total){reason="invalid SOMA parameter count: "+path.string();return MB_INCOMPATIBLE_MODEL;}
            total+=count;
        }
        if(total!=spec.parameters){reason="invalid SOMA parameter count: "+path.string();return MB_INCOMPATIBLE_MODEL;}
    }
    return MB_OK;
#else
    (void)bundle;reason="this build has no GGML support";return MB_BACKEND_UNAVAILABLE;
#endif
}
std::vector<soma_transform> transforms(const float * values,std::size_t count) {
    std::vector<soma_transform> output(count/7);
    for(std::size_t i=0;i<output.size();++i)std::copy_n(values+i*7,7,output[i].begin());
    return output;
}
std::vector<float> flatten(const std::vector<soma_transform> & values) {
    std::vector<float> output;output.reserve(values.size()*7);
    for(const auto & value:values)output.insert(output.end(),value.begin(),value.end());
    return output;
}
bool valid_transform(const soma_transform & t) {
    if(!std::all_of(t.begin(),t.end(),[](float x){return std::isfinite(x)&&std::abs(x)<=1e4F;}))return false;
    return std::abs(t[3]*t[3]+t[4]*t[4]+t[5]*t[5]+t[6]*t[6]-1)<0.02F;
}
}
mb_status motionbricks::detail::load_soma_model(const char * bundle,const mb_runtime_options * options,
    mb_model ** output,char * error,uint64_t capacity) {
    return guard(error,capacity,[&]() -> mb_status {
        if(!output)return fail(MB_INVALID_ARGUMENT,error,capacity,"output is null");
        *output=nullptr;
        const auto device=options?options->device:MB_DEVICE_AUTO;
        const auto threads=options&&options->threads?options->threads:4;
        if(!bundle||!*bundle)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA bundle directory is empty");
        if(device>MB_DEVICE_VULKAN)return fail(MB_INVALID_ARGUMENT,error,capacity,"device must be AUTO, CPU or VULKAN");
        if(threads>1024)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA thread count must be 0 (default) or 1..1024");
        auto value=std::make_unique<mb_model>();std::string reason;
        soma_public_arithmetic_scope arithmetic;
        auto status=validate_soma_bundle(bundle,reason);
        if(status!=MB_OK)return fail(status,error,capacity,reason);
        status=create_neural_runtime(bundle,device,threads,options?options->backend_directory:"",value->runtime,reason);
        if(status!=MB_OK)return fail(status,error,capacity,reason);
        std::vector<float> mean,codebook;
        if(!neural_copy_f32(*value->runtime,"support","motion_mean",mean,reason)||mean.size()!=370||
           !neural_copy_f32(*value->runtime,"vq-decoder","quantizer.vq._codebook.embed",codebook,reason)||codebook.size()!=2560)
            return fail(MB_INCOMPATIBLE_MODEL,error,capacity,"bundle does not have the SOMA feature/codebook layout");
        value->kind=MB_MODEL_SOMA;value->bundle_directory=bundle;
        value->parameter_count=136520976+40418385+12202221+1280;
        std::vector<float> scales;
        if(neural_copy_f32(*value->runtime,"support","mixed_motion_scale",scales,reason))value->parameter_count+=scales.size();
        *output=value.release();return MB_OK;
    });
}
extern "C" {
mb_status mb_model_set_arithmetic_calibration(mb_model * model,
    const char * rsqrt,const char * exp2,char * error,uint64_t capacity) {
    return guard(error,capacity,[&]() -> mb_status {
        if(!model||!rsqrt||!*rsqrt||!exp2||!*exp2)
            return fail(MB_INVALID_ARGUMENT,error,capacity,"model and calibration paths are required");
        if(model->kind!=MB_MODEL_SOMA)return fail(MB_INCOMPATIBLE_MODEL,error,capacity,"arithmetic calibration is only supported by SOMA");
        if(neural_device(*model->runtime)!=MB_DEVICE_CPU)
            return fail(MB_BACKEND_UNAVAILABLE,error,capacity,"sm120 emulation requires CPU execution");
        std::vector<float> scales;std::string reason;
        if(!neural_copy_f32(*model->runtime,"support","mixed_motion_scale",scales,reason)||scales.size()!=370||
           !std::all_of(scales.begin(),scales.end(),[](float x){return std::isfinite(x)&&x>0;}))
            return fail(MB_INCOMPATIBLE_MODEL,error,capacity,"calibration requires exact exported normalization scales");
        try {
            load_soma_rsqrt_table(rsqrt,true);load_soma_exp2_table(exp2,true);
        } catch(const std::runtime_error & exception) {
            return fail(MB_INVALID_FORMAT,error,capacity,exception.what());
        }
        model->calibrated=true;return MB_OK;
    });
}
mb_status mb_inference_request_set_skeleton(mb_inference_request * request,uint32_t joints,const int32_t * parents,const float * reference,
    uint64_t reference_count,const uint32_t * mapping,uint64_t mapping_count,uint32_t right,uint32_t left,
    char * error,uint64_t capacity) {
    return guard(error,capacity,[&]() -> mb_status {
        if(!request)return fail(MB_INVALID_ARGUMENT,error,capacity,"request is null");
        for(const auto & field:request->initialized)if(field[0]||field[1])return fail(MB_INVALID_ARGUMENT,error,capacity,"retargeting skeleton cannot be combined with raw G1 features");
        if(!parents||!reference||!mapping)return fail(MB_INVALID_ARGUMENT,error,capacity,"skeleton parents, reference transforms and mapping are required");
        if(joints<30||joints>256)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA output skeleton must contain 30..256 joints");
        if(reference_count!=joints*7)return fail(MB_INVALID_ARGUMENT,error,capacity,"reference transform count must equal joints*7 XYZ/XYZW floats");
        if(mapping_count!=30)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA requires exactly 30 model-to-pose mapping indices");
        if(right>=joints||left>=joints||right==left)return fail(MB_INVALID_ARGUMENT,error,capacity,"right/left heading joints must be distinct valid skeleton indices");
        if(parents[0]!=-1||mapping[0]!=0)return fail(MB_INVALID_ARGUMENT,error,capacity,"Hips must map to output joint zero, with root parent -1");
        auto value=std::make_shared<soma_api_skeleton>();
        value->data.parents.assign(parents,parents+joints);
        for(unsigned j=1;j<joints;++j)if(parents[j]<0||static_cast<unsigned>(parents[j])>=j)
            return fail(MB_INVALID_ARGUMENT,error,capacity,"parents must precede children");
        value->data.reference=transforms(reference,reference_count);
        if(!std::all_of(value->data.reference.begin(),value->data.reference.end(),valid_transform))
            return fail(MB_INVALID_ARGUMENT,error,capacity,"reference transforms require finite XYZ/XYZW within +/-10000 and unit quaternions");
        std::vector<bool> seen(joints,false);value->data.pose_to_model.assign(joints,-1);
        for(unsigned m=0;m<30;++m) {
            const auto p=mapping[m];
            if(p>=joints||seen[p])return fail(MB_INVALID_ARGUMENT,error,capacity,"invalid or duplicate model joint mapping");
            seen[p]=true;value->mapping.push_back(static_cast<int>(p));
            if(parents[p]==-1||value->data.pose_to_model[static_cast<unsigned>(parents[p])]>=0)
                value->data.pose_to_model[p]=static_cast<int>(m);
        }
        value->right_leg=right;value->left_leg=left;request->skeleton=std::move(value);request->transforms={};return MB_OK;
    });
}
} // extern C
mb_status motionbricks::detail::infer_soma(const mb_model * model,const mb_inference_request * request,
    mb_motion ** output,char * error,uint64_t capacity) {
    return guard(error,capacity,[&]() -> mb_status {
        if(!output)return fail(MB_INVALID_ARGUMENT,error,capacity,"output is null");
        *output=nullptr;
        const auto * skeleton=request->skeleton.get();
        const auto & cv=request->transforms[0];const auto & tv=request->transforms[1];
        const auto * context=cv.data();const auto * target=tv.data();
        const auto context_count=cv.size(),target_count=tv.size();
        const auto precision=request->precision;const auto heading_angle=request->heading;
        uint32_t allowed=0;for(unsigned i=0;i<11;++i)allowed|=uint32_t(request->constraints.allowed_tokens[i])<<i;
        if(request->sampling_set)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA uses deterministic pose argmax; seed/sampling overrides are unsupported");
        if(request->feature_masks_set)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA feature masks are derived from pose boundaries; only duration masks are configurable");
        if(!skeleton)return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA requires a skeleton: call mb_inference_request_set_skeleton");
        if(cv.empty()||tv.empty())return fail(MB_INVALID_ARGUMENT,error,capacity,"SOMA requires context and target: call mb_inference_request_set_boundary_transforms for boundaries 0 and 1");
        if(!model||!context||!target||precision>MB_PRECISION_MIXED_FP16_VULKAN_SM120)
            return fail(MB_INVALID_ARGUMENT,error,capacity,"invalid generation arguments");
        const bool calibrated=precision==MB_PRECISION_MIXED_FP16_SM120_EMULATED;
        if(calibrated&&!model->calibrated)
            return fail(MB_INVALID_ARGUMENT,error,capacity,"model has no sm120 calibration");
        const bool vulkan_calibrated=precision==MB_PRECISION_MIXED_FP16_VULKAN_SM120;
        if(vulkan_calibrated) {
            std::string qualification_reason;
            if(!neural_soma_vulkan_sm120(*model->runtime,qualification_reason))
                return fail(MB_BACKEND_UNAVAILABLE,error,capacity,qualification_reason);
            std::vector<float> scales;
            if(!neural_copy_f32(*model->runtime,"support","mixed_motion_scale",scales,qualification_reason)||
               scales.size()!=370||!std::all_of(scales.begin(),scales.end(),[](float x){return std::isfinite(x)&&x>0;}))
                return fail(MB_INCOMPATIBLE_MODEL,error,capacity,"Vulkan SM120 requires exact exported normalization scales");
        }
        soma_public_arithmetic_scope arithmetic(calibrated,vulkan_calibrated);
        const auto joints=skeleton->data.parents.size();
        if(context_count!=4*joints*7||!target_count||target_count>context_count||target_count%(joints*7))
            return fail(MB_INVALID_ARGUMENT,error,capacity,"invalid context or target transform counts");
        const auto c=transforms(context,context_count),t=transforms(target,target_count);
        soma_request encoded;soma_transform alignment;std::vector<soma_transform> world;std::string reason;
        auto status=encode_soma_poses(skeleton->data,skeleton->mapping,skeleton->right_leg,skeleton->left_leg,
            c,t,{0,0,0,0,0,0,1},allowed,encoded,alignment,world,reason,heading_angle);
        if(status!=MB_OK)return fail(status,error,capacity,reason);
        encoded.mixed_fp16=precision!=MB_PRECISION_FP32;
        soma_result prediction;
        status=run_soma_features(*model->runtime,encoded,prediction,reason);
        if(status!=MB_OK)return fail(status,error,capacity,reason);
        soma_playback playback;
        status=decode_soma_playback(prediction.features,prediction.tokens,skeleton->data,alignment,
            {0,0,0,0,0,0,1},c,t,playback,reason);
        if(status!=MB_OK)return fail(status,error,capacity,reason);
        auto value=std::make_unique<mb_motion>();
        value->frames=playback.frames;value->joints=static_cast<unsigned>(joints);value->tokens=prediction.tokens;
        value->fields[0]=flatten(playback.corrected);value->fields[1]=flatten(playback.world);
        value->fields[2]=std::move(prediction.features);value->fields[3]=std::move(playback.contacts);
        for(size_t f=0;f<value->frames;++f)for(size_t j=0;j<joints;++j) {
            const auto & pose=playback.corrected[f*joints+j];
            value->local_translations.insert(value->local_translations.end(),pose.begin(),pose.begin()+3);
            value->local_rotations_xyzw.insert(value->local_rotations_xyzw.end(),pose.begin()+3,pose.end());
            if(j==0)value->root_translations.insert(value->root_translations.end(),pose.begin(),pose.begin()+3);
        }

        *output=value.release();return MB_OK;
    });
}
extern "C" {
mb_status mb_motion_copy_field(const mb_motion * motion,uint32_t field,float * data,uint64_t capacity,
    uint64_t * count,char * error,uint64_t error_capacity) {
    return guard(error,error_capacity,[&]() -> mb_status {
        if(!count)return fail(MB_INVALID_ARGUMENT,error,error_capacity,"count is null");
        *count=0;
        if(!motion||field>=4)return fail(MB_INVALID_ARGUMENT,error,error_capacity,"motion is null or field is not LOCAL_TRANSFORMS, WORLD_TRANSFORMS, FEATURES or CONTACTS");
        const auto & values=motion->fields[field];*count=values.size();
        if(!data&&capacity==0)return MB_OK;
        if(!data||capacity<values.size())return fail(MB_INVALID_ARGUMENT,error,error_capacity,"output buffer too small");
        std::copy(values.begin(),values.end(),data);return MB_OK;
    });
}
}

extern "C" {
mb_status mb_runtime_options_set_model_kind(mb_runtime_options * o,uint32_t kind,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!o)return fail(MB_INVALID_ARGUMENT,e,n,"runtime options are null");
        if(kind>MB_MODEL_SOMA)return fail(MB_INVALID_ARGUMENT,e,n,"model kind must be MB_MODEL_G1 or MB_MODEL_SOMA");
        o->model_kind=kind;return MB_OK;
    });
}
mb_status mb_runtime_options_get_model_kind(const mb_runtime_options * o,uint32_t * kind,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!o||!kind)return fail(MB_INVALID_ARGUMENT,e,n,"options or kind output is null");*kind=o->model_kind;return MB_OK;});
}
mb_status mb_model_get_kind(const mb_model * m,uint32_t * kind,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!m||!kind)return fail(MB_INVALID_ARGUMENT,e,n,"model or kind output is null");*kind=m->kind;return MB_OK;});
}
mb_status mb_inference_request_set_precision(mb_inference_request * r,uint32_t v,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r)return fail(MB_INVALID_ARGUMENT,e,n,"request is null");if(v>3)return fail(MB_INVALID_ARGUMENT,e,n,"precision must be FP32 (0), experimental mixed FP16 (1), SM120 CPU emulation (2), or qualified Vulkan SM120 (3)");r->precision=v;return MB_OK;});
}
mb_status mb_inference_request_get_precision(const mb_inference_request * r,uint32_t * v,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||!v)return fail(MB_INVALID_ARGUMENT,e,n,"request or precision output is null");*v=r->precision;return MB_OK;});
}
mb_status mb_inference_request_set_heading(mb_inference_request * r,float v,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||!std::isfinite(v)||std::abs(v)>1e4F)return fail(MB_INVALID_ARGUMENT,e,n,"request is null or heading is not finite within +/-10000 radians");r->heading=v;return MB_OK;});
}
mb_status mb_inference_request_get_heading(const mb_inference_request * r,float * v,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!r||!v)return fail(MB_INVALID_ARGUMENT,e,n,"request or heading output is null");*v=r->heading;return MB_OK;});
}
mb_status mb_inference_request_set_boundary_transforms(mb_inference_request * r,uint32_t boundary,const float * data,uint64_t count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!data||boundary>1)return fail(MB_INVALID_ARGUMENT,e,n,"request/data is null or boundary is not 0 (context) or 1 (target)");
        if(!r->skeleton)return fail(MB_INVALID_ARGUMENT,e,n,"set the retargeting skeleton before boundary transforms");
        const auto width=r->skeleton->data.parents.size()*7;
        if((boundary==0&&count!=4*width)||(boundary==1&&(!count||count>4*width||count%width)))
            return fail(MB_INVALID_ARGUMENT,e,n,"boundary transform count must be 4*joints*7 for context, or 1..4*joints*7 for target");
        auto poses=transforms(data,count);
        if(!std::all_of(poses.begin(),poses.end(),valid_transform))return fail(MB_INVALID_ARGUMENT,e,n,"boundary transforms require finite XYZ/XYZW and unit quaternions (squared norm within .02)");
        if(!std::all_of(data,data+count,[](float x){return std::abs(x)<=1e4F;}))return fail(MB_INVALID_ARGUMENT,e,n,"boundary transform values must be within +/-10000");
        std::vector<float> copy(data,data+count);r->transforms[boundary]=std::move(copy);return MB_OK;
    });
}
mb_status mb_inference_request_get_boundary_transforms(const mb_inference_request * r,uint32_t boundary,float * data,uint64_t capacity,uint64_t * count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {
        if(!r||!count||boundary>1)return fail(MB_INVALID_ARGUMENT,e,n,"request/count is null or boundary is not 0 or 1");
        const auto & v=r->transforms[boundary];*count=v.size();
        if(!data&&!capacity)return MB_OK;
        if(!data||capacity<v.size())return fail(MB_INVALID_ARGUMENT,e,n,"boundary output buffer is too small");
        std::copy(v.begin(),v.end(),data);return MB_OK;
    });
}
mb_status mb_motion_get_local_translations(const mb_motion * m,const float ** data,uint64_t * count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!m||!data||!count)return fail(MB_INVALID_ARGUMENT,e,n,"motion or translation output is null");*data=m->local_translations.data();*count=m->local_translations.size();return MB_OK;});
}
mb_status mb_motion_get_token_count(const mb_motion * m,uint32_t * count,char * e,uint64_t n) {
    return guard(e,n,[&]() -> mb_status {if(!m||!count)return fail(MB_INVALID_ARGUMENT,e,n,"motion or token output is null");*count=m->tokens?m->tokens:static_cast<uint32_t>(m->frames/4);return MB_OK;});
}
}
