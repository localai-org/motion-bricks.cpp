#include "soma.hpp"
#include "neural_precision.hpp"
#include "root.hpp"
#include "pose.hpp"
#include "decoder.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace motionbricks::detail {
namespace {
// 6D rotations contain the first two matrix columns, not rows.
bool change_pose(float * six, const float * from, const float * to,bool mixed_fp16,bool output_conversion=false) {
    float x[3]{six[0],six[1],six[2]}, y[3]{six[3],six[4],six[5]};
    const auto normalize = [mixed_fp16](float * v) {
        // CUDA ReduceL2 combines lanes 0 and 2 before lane 1 for width three.
        const float n = std::sqrt(mixed_fp16 ? (v[0]*v[0]+v[2]*v[2])+v[1]*v[1] :
                                              (v[0]*v[0]+v[1]*v[1])+v[2]*v[2]);
        if (!(n > 0) || !std::isfinite(n)) return false;
        for (int i=0;i<3;++i) v[i]/=n;
        return true;
    };
    if (!normalize(x)) return false;
    float z[3]{x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]};
    if (!normalize(z)) return false;
    y[0]=z[1]*x[2]-z[2]*x[1]; y[1]=z[2]*x[0]-z[0]*x[2]; y[2]=z[0]*x[1]-z[1]*x[0];
    float matrix[9], intermediate[9]{};
    for (int i=0;i<3;++i) { matrix[i*3]=x[i]; matrix[i*3+1]=y[i]; matrix[i*3+2]=z[i]; }
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        if(mixed_fp16 && output_conversion) {
            // Pinned CUDA output MatMul: fuse the second product only.
            const float first=matrix[i*3]*from[j];
            intermediate[i*3+j]=std::fma(matrix[i*3+1],from[3+j],first)+matrix[i*3+2]*from[6+j];
        } else for (int k=0;k<3;++k)intermediate[i*3+j]+=matrix[i*3+k]*from[k*3+j];
    }
    for (int column=0;column<2;++column) for (int row=0;row<3;++row) {
        six[column*3+row]=0;
        for (int k=0;k<3;++k) six[column*3+row]+=intermediate[row*3+k]*to[column*3+k];
    }
    return true;
}
}

namespace {
template<class V> bool finite(const V & values) {
    return std::all_of(values.begin(),values.end(),[](auto x){return std::isfinite(x);});
}
template<class V> bool binary(const V & values) {
    return std::all_of(values.begin(),values.end(),[](auto x){return x<=1;});
}
struct soma_support {
    std::vector<float> mean,stddev,mixed_scale,capture,standard,codebook;
    mb_status load(const neural_runtime & runtime,std::string & reason) {
        for (auto [name, target] : {std::pair{"motion_mean",&mean}, {"motion_std",&stddev},
                                   {"capture_offsets",&capture}, {"standard_offsets",&standard}})
            if (!neural_copy_f32(runtime,"support",name,*target,reason)) return MB_INCOMPATIBLE_MODEL;
        if (!neural_copy_f32(runtime,"vq-decoder","quantizer.vq._codebook.embed",codebook,reason)) return MB_INCOMPATIBLE_MODEL;
        if (mean.size()!=370||stddev.size()!=370||capture.size()!=270||standard.size()!=270||codebook.size()!=2560 ||
            !finite(mean)||!finite(stddev)||!finite(capture)||!finite(standard)||!finite(codebook)||
            std::any_of(stddev.begin(),stddev.end(),[](float x){return x<0;})) {
            reason="invalid SOMA support tensors"; return MB_INCOMPATIBLE_MODEL;
        }
#if defined(MOTIONBRICKS_HAVE_GGML)
        if(neural_weight(runtime,"support","mixed_motion_scale")) {
            if(!neural_copy_f32(runtime,"support","mixed_motion_scale",mixed_scale,reason))return MB_INCOMPATIBLE_MODEL;
            if(mixed_scale.size()!=370 || !finite(mixed_scale) ||
               std::any_of(mixed_scale.begin(),mixed_scale.end(),[](float x){return x<=0;})) {
                reason="invalid SOMA mixed normalization scales";return MB_INCOMPATIBLE_MODEL;
            }
        }
#endif
        return MB_OK;
    }
    float norm(float x,std::size_t i) const {return (x-mean[i])/std::sqrt(stddev[i]*stddev[i]+1e-5F);}
    float unnorm(float x,std::size_t i) const {return x*std::sqrt(stddev[i]*stddev[i]+1e-5F)+mean[i];}
};
struct soma_state {
    const soma_request & input;
    const soma_support & support;
    soma_result output;
    soma_boundaries boundaries;
    bool capture_taps;
    float initial_x{},initial_z{},initial_heading{};
    std::vector<float> global,local,poses,duration_logits,raw_root,cond,pose_root,external,logits,quantized,decoded;
    std::vector<std::uint8_t> mask;
    std::vector<std::int32_t> tokens;
    root_result root;
    soma_state(const soma_request & request,const soma_support & model,bool taps)
        : input(request),support(model),capture_taps(taps) {}
    float norm(float x,std::size_t i) const {
        // The exported scales also exactly match the pinned FP32 PyTorch
        // normalization expression; scalar sqrt differs in 63 components.
        return !support.mixed_scale.empty() ? (x-support.mean[i])/support.mixed_scale[i] : support.norm(x,i);
    }
    float unnorm(float x,std::size_t i) const {
        return !support.mixed_scale.empty() ? x*support.mixed_scale[i]+support.mean[i] : support.unnorm(x,i);
    }
    void record(const char * name,const std::vector<float> & values) {if(capture_taps)boundaries[name]=values;}
    mb_status prepare(std::string & reason) {
        if (!finite(input.global_root)||!finite(input.local_root)||!finite(input.poses)||
            !binary(input.has_global)||!binary(input.has_local)||!binary(input.has_pose)||!binary(input.allowed)||
            std::none_of(input.allowed.begin(),input.allowed.end(),[](auto x){return x!=0;})) {
            reason="invalid SOMA features or masks"; return MB_INVALID_ARGUMENT;
        }
        std::vector<float> retargeted(input.poses.begin(),input.poses.end());
        for (unsigned f=0;f<8;++f) for (unsigned j=0;j<30;++j)
            if (!change_pose(retargeted.data()+f*267+87+j*6,support.capture.data()+j*9,support.standard.data()+j*9,input.mixed_fp16)) {
                reason="degenerate SOMA rotation"; return MB_INVALID_ARGUMENT;
            }
        record("retargeted_poses",retargeted);
        initial_x=input.global_root[0]; initial_z=input.global_root[2];
        initial_heading=std::atan2(input.global_root[4],input.global_root[3]);
        global.resize(40);local.resize(32);poses.resize(8*268);
        for (unsigned f=0;f<8;++f) {
            for(unsigned i=0;i<5;++i) global[f*5+i]=norm(input.global_root[f*5+i]-(i==0?initial_x:i==2?initial_z:0),i);
            for(unsigned i=0;i<4;++i) local[f*4+i]=norm(input.local_root[f*4+i],5+i);
            poses[f*268]=norm(input.global_root[f*5+1],8);
            for(unsigned i=0;i<267;++i) poses[f*268+1+i]=norm(retargeted[f*267+i],9+i);
        }
        record("normalized_global",global); record("normalized_local",local); record("normalized_poses",poses);
        return MB_OK;
    }
    mb_status select_duration(std::string & reason) {
        record("duration_logits",duration_logits);
        float best=-std::numeric_limits<float>::infinity();
        for(unsigned i=0;i<11;++i) if(input.allowed[i]&&duration_logits[i]>best){best=duration_logits[i];output.tokens=6+i;}
        if(!output.tokens){reason="nonfinite SOMA duration logits";return MB_COMPUTE_FAILED;}
        return MB_OK;
    }
    void prepare_pose() {
        record("predicted_global_root",root.global_root_values);
        const unsigned frames=output.tokens*4;
        raw_root.resize(frames*5);std::vector<float> heading(frames),pred_local(frames*4);
        for(unsigned f=0;f<frames;++f){
            for(unsigned i=0;i<5;++i)raw_root[f*5+i]=unnorm(root.global_root_values[f*5+i],i);
            heading[f]=std::atan2(raw_root[f*5+4],raw_root[f*5+3]);
        }
        for(unsigned f=0;f<frames;++f){
            const unsigned a=f+1<frames?f:f-1,b=a+1;
            const float delta=heading[b]-heading[a];
            pred_local[f*4]=norm(std::atan2(std::sin(delta),std::cos(delta))*30,5);
            pred_local[f*4+1]=norm((raw_root[b*5]-raw_root[a*5])*30,6);
            pred_local[f*4+2]=norm((raw_root[b*5+2]-raw_root[a*5+2])*30,7);
            pred_local[f*4+3]=norm(raw_root[f*5+1],8);
        }
        if(input.has_local[7])std::copy_n(local.data()+28,4,pred_local.data()+(frames-1)*4);
        record("predicted_local_root",pred_local);
        cond.assign(frames*268,0);pose_root.resize(frames*4);external.resize(frames*2);
        mask.assign(frames,0);
        for(unsigned i=0;i<8;++i){const unsigned f=i<4?i:frames-8+i;std::copy_n(poses.data()+i*268,268,cond.data()+f*268);mask[f]=input.has_pose[i];}
        for(unsigned f=0;f<frames;++f){
            for(unsigned i=0;i<4;++i)pose_root[f*4+i]=root.global_root_values[f*5+(i==0?0:i+1)];
            external[f*2]=pred_local[f*4+1];external[f*2+1]=pred_local[f*4+2];
        }
        record("pose_condition",cond);record("pose_root",pose_root);record("decoder_external",external);
        tokens.assign(output.tokens*5,16);
    }
    void prepare_decoder() {
        record("pose_logits",logits);
        for(unsigned i=0;i<tokens.size();++i)tokens[i]=static_cast<std::int32_t>(std::max_element(logits.begin()+i*16,logits.begin()+(i+1)*16)-(logits.begin()+i*16));
        output.pose_tokens=tokens;
        quantized.resize(output.tokens*160);
        for(unsigned p=0;p<output.tokens;++p)for(unsigned h=0;h<5;++h)for(unsigned d=0;d<32;++d)
            quantized[(h*32+d)*output.tokens+p]=support.codebook[(h*16+tokens[p*5+h])*32+d];
        record("codebook_lookup",quantized);
    }
    mb_status finish(std::string & reason) {
        const unsigned frames=output.tokens*4;
        record("decoder_normalized",decoded);
        output.features.resize(frames*366);
        float x=0,z=0,angle=0;
        for(unsigned f=0;f<frames;++f){
            auto * out=output.features.data()+f*366;
            out[0]=x;out[1]=unnorm(decoded[f*365+3],8);out[2]=z;out[3]=std::cos(angle);out[4]=std::sin(angle);
            for(unsigned i=0;i<361;++i)out[5+i]=unnorm(decoded[f*365+4+i],9+i);
            x+=unnorm(decoded[f*365+1],6)/30;z+=unnorm(decoded[f*365+2],7)/30;angle+=unnorm(decoded[f*365],5)/30;
        }
        record("decoded_global",output.features);
        float pose_error=0,root_error=0;unsigned count=0;
        for(unsigned i=0;i<4;++i)if(input.has_global[4+i]){
            const unsigned f=frames-4+i;const float target=input.global_root[(4+i)*5+1];
            pose_error+=std::abs(output.features[f*366+1]-target);root_error+=std::abs(raw_root[f*5+1]-target);++count;
        }
        output.used_root_predictor=count&&pose_error/static_cast<float>(count)>0.05F&&root_error<pose_error;
        const float no_targets=std::numeric_limits<float>::quiet_NaN();
        record("root_selection.pose_height_error",{count?pose_error/static_cast<float>(count):no_targets});
        record("root_selection.root_height_error",{count?root_error/static_cast<float>(count):no_targets});
        record("root_selection.use_root",{output.used_root_predictor?1.0F:0.0F});
        for(unsigned f=0;f<frames;++f){
            auto * out=output.features.data()+f*366;
            if(output.used_root_predictor)std::copy_n(raw_root.data()+f*5,5,out);
            out[0]+=initial_x;out[2]+=initial_z;
            const float a=std::atan2(out[4],out[3])+initial_heading;out[3]=std::cos(a);out[4]=std::sin(a);
            for(unsigned j=0;j<30;++j)if(!change_pose(out+92+j*6,support.standard.data()+j*9,support.capture.data()+j*9,input.mixed_fp16,true)){
                reason="degenerate decoded SOMA rotation";return MB_COMPUTE_FAILED;
            }
        }
        record("output_features",output.features);
        if(!finite(output.features)){reason="nonfinite SOMA output";return MB_COMPUTE_FAILED;}        return MB_OK;
    }
};
template<class T,class Getter>
std::vector<T> collect(const std::vector<soma_state> & states,const std::vector<std::size_t> & indices,Getter getter) {
    std::vector<T> values;
    for(auto index:indices){const auto & v=getter(states[index]);values.insert(values.end(),v.begin(),v.end());}
    return values;
}
}

mb_status run_soma_features(const neural_runtime & runtime,const soma_request & input,
                           soma_result & output,std::string & reason,soma_boundaries * taps) {
    output={};if(taps)taps->clear();
    neural_precision_scope precision(input.mixed_fp16);
    soma_support support;auto status=support.load(runtime,reason);if(status!=MB_OK)return status;
    soma_state s(input,support,taps!=nullptr);
    status=s.prepare(reason);if(status!=MB_OK)return status;
    status=run_root_duration_probe(runtime,s.global,input.has_global,s.local,input.has_local,s.poses,input.has_pose,s.duration_logits,reason);
    if(status!=MB_OK)return status;
    status=s.select_duration(reason);if(status!=MB_OK)return status;
    status=run_root_planner_auto_probe(runtime,s.global,input.has_global,s.local,input.has_local,s.poses,input.has_pose,s.output.tokens,s.root,reason);
    if(status!=MB_OK)return status;
    s.prepare_pose();
    status=run_pose_planner(runtime,s.tokens,s.pose_root,s.cond,s.mask,s.output.tokens,s.output.tokens,s.logits,reason);
    if(status!=MB_OK)return status;
    s.prepare_decoder();
    status=run_vq_decoder(runtime,s.quantized,s.external,s.cond,s.mask,s.output.tokens,s.decoded,reason);
    if(status!=MB_OK)return status;
    status=s.finish(reason);if(status!=MB_OK)return status;
    output=std::move(s.output);if(taps)*taps=std::move(s.boundaries);
    return MB_OK;
}

mb_status run_soma_features_batch(const neural_runtime & runtime,std::span<const soma_request> inputs,
        std::vector<soma_result> & outputs,std::string & reason,std::vector<soma_boundaries> * taps) {
    outputs.clear();if(taps)taps->clear();
    if(inputs.empty()||inputs.size()>64){reason="SOMA batch size must be between 1 and 64";return MB_INVALID_ARGUMENT;}
    soma_support support;auto status=support.load(runtime,reason);if(status!=MB_OK)return status;
    std::vector<soma_state> states;states.reserve(inputs.size());
    std::map<bool,std::vector<std::size_t>> precision_groups;
    for(std::size_t i=0;i<inputs.size();++i){
        states.emplace_back(inputs[i],support,taps!=nullptr);
        status=states.back().prepare(reason);if(status!=MB_OK)return status;
        precision_groups[inputs[i].mixed_fp16].push_back(i);
    }
    for(const auto & [half,indices]:precision_groups){
        neural_precision_scope precision(half);
        const auto globals=collect<float>(states,indices,[](const auto & s)->const auto & {return s.global;});
        const auto locals=collect<float>(states,indices,[](const auto & s)->const auto & {return s.local;});
        const auto poses=collect<float>(states,indices,[](const auto & s)->const auto & {return s.poses;});
        const auto global_masks=collect<std::uint8_t>(states,indices,[](const auto & s)->const auto & {return s.input.has_global;});
        const auto local_masks=collect<std::uint8_t>(states,indices,[](const auto & s)->const auto & {return s.input.has_local;});
        const auto pose_masks=collect<std::uint8_t>(states,indices,[](const auto & s)->const auto & {return s.input.has_pose;});
        std::vector<std::vector<float>> duration;
        status=run_root_duration_probe_batch(runtime,globals,global_masks,locals,local_masks,poses,pose_masks,
                                             static_cast<std::uint32_t>(indices.size()),duration,reason);
        if(status!=MB_OK)return status;
        if(duration.size()!=indices.size()){reason="SOMA duration batch output count mismatch";return MB_COMPUTE_FAILED;}
        std::map<std::uint32_t,std::vector<std::size_t>> duration_groups;
        for(std::size_t j=0;j<indices.size();++j){
            auto & s=states[indices[j]];s.duration_logits=std::move(duration[j]);
            status=s.select_duration(reason);if(status!=MB_OK)return status;
            duration_groups[s.output.tokens].push_back(indices[j]);
        }
        for(const auto & [count,group]:duration_groups){
            const auto size=static_cast<std::uint32_t>(group.size());
            std::vector<root_result> roots;
            status=run_root_planner_auto_probe_batch(runtime,
                collect<float>(states,group,[](const auto & s)->const auto & {return s.global;}),
                collect<std::uint8_t>(states,group,[](const auto & s)->const auto & {return s.input.has_global;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.local;}),
                collect<std::uint8_t>(states,group,[](const auto & s)->const auto & {return s.input.has_local;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.poses;}),
                collect<std::uint8_t>(states,group,[](const auto & s)->const auto & {return s.input.has_pose;}),
                count,size,roots,reason);
            if(status!=MB_OK)return status;
            if(roots.size()!=group.size()){reason="SOMA root batch output count mismatch";return MB_COMPUTE_FAILED;}
            for(std::size_t j=0;j<group.size();++j){auto & s=states[group[j]];s.root=std::move(roots[j]);s.prepare_pose();}
            std::vector<std::vector<float>> logits,decoded;
            status=run_pose_planner_batch(runtime,
                collect<std::int32_t>(states,group,[](const auto & s)->const auto & {return s.tokens;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.pose_root;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.cond;}),
                collect<std::uint8_t>(states,group,[](const auto & s)->const auto & {return s.mask;}),
                count,count,size,logits,reason);
            if(status!=MB_OK)return status;
            if(logits.size()!=group.size()){reason="SOMA pose batch output count mismatch";return MB_COMPUTE_FAILED;}
            for(std::size_t j=0;j<group.size();++j){auto & s=states[group[j]];s.logits=std::move(logits[j]);s.prepare_decoder();}
            status=run_vq_decoder_batch(runtime,
                collect<float>(states,group,[](const auto & s)->const auto & {return s.quantized;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.external;}),
                collect<float>(states,group,[](const auto & s)->const auto & {return s.cond;}),
                collect<std::uint8_t>(states,group,[](const auto & s)->const auto & {return s.mask;}),
                count,size,decoded,reason);
            if(status!=MB_OK)return status;
            if(decoded.size()!=group.size()){reason="SOMA decoder batch output count mismatch";return MB_COMPUTE_FAILED;}
            for(std::size_t j=0;j<group.size();++j){auto & s=states[group[j]];s.decoded=std::move(decoded[j]);
                status=s.finish(reason);if(status!=MB_OK)return status;}
        }
    }
    for(auto & s:states){outputs.push_back(std::move(s.output));if(taps)taps->push_back(std::move(s.boundaries));}
    return MB_OK;
}
}
