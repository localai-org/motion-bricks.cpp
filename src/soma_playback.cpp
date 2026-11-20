#include "soma_playback.hpp"
#include "soma.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace motionbricks::detail {
namespace {
using vec = std::array<float, 3>;
using quat = std::array<float, 4>;
constexpr quat identity{0,0,0,1};
vec cross(vec a, vec b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
template<std::size_t N> std::array<float,N> normalized(std::array<float,N> a) {
    float norm=0;
    for(float v:a) norm+=v*v;
    if(!(norm>1e-20F)||!std::isfinite(norm)) throw std::runtime_error("degenerate SOMA rotation");
    norm=std::sqrt(norm);
    if constexpr(N==4) {
        const float reciprocal=1/norm;
        for(float &v:a) v*=reciprocal;
    } else for(float &v:a) v/=norm;
    return a;
}
quat rotation(const soma_transform & t) { return {t[3],t[4],t[5],t[6]}; }
vec position(const soma_transform & t) { return {t[0],t[1],t[2]}; }
soma_transform transform(vec p,quat q) {return {p[0],p[1],p[2],q[0],q[1],q[2],q[3]};}
quat inverse(quat q) {return {-q[0],-q[1],-q[2],q[3]};}
quat mul(quat a,quat b) {
    // Preserve Warp's summation order: reassociation can change FP16 features.
    return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
            a[3]*b[1]+b[3]*a[1]+a[2]*b[0]-b[2]*a[0],
            a[3]*b[2]+b[3]*a[2]+a[0]*b[1]-b[0]*a[1],
            a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}
vec rotate(quat q,vec v) {
    // Match Warp's arithmetic, including its unit-quaternion assumption.
    const float c=2*q[3]*q[3]-1,d=2*(q[0]*v[0]+q[1]*v[1]+q[2]*v[2]);
    const auto product=cross({q[0],q[1],q[2]},v);
    for(unsigned i=0;i<3;++i)v[i]=v[i]*c+q[i]*d+product[i]*q[3]*2;
    return v;
}
soma_transform mul(const soma_transform & a,const soma_transform & b) {
    auto p=rotate(rotation(a),position(b));
    for(unsigned i=0;i<3;++i)p[i]+=a[i];
    return transform(p,mul(rotation(a),rotation(b)));
}
soma_transform inverse(const soma_transform & t) {
    auto q=inverse(rotation(t));
    return transform(rotate(q,{-t[0],-t[1],-t[2]}),q);
}
quat slerp(quat b,float t) {
    // Warp interpolates through axis-angle, without normalizing the result.
    const float length=std::sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
    vec axis{};
    if(length>0)for(unsigned i=0;i<3;++i)axis[i]=b[i]/length;
    if(b[3]<0)for(float &v:axis)v=-v;
    const float angle=2*std::atan2(length,std::abs(b[3]));
    const float half=(t*angle)*0.5F,scale=std::sin(half);
    return mul(identity,quat{axis[0]*scale,axis[1]*scale,axis[2]*scale,std::cos(half)});
}
soma_transform weighted(const soma_transform & t,float w) {
    return transform({t[0]*w,t[1]*w,t[2]*w},slerp(rotation(t),w));
}
quat from_six(const float * s) {
    const auto x=normalized(vec{s[0],s[1],s[2]});
    const auto z=normalized(cross(x,{s[3],s[4],s[5]}));
    const auto y=cross(z,x);
    const float m[3][3]{{x[0],y[0],z[0]},{x[1],y[1],z[1]},{x[2],y[2],z[2]}};
    quat q{};
    const float trace=m[0][0]+m[1][1]+m[2][2];
    if(trace>=0) {
        float h=std::sqrt(trace+1);
        q[3]=0.5F*h;h=0.5F/h;
        q[0]=(m[2][1]-m[1][2])*h;q[1]=(m[0][2]-m[2][0])*h;q[2]=(m[1][0]-m[0][1])*h;
    } else {
        unsigned i=0;
        if(m[1][1]>m[0][0])i=1;
        if(m[2][2]>m[i][i])i=2;
        const unsigned j=(i+1)%3,k=(i+2)%3;
        float h=std::sqrt((m[i][i]-(m[j][j]+m[k][k]))+1);
        q[i]=0.5F*h;h=0.5F/h;
        q[j]=(m[i][j]+m[j][i])*h;q[k]=(m[i][k]+m[k][i])*h;
        q[3]=(m[k][j]-m[j][k])*h;
    }
    return normalized(q);
}
}

mb_status encode_soma_poses(const soma_skeleton & skeleton, std::span<const int> mapping,
    unsigned right_leg, unsigned left_leg, std::span<const soma_transform> context,
    std::span<const soma_transform> target, const soma_transform & runtime_to_model,
    unsigned allowed, soma_request & out, soma_transform & alignment,
    std::vector<soma_transform> & world, std::string & reason, float desired_heading_angle) {
    out={};world.clear();
    const auto joints=skeleton.parents.size();
    if(!joints||context.size()!=4*joints||target.empty()||target.size()%joints||
       target.size()>4*joints||mapping.size()!=30||mapping[0]!=0||
       right_leg>=joints||left_leg>=joints||!std::isfinite(desired_heading_angle)||!(allowed&2047)||allowed>2047) {
        reason="invalid SOMA encoding dimensions or duration mask";return MB_INVALID_ARGUMENT;
    }
    for(std::size_t j=0;j<joints;++j) {
        if((j==0&&skeleton.parents[j]!=-1)||(j&&(skeleton.parents[j]<0||static_cast<std::size_t>(skeleton.parents[j])>=j))) {
            reason="invalid SOMA encoding hierarchy";return MB_INVALID_ARGUMENT;
        }
    }
    for(int joint:mapping)if(joint<0||static_cast<std::size_t>(joint)>=joints) {
        reason="invalid SOMA encoding joint map";return MB_INVALID_ARGUMENT;
    }
    const auto valid=[](const soma_transform & t) {
        if(!std::all_of(t.begin(),t.end(),[](float x){return std::isfinite(x);}))return false;
        const auto q=rotation(t);
        return std::abs(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]-1)<0.02F;
    };
    if(!valid(runtime_to_model)||!std::all_of(context.begin(),context.end(),valid)||!std::all_of(target.begin(),target.end(),valid)) {
        reason="invalid SOMA encoding transform";return MB_INVALID_ARGUMENT;
    }
    const auto target_frames=static_cast<unsigned>(target.size()/joints);
    world.resize(8*joints);
    for(unsigned f=0;f<8;++f)for(std::size_t j=0;j<joints;++j) {
        const auto & local=f<4?context[f*joints+j]:target[std::min(f-4,target_frames-1)*joints+j];
        world[f*joints+j]=j?mul(world[f*joints+static_cast<std::size_t>(skeleton.parents[j])],local):mul(runtime_to_model,local);
    }
    const auto heading=[](vec right,vec left) {
        const float dx=right[0]-left[0],dz=right[2]-left[2];
        const float length=std::sqrt(dx*dx+dz*dz);
        if(length<=1e-6F)return 0.0F;
        // Upstream normalizes across before crossing it with the up axis.
        return std::atan2(dz/length,-(dx/length));
    };
    const float angle=heading(position(world[3*joints+right_leg]),position(world[3*joints+left_leg]))-desired_heading_angle;
    alignment=transform({world[3*joints][0],0,world[3*joints][2]},{0,std::sin(angle/2),0,std::cos(angle/2)});
    const auto aligned=[&](unsigned f,unsigned j) {
        const auto & p=world[f*joints+j];
        return rotate(inverse(rotation(alignment)),{p[0]-alignment[0],p[1]-alignment[1],p[2]-alignment[2]});
    };
    for(unsigned f=0;f<8;++f) {
        const unsigned next=f==3?f:std::min(f+1,7U);
        const auto root=aligned(f,0),following=aligned(next,0);
        const float current=heading(aligned(f,right_leg),aligned(f,left_leg));
        const float delta=heading(aligned(next,right_leg),aligned(next,left_leg))-current;
        auto * g=out.global_root.data()+f*5;
        g[0]=root[0];g[1]=root[1];g[2]=root[2];g[3]=std::cos(current);g[4]=std::sin(current);
        auto * l=out.local_root.data()+f*4;
        l[0]=std::atan2(std::sin(delta),std::cos(delta))*30;
        l[1]=(following[0]-root[0])*30;l[2]=(following[2]-root[2])*30;l[3]=root[1];
        for(unsigned m=0;m<30;++m) {
            const auto joint=static_cast<unsigned>(mapping[m]);
            const auto p=aligned(f,joint);
            if(m)for(unsigned d=0;d<3;++d)out.poses[f*267+(m-1)*3+d]=p[d]-(d==1?0:root[d]);
            const auto q=mul(inverse(rotation(alignment)),rotation(world[f*joints+joint]));
            const auto x=rotate(q,{1,0,0}),y=rotate(q,{0,1,0});
            for(unsigned d=0;d<3;++d) {
                out.poses[f*267+87+m*6+d]=x[d];out.poses[f*267+90+m*6+d]=y[d];
            }
        }
        const bool given=f<4+target_frames;
        out.has_global[f]=out.has_pose[f]=given;
        out.has_local[f]=given&&f!=3&&f!=3+target_frames;
    }
    for(unsigned i=0;i<11;++i)out.allowed[i]=(allowed>>i)&1;
    return MB_OK;
}

mb_status run_soma_motion_batch(const neural_runtime & runtime,std::span<const soma_motion_request> inputs,
    std::vector<soma_motion_result> & outputs,std::string & reason,bool capture_boundaries) {
    outputs.clear();
    if(inputs.empty()||inputs.size()>64){reason="SOMA motion batch size must be between 1 and 64";return MB_INVALID_ARGUMENT;}
    std::vector<soma_motion_result> results(inputs.size());
    std::vector<soma_request> features(inputs.size());
    for(std::size_t i=0;i<inputs.size();++i){
        const auto & input=inputs[i];auto & result=results[i];
        if(!input.skeleton){reason="SOMA motion batch skeleton is null";return MB_INVALID_ARGUMENT;}
        const auto status=encode_soma_poses(*input.skeleton,input.model_to_pose,input.right_leg,input.left_leg,
            input.context,input.target,input.runtime_to_model,input.allowed_token_bits,features[i],
            result.alignment,result.input_world,reason);
        if(status!=MB_OK)return status;
        features[i].mixed_fp16=input.mixed_fp16;
        result.encoded=features[i];
    }
    std::vector<soma_result> predictions;std::vector<soma_boundaries> taps;
    auto status=run_soma_features_batch(runtime,features,predictions,reason,capture_boundaries?&taps:nullptr);
    if(status!=MB_OK)return status;
    for(std::size_t i=0;i<inputs.size();++i){
        const auto & input=inputs[i];auto & result=results[i];
        result.prediction=std::move(predictions[i]);
        if(capture_boundaries)result.boundaries=std::move(taps[i]);
        status=decode_soma_playback(result.prediction.features,result.prediction.tokens,*input.skeleton,
            result.alignment,input.model_to_runtime,input.context,input.target,result.playback,reason);
        if(status!=MB_OK)return status;
    }
    outputs=std::move(results);return MB_OK;
}

mb_status decode_soma_playback(std::span<const float> features,unsigned tokens,
    const soma_skeleton & skeleton,const soma_transform & alignment,
    const soma_transform & model_to_runtime,std::span<const soma_transform> context,
    std::span<const soma_transform> target,soma_playback & out,std::string & reason) {
    out={};
    const auto joints=skeleton.parents.size();
    if(tokens<6||tokens>16||features.size()!=tokens*4*366||!joints||
       skeleton.pose_to_model.size()!=joints||skeleton.reference.size()!=joints||
       context.size()!=4*joints||target.empty()||target.size()%joints!=0||
       skeleton.parents[0]!=-1||skeleton.pose_to_model[0]!=0) {
        reason="invalid SOMA playback dimensions";return MB_INVALID_ARGUMENT;
    }
    for(std::size_t j=0;j<joints;++j) {
        const int parent=skeleton.parents[j],model=skeleton.pose_to_model[j];
        if(model < -1 || model>=30 || (j && (parent<0||static_cast<std::size_t>(parent)>=j)) ||
           (j && model>=0 && skeleton.pose_to_model[static_cast<std::size_t>(parent)]<0)) {
            reason="invalid SOMA hierarchy or model mapping";return MB_INVALID_ARGUMENT;
        }
    }
    if(!std::all_of(features.begin(),features.end(),[](float v){return std::isfinite(v);})) {
        reason="nonfinite SOMA playback features";return MB_INVALID_ARGUMENT;
    }
    const auto valid_transform=[](const soma_transform & t) {
        if(!std::all_of(t.begin(),t.end(),[](float v){return std::isfinite(v);}))return false;
        const float norm=t[3]*t[3]+t[4]*t[4]+t[5]*t[5]+t[6]*t[6];
        return std::abs(norm-1)<0.02F;
    };
    if(!valid_transform(alignment)||!valid_transform(model_to_runtime)||
       !std::all_of(context.begin(),context.end(),valid_transform)||
       !std::all_of(target.begin(),target.end(),valid_transform)||
       !std::all_of(skeleton.reference.begin(),skeleton.reference.end(),valid_transform)) {
        reason="invalid SOMA playback transform";return MB_INVALID_ARGUMENT;
    }
    try {
        out.frames=tokens*4-6;
        out.decoded.resize(out.frames*joints);out.contacts.resize(out.frames*4);
        for(unsigned f=0;f<out.frames;++f) {
            const float * row=features.data()+(f+3)*366;
            std::array<quat,30> rotations;
            for(unsigned j=0;j<30;++j)rotations[j]=from_six(row+92+j*6);
            out.decoded[f*joints]=mul(model_to_runtime,mul(alignment,transform({row[0],row[1],row[2]},rotations[0])));
            for(std::size_t j=1;j<joints;++j) {
                auto t=skeleton.reference[j];
                const int model=skeleton.pose_to_model[j];
                if(model>=0) {
                    const int parent=skeleton.pose_to_model[static_cast<std::size_t>(skeleton.parents[j])];
                    t=transform(position(t),normalized(mul(inverse(rotations[parent]),rotations[model])));
                }
                out.decoded[f*joints+j]=t;
            }
            for(unsigned c=0;c<4;++c)out.contacts[f*4+c]=std::clamp(row[362+c],0.0F,1.0F);
        }
        out.start_deltas.resize(joints);out.end_deltas.resize(joints);
        for(std::size_t j=0;j<joints;++j) {
            const auto delta=[&](const soma_transform & src,const soma_transform & dst) {
                if(j)return mul(inverse(src),dst);
                return transform({dst[0]-src[0],dst[1]-src[1],dst[2]-src[2]},
                    normalized(mul(inverse(normalized(rotation(src))),normalized(rotation(dst)))));
            };
            out.start_deltas[j]=delta(out.decoded[j],context[3*joints+j]);
            out.end_deltas[j]=delta(out.decoded[(out.frames-1)*joints+j],target[j]);
        }
        out.corrected=out.decoded;out.world.resize(out.decoded.size());
        for(unsigned f=0;f<out.frames;++f) {
            const float end=static_cast<float>(f)/static_cast<float>(out.frames-1),start=1-end;
            for(std::size_t j=0;j<joints;++j) {
                auto & t=out.corrected[f*joints+j];
                if(j)t=mul(mul(t,weighted(out.start_deltas[j],start)),weighted(out.end_deltas[j],end));
                else {
                    auto p=position(t);
                    for(unsigned i=0;i<3;++i)p[i]=(p[i]+out.start_deltas[0][i]*start)+out.end_deltas[0][i]*end;
                    t=transform(p,normalized(mul(mul(rotation(t),slerp(rotation(out.start_deltas[0]),start)),
                        slerp(rotation(out.end_deltas[0]),end))));
                }
                out.world[f*joints+j]=j?mul(out.world[f*joints+static_cast<std::size_t>(skeleton.parents[j])],t):t;
            }
        }
        return MB_OK;
    } catch(const std::exception & e) {
        out={};reason=e.what();return MB_INVALID_ARGUMENT;
    }
}
}
