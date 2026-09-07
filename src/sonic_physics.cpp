// Native simulation wrapper. Observation order and motor equations follow
// NVIDIA's pinned original-release SONIC deployment, independently checked
// by reference/check_sonic_boundaries.py. No DDS, TensorRT or Python runtime.
#include <motionbricks/physics.h>
#include "error.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
#include <mujoco/mujoco.h>
#endif
using motionbricks::detail::guard;
using motionbricks::detail::fail;

#if defined(MOTIONBRICKS_HAVE_MUJOCO)
namespace {
constexpr std::array<int,29> to_isaac{0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28};
constexpr std::array<const char *,29> names{
 "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
 "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
 "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
 "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint",
 "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint"};
struct ModelDelete {void operator()(mjModel * p) const {mj_deleteModel(p);}};
struct DataDelete {void operator()(mjData * p) const {mj_deleteData(p);}};
struct Joint {uint32_t native{}; std::array<double,17> values{}; int q{},v{},actuator{},body{};};
using Pose=std::array<double,36>;
struct History {std::array<float,3> angular{},gravity{}; std::array<float,29> q{},dq{},action{};};
bool valid_pose(const float * roots,uint64_t nr,const float * rotations,uint64_t nq) {
    if(!std::all_of(roots,roots+nr,[](float x){return std::isfinite(x) && std::abs(x)<=1e6F;})) return false;
    for(uint64_t i=0;i<nq;i+=4) {
        double norm=0;for(uint64_t j=0;j<4;++j) norm+=double(rotations[i+j])*rotations[i+j];
        if(!std::isfinite(norm) || norm<.25 || norm>2.25) return false;
    }
    return true;
}
void quaternion(const float * a,double * q) {
    q[0]=a[3]; q[1]=a[0]; q[2]=a[1]; q[3]=a[2];
    double norm=mju_norm(q,4); if(norm<.5 || norm>1.5) throw std::invalid_argument("invalid native quaternion");
    mju_normalize4(q);
}
}
struct mb_physics {
    mb_sonic * sonic{};
    std::unique_ptr<mjModel,ModelDelete> model;
    std::unique_ptr<mjData,DataDelete> data,reference;
    std::array<Joint,29> joints{};
    std::array<int,30> bodies{},parents{};
    std::array<History,10> history{};
    std::array<float,29> last_action{};
    std::array<std::vector<double>,6> trace;
    std::vector<int> collision_geoms;
    bool started=false,fallen=false;
    double heading=0;

    Pose convert(const float * root,const float * rotations) const {
        Pose p{}; p[0]=root[2]; p[1]=root[0]; p[2]=root[1];
        double native_q[4]; quaternion(rotations,native_q);
        // Cyclic proper coordinate rotation: native XYZ -> MuJoCo ZXY.
        p[3]=native_q[0]; p[4]=native_q[3]; p[5]=native_q[1]; p[6]=native_q[2];
        for(size_t j=0;j<29;++j) {
            double q[4],r[9],corrected[9]; quaternion(rotations+4*joints[j].native,q); mju_quat2Mat(r,q);
            mju_mulMatMat(corrected,joints[j].values.data()+3,r,3,3,3);
            const double angles[3]{std::atan2(corrected[7],corrected[8]),std::atan2(corrected[2],corrected[0]),std::atan2(corrected[3],corrected[4])};
            for(int axis=0;axis<3;++axis) p[7+j]+=angles[axis]*joints[j].values[static_cast<size_t>(axis)];
        }
        return p;
    }
    Pose sample(const float * roots,const float * rotations,uint32_t frames,double t) const {
        const double frame=std::clamp(t*30.,0.,double(frames-1)); const auto i=static_cast<uint32_t>(frame);
        const uint32_t next=std::min(i+1,frames-1); const double u=frame-i;
        const auto a=convert(roots+3*i,rotations+136*i),b=convert(roots+3*next,rotations+136*next);
        Pose out=a;
        for(size_t k=0;k<3;++k) out[k]=a[k]+u*(b[k]-a[k]);
        double qb[4]; std::copy_n(b.data()+3,4,qb); if(mju_dot(a.data()+3,qb,4)<0) mju_scl(qb,qb,-1,4);
        double angle=std::acos(std::clamp(mju_dot(a.data()+3,qb,4),-1.,1.));
        for(size_t k=0;k<4;++k) out[3+k]=angle<1e-8?a[3+k]+u*(qb[k]-a[3+k]):(std::sin((1-u)*angle)*a[3+k]+std::sin(u*angle)*qb[k])/std::sin(angle);
        mju_normalize4(out.data()+3);
        for(size_t k=7;k<36;++k) out[k]=a[k]+u*std::remainder(b[k]-a[k],2*3.14159265358979323846);
        return out;
    }
    void set_pose(mjData * d,const Pose & p) {
        std::copy_n(p.data(),7,d->qpos);
        for(size_t j=0;j<29;++j) d->qpos[joints[j].q]=p[7+j];
        mj_kinematics(model.get(),d);
    }
    void positions(mjData * d,float * out) {
        mj_kinematics(model.get(),d);
        for(size_t j=0;j<30;++j) {
            const auto * p=d->xpos+3*bodies[j]; out[3*j]=float(p[1]);out[3*j+1]=float(p[2]);out[3*j+2]=float(p[0]);
        }
    }
    void tick(const float * roots,const float * rotations,uint32_t frames,double t) {
        for(auto & field:trace) field.clear();
        auto & state=trace[4];
        for(const auto & j:joints) state.push_back(data->qpos[j.q]);
        for(const auto & j:joints) state.push_back(data->qvel[j.v]);
        state.insert(state.end(),data->qpos+3,data->qpos+7);
        state.insert(state.end(),data->qvel+3,data->qvel+6);
        std::move(history.begin()+1,history.end(),history.begin());
        History h{};
        // Upstream DDS publishes F32 sensor values before conversion to policy
        // doubles. Reproduce those roundings at the observation boundary.
        double base[4]; for(int k=0;k<4;++k) base[k]=float(data->qpos[3+k]);
        double inverse[4]; mju_negQuat(inverse,base);
        double gravity[3]{0,0,-1},g[3]; mju_rotVecQuat(g,gravity,inverse);
        for(size_t k=0;k<3;++k) {h.angular[k]=float(data->qvel[3+k]);h.gravity[k]=float(g[k]);}
        for(size_t i=0;i<29;++i) {
            const auto & j=joints[static_cast<size_t>(to_isaac[i])];
            h.q[i]=float(double(float(data->qpos[j.q]))-j.values[12]); h.dq[i]=float(data->qvel[j.v]);
        }
        h.action=last_action;history.back()=h;
        std::array<float,1762> enc{}; std::array<float,994> dec{};
        double align[4]{std::cos(heading/2),0,0,std::sin(heading/2)};
        for(size_t f=0;f<10;++f) {
            double time=t+double(f)*.1; auto p=sample(roots,rotations,frames,time),next=sample(roots,rotations,frames,time+.02);
            for(size_t i=0;i<29;++i) {
                const auto j=static_cast<size_t>(to_isaac[i]);
                enc[4+f*29+i]=float(p[7+j]); enc[294+f*29+i]=float((next[7+j]-p[7+j])*50);
            }
            double q[4],relative[4],mat[9];mju_mulQuat(q,align,p.data()+3);mju_mulQuat(relative,inverse,q);mju_quat2Mat(mat,relative);
            for(size_t row=0;row<3;++row) for(size_t col=0;col<2;++col) enc[601+f*6+row*2+col]=float(mat[row*3+col]);
        }
        for(size_t f=0;f<10;++f) {
            const auto & v=history[f];std::copy(v.angular.begin(),v.angular.end(),dec.begin()+64+f*3);
            std::copy(v.q.begin(),v.q.end(),dec.begin()+94+f*29);std::copy(v.dq.begin(),v.dq.end(),dec.begin()+384+f*29);
            std::copy(v.action.begin(),v.action.end(),dec.begin()+674+f*29);std::copy(v.gravity.begin(),v.gravity.end(),dec.begin()+964+f*3);
        }
        char error[1024]{};
        if(mb_sonic_encode(sonic,enc.data(),enc.size(),dec.data(),64,error,sizeof(error))!=MB_OK ||
           mb_sonic_decode(sonic,dec.data(),dec.size(),last_action.data(),29,error,sizeof(error))!=MB_OK) throw std::runtime_error(error);
        std::array<float,29> target{};
        for(size_t i=0;i<29;++i) {size_t j=static_cast<size_t>(to_isaac[i]);target[j]=float(joints[j].values[12]+joints[j].values[13]*last_action[i]);}
        trace[0].assign(enc.begin(),enc.end());trace[1].assign(dec.begin(),dec.end());
        trace[2].assign(last_action.begin(),last_action.end());trace[3].assign(target.begin(),target.end());
        for(int step=0;step<4;++step) {
            for(const auto & j:joints) trace[5].push_back(data->qpos[j.q]);
            for(const auto & j:joints) trace[5].push_back(data->qvel[j.v]);
            mju_zero(data->ctrl,model->nu);
            for(size_t j=0;j<29;++j) {
                const auto & c=joints[j]; double tau=c.values[14]*(target[j]-data->qpos[c.q])-c.values[15]*data->qvel[c.v];
                data->ctrl[c.actuator]=std::clamp(tau,-c.values[16],c.values[16]);
                trace[5].push_back(data->ctrl[c.actuator]);
            }
            mj_step(model.get(),data.get());
            if(data->warning[mjWARN_BADQPOS].number || data->warning[mjWARN_BADQVEL].number ||
               data->warning[mjWARN_BADQACC].number || data->warning[mjWARN_BADCTRL].number)
                throw std::runtime_error("MuJoCo rejected unstable physical state; simulation stopped");
            if(!std::all_of(data->qpos,data->qpos+model->nq,[](double x){return std::isfinite(x);}) ||
               !std::all_of(data->qvel,data->qvel+model->nv,[](double x){return std::isfinite(x);})) throw std::runtime_error("non-finite physical state");
            // Diagnostic only: keep running the policy and every physics
            // substep after a fall. Numerical instability still throws above.
            if(data->qpos[2]<.2) fallen=true;
        }
    }
};
#else
struct mb_physics {};
#endif

extern "C" {
mb_status mb_physics_create(mb_sonic * sonic,const char * scene,const char * config,mb_physics ** output,char * error,uint64_t cap) {
    if(output) *output=nullptr;
    return guard(error,cap,[&]()->mb_status {
        if(!sonic || !scene || !config || !output) return fail(MB_INVALID_ARGUMENT,error,cap,"physics model/scene/config/output required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        auto s=std::make_unique<mb_physics>();s->sonic=sonic;
        std::ifstream f(config,std::ios::binary);char magic[8]{};f.read(magic,8);
        if(!f || std::memcmp(magic,"MBSPHY01",8)) return fail(MB_INVALID_FORMAT,error,cap,"invalid physical config header");
        std::array<bool,34> used{};
        for(auto & j:s->joints) {
            f.read(reinterpret_cast<char*>(&j.native),4);f.read(reinterpret_cast<char*>(j.values.data()),17*8);
            if(!f || j.native==0 || j.native>=34 || used[j.native] || !std::all_of(j.values.begin(),j.values.end(),[](double v){return std::isfinite(v);}) ||
               j.values[13]<=0 || j.values[13]>10 || j.values[14]<=0 || j.values[14]>1000 || j.values[15]<0 || j.values[15]>100 || j.values[16]<=0 || j.values[16]>1000)
                return fail(MB_INVALID_FORMAT,error,cap,"invalid physical joint config");
            used[j.native]=true;
            double axis_norm=0;for(size_t i=0;i<3;++i) axis_norm+=j.values[i]*j.values[i];
            if(std::abs(axis_norm-1)>1e-6 || std::abs(j.values[12])>6.4) return fail(MB_INVALID_FORMAT,error,cap,"invalid axis/default angle");
            for(size_t i=0;i<3;++i) if(j.values[i]!=0 && j.values[i]!=1) return fail(MB_INVALID_FORMAT,error,cap,"requires positive coordinate hinge axes");
            for(size_t r=0;r<3;++r) for(size_t c=0;c<3;++c) {
                double dot=0;for(size_t k=0;k<3;++k) dot+=j.values[3+r*3+k]*j.values[3+c*3+k];
                if(std::abs(dot-(r==c?1:0))>1e-6) return fail(MB_INVALID_FORMAT,error,cap,"non-orthonormal joint rest rotation");
            }
            const auto * r=j.values.data()+3;
            const double determinant=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]);
            if(std::abs(determinant-1)>1e-6) return fail(MB_INVALID_FORMAT,error,cap,"reflected joint rest rotation");
        }
        if(f.peek()!=std::char_traits<char>::eof()) return fail(MB_INVALID_FORMAT,error,cap,"trailing physical config data");
        char message[1024]{};s->model.reset(mj_loadXML(scene,nullptr,message,sizeof(message)));
        if(!s->model) return fail(MB_INVALID_FORMAT,error,cap,message);
        auto * m=s->model.get();
        if(m->nq!=50 || m->nv!=49 || m->nu!=43 || mj_version()!=335) return fail(MB_INVALID_FORMAT,error,cap,"requires original 43-actuator G1 and MuJoCo 3.3.5");
        for(int g=0;g<m->ngeom;++g) {
            bool enabled=m->geom_contype[g] || m->geom_conaffinity[g];
            for(int p=0;p<m->npair;++p) enabled=enabled || m->pair_geom1[p]==g || m->pair_geom2[p]==g;
            if(m->geom_bodyid[g]!=0 && enabled) s->collision_geoms.push_back(g);
        }
        m->opt.timestep=.005;s->data.reset(mj_makeData(m));s->reference.reset(mj_makeData(m));
        if(!s->data || !s->reference) throw std::bad_alloc();
        for(size_t k=0;k<29;++k) {
            auto & j=s->joints[k];int id=mj_name2id(m,mjOBJ_JOINT,names[k]);
            if(id<0 || m->jnt_type[id]!=mjJNT_HINGE) return fail(MB_INVALID_FORMAT,error,cap,"missing G1 hinge");
            j.q=m->jnt_qposadr[id];j.v=m->jnt_dofadr[id];j.body=m->jnt_bodyid[id];j.actuator=-1;
            for(int a=0;a<m->nu;++a) if(m->actuator_trnid[a*2]==id) j.actuator=a;
            if(j.actuator<0) return fail(MB_INVALID_FORMAT,error,cap,"missing G1 actuator");
        }
        s->bodies[0]=mj_name2id(m,mjOBJ_BODY,"pelvis");
        for(size_t i=0;i<29;++i) s->bodies[i+1]=s->joints[static_cast<size_t>(to_isaac[i])].body;
        s->parents[0]=-1;
        for(size_t i=1;i<30;++i) {
            int parent=m->body_parentid[s->bodies[i]];auto found=std::find(s->bodies.begin(),s->bodies.begin()+static_cast<ptrdiff_t>(i),parent);
            if(found==s->bodies.begin()+static_cast<ptrdiff_t>(i)) return fail(MB_INVALID_FORMAT,error,cap,"invalid G1 body tree");
            s->parents[i]=int(found-s->bodies.begin());
        }
        *output=s.release();return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"build with MOTIONBRICKS_ENABLE_PHYSICS and MuJoCo");
#endif
    });
}
void mb_physics_free(mb_physics * s) {delete s;}
mb_status mb_physics_reset(mb_physics * s,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s) return fail(MB_INVALID_ARGUMENT,error,cap,"physical session required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        mj_resetData(s->model.get(),s->data.get());mj_resetData(s->model.get(),s->reference.get());
        s->history={};s->last_action={};s->started=false;s->fallen=false;s->heading=0;
        for(auto & field:s->trace) field.clear();
        return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_start(mb_physics * s,const float * root,uint64_t nr,const float * rot,uint64_t nq,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s || !root || !rot || nr!=3 || nq!=136) return fail(MB_INVALID_ARGUMENT,error,cap,"invalid initial pose buffers");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        if(s->started || !valid_pose(root,nr,rot,nq)) return fail(MB_INVALID_ARGUMENT,error,cap,"session already started or invalid pose");
        auto pose=s->convert(root,rot);s->heading=0;
        // Initial alignment only. Begin in the supplied reference pose;
        // subsequent ticks are exclusively actuator-driven integration.
        s->set_pose(s->data.get(),pose);mj_forward(s->model.get(),s->data.get());
        for(auto & h:s->history) h.gravity={0,0,1}; // upstream zero-quaternion history padding
        s->started=true;return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_step(mb_physics * s,const float * roots,uint64_t nr,const float * rot,uint64_t nq,uint32_t frames,double t,
    float * actual,uint64_t na,float * reference,uint64_t nb,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s || !roots || !rot || !actual || !reference || frames<2 || frames>1800 || nr!=uint64_t(frames)*3 || nq!=uint64_t(frames)*136 || na!=90 || nb!=90 || !std::isfinite(t) || t<0 || t>120)
            return fail(MB_INVALID_ARGUMENT,error,cap,"invalid physical step buffers/time");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        if(!s->started || !valid_pose(roots,nr,rot,nq)) return fail(MB_INVALID_ARGUMENT,error,cap,"session not started or invalid reference pose");
        s->tick(roots,rot,frames,t);
        s->positions(s->data.get(),actual);s->set_pose(s->reference.get(),s->sample(roots,rot,frames,t+.02));s->positions(s->reference.get(),reference);return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_skeleton(mb_physics * s,int32_t * parents,uint64_t count,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s || !parents || count!=30) return fail(MB_INVALID_ARGUMENT,error,cap,"30 parent entries required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        std::copy(s->parents.begin(),s->parents.end(),parents);return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_status(mb_physics * s,double * time,uint32_t * fallen,uint32_t * contacts,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s || !time || !fallen || !contacts) return fail(MB_INVALID_ARGUMENT,error,cap,"status outputs required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        *time=s->data->time;*fallen=s->fallen;*contacts=static_cast<uint32_t>(s->data->ncon);return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_collision_count(mb_physics * s,uint32_t * count,char * error,uint64_t cap) {
    if(count) *count=0;
    return guard(error,cap,[&]()->mb_status {
        if(!s || !count) return fail(MB_INVALID_ARGUMENT,error,cap,"collision count output required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        *count=static_cast<uint32_t>(s->collision_geoms.size());return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_collision_shape(mb_physics * s,uint32_t index,uint32_t * type,float * size,uint64_t n,char * name,uint64_t nc,char * error,uint64_t cap) {
    if(name && nc) name[0]=0;
    return guard(error,cap,[&]()->mb_status {
        if(!s || !type || !size || n!=3 || !name || !nc) return fail(MB_INVALID_ARGUMENT,error,cap,"collision shape outputs required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        if(index>=s->collision_geoms.size()) return fail(MB_INVALID_ARGUMENT,error,cap,"collision index out of range");
        const int g=s->collision_geoms[index];const auto * m=s->model.get();
        const char * label=mj_id2name(m,mjOBJ_BODY,m->geom_bodyid[g]);if(!label) label="unnamed";
        if(std::strlen(label)+1>nc) return fail(MB_INVALID_ARGUMENT,error,cap,"collision name capacity too small");
        *type=static_cast<uint32_t>(m->geom_type[g]);
        for(int k=0;k<3;++k) size[k]=float(m->geom_size[g*3+k]);
        std::memcpy(name,label,std::strlen(label)+1);return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_collision_triangles(mb_physics * s,uint32_t index,float * xyz,uint64_t capacity,uint64_t * count,char * error,uint64_t cap) {
    if(count) *count=0;
    return guard(error,cap,[&]()->mb_status {
        if(!s || !count || (!xyz && capacity)) return fail(MB_INVALID_ARGUMENT,error,cap,"invalid collision mesh request");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        if(index>=s->collision_geoms.size()) return fail(MB_INVALID_ARGUMENT,error,cap,"collision index out of range");
        const auto * m=s->model.get();const int g=s->collision_geoms[index];
        if(m->geom_type[g]!=mjGEOM_MESH) return MB_OK;
        const int mesh=m->geom_dataid[g];
        if(mesh<0 || mesh>=m->nmesh || m->mesh_graphadr[mesh]<0) return fail(MB_INVALID_FORMAT,error,cap,"collision mesh has no compiled convex hull");
        const uint64_t base=static_cast<uint64_t>(m->mesh_graphadr[mesh]);
        if(base+2>static_cast<uint64_t>(m->nmeshgraph)) return fail(MB_INVALID_FORMAT,error,cap,"invalid convex graph header");
        const int nv=m->mesh_graph[base],nf=m->mesh_graph[base+1];
        if(nv<0 || nf<0) return fail(MB_INVALID_FORMAT,error,cap,"invalid convex graph sizes");
        // Packed hull-face layout follows MuJoCo 3.3.5 mjr_uploadMesh:
        // https://github.com/google-deepmind/mujoco/blob/3.3.5/src/render/render_context.c
        const uint64_t faces=base+2+3ULL*static_cast<uint64_t>(nv)+3ULL*static_cast<uint64_t>(nf);
        if(faces+3ULL*static_cast<uint64_t>(nf)>static_cast<uint64_t>(m->nmeshgraph)) return fail(MB_INVALID_FORMAT,error,cap,"invalid convex graph faces");
        *count=9ULL*static_cast<uint64_t>(nf);
        if(!xyz && !capacity) return MB_OK;
        if(capacity<*count) return fail(MB_INVALID_ARGUMENT,error,cap,"collision mesh capacity too small");
        for(uint64_t v=0;v<*count/3;++v) {
            const int vertex=m->mesh_graph[faces+v];
            if(vertex<0 || vertex>=m->mesh_vertnum[mesh]) return fail(MB_INVALID_FORMAT,error,cap,"invalid hull vertex");
            std::copy_n(m->mesh_vert+3*(m->mesh_vertadr[mesh]+vertex),3,xyz+3*v);
        }
        return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_collision_transforms(mb_physics * s,float * transforms,uint64_t count,char * error,uint64_t cap) {
    return guard(error,cap,[&]()->mb_status {
        if(!s || (!transforms && count)) return fail(MB_INVALID_ARGUMENT,error,cap,"collision transform output required");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        if(!s->started || count!=s->collision_geoms.size()*7) return fail(MB_INVALID_ARGUMENT,error,cap,"collision transform shape/session invalid");
        mj_kinematics(s->model.get(),s->data.get());
        for(size_t i=0;i<s->collision_geoms.size();++i) {
            const int g=s->collision_geoms[i];const auto * p=s->data->geom_xpos+3*g;const auto * r=s->data->geom_xmat+9*g;
            double basis[9],q[4];
            for(int row=0;row<3;++row) for(int col=0;col<3;++col) basis[row*3+col]=r[((row+1)%3)*3+col];
            mju_mat2Quat(q,basis);
            float * out=transforms+7*i;
            out[0]=float(p[1]);out[1]=float(p[2]);out[2]=float(p[0]);
            for(int k=0;k<3;++k) out[3+k]=float(q[1+k]);
            out[6]=float(q[0]);
            if(!std::all_of(out,out+7,[](float x){return std::isfinite(x);})) return fail(MB_INVALID_FORMAT,error,cap,"non-finite collision transform");
        }
        return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
mb_status mb_physics_trace(mb_physics * s,uint32_t field,double * data,uint64_t capacity,uint64_t * count,char * error,uint64_t cap) {
    if(count) *count=0;
    return guard(error,cap,[&]()->mb_status {
        if(!s || field>=6 || !count || (!data && capacity)) return fail(MB_INVALID_ARGUMENT,error,cap,"invalid physical trace request");
#if defined(MOTIONBRICKS_HAVE_MUJOCO)
        const auto & values=s->trace[field];
        if(values.empty()) return fail(MB_INVALID_ARGUMENT,error,cap,"physical trace unavailable");
        *count=values.size();if(!data && !capacity) return MB_OK;
        if(capacity<*count) return fail(MB_INVALID_ARGUMENT,error,cap,"physical trace capacity too small");
        std::copy(values.begin(),values.end(),data);return MB_OK;
#else
        return fail(MB_BACKEND_UNAVAILABLE,error,cap,"MuJoCo unavailable");
#endif
    });
}
}
