// Independent MuJoCo instance: compare exported geometry frames and a full tick
// using captured native motor torques, including compiled mesh offsets and feet.
#include <motionbricks/physics.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

void check_collision_geometry(mb_physics * physics,const char * scene,const float * roots,const float * rotations) {
    char error[1024]{};uint32_t count=0;
    assert(mb_physics_collision_count(physics,&count,error,sizeof(error))==MB_OK && count>2 && count<1024);
    std::vector<float> before(count*7),after(count*7);
    assert(mb_physics_collision_transforms(physics,before.data(),before.size()-1,error,sizeof(error))==MB_INVALID_ARGUMENT);
    assert(mb_physics_collision_transforms(physics,before.data(),before.size(),error,sizeof(error))==MB_OK);
    std::array<float,90> actual{},reference{};
    assert(mb_physics_step(physics,roots,6,rotations,272,2,0,actual.data(),90,reference.data(),90,error,sizeof(error))==MB_OK);
    assert(mb_physics_collision_transforms(physics,after.data(),after.size(),error,sizeof(error))==MB_OK);
    std::array<double,65> state{};std::array<double,348> trace{};uint64_t n=0;
    assert(mb_physics_trace(physics,4,state.data(),state.size(),&n,error,sizeof(error))==MB_OK && n==65);
    assert(mb_physics_trace(physics,5,trace.data(),trace.size(),&n,error,sizeof(error))==MB_OK && n==348);
    mjModel * model=mj_loadXML(scene,nullptr,error,sizeof(error));assert(model);
    mjData * data=mj_makeData(model);assert(data);model->opt.timestep=.005;
    data->qpos[0]=roots[2];data->qpos[1]=roots[0];data->qpos[2]=roots[1];
    std::copy_n(state.data()+58,4,data->qpos+3);
    for(int j=0;j<29;++j) {
        const int actuator=j<22?j:j+7; // Original XML inserts seven left finger motors.
        const int joint=model->actuator_trnid[actuator*2];
        data->qpos[model->jnt_qposadr[joint]]=state[j];
        data->qvel[model->jnt_dofadr[joint]]=state[29+j];
    }
    mj_forward(model,data);
    std::vector<int> geoms;
    for(int g=0;g<model->ngeom;++g) if(model->geom_bodyid[g] && (model->geom_contype[g] || model->geom_conaffinity[g])) geoms.push_back(g);
    assert(geoms.size()==count);
    double maximum=0;
    const auto compare=[&](const std::vector<float>& transforms) {
        mj_kinematics(model,data);
        for(size_t i=0;i<geoms.size();++i) {
            const int g=geoms[i];const float * out=transforms.data()+i*7;
            double q[4]{out[6],out[3],out[4],out[5]};
            // Origin plus all local unit axes: detects scale/basis errors,
            // missing geom offsets, and wrong quaternion component ordering.
            for(int axis=-1;axis<3;++axis) {
                double local[3]{},rotated[3]{};if(axis>=0)local[axis]=1;
                mju_rotVecQuat(rotated,local,q);
                for(int row=0;row<3;++row) {
                    const int nativeRow=(row+1)%3;
                    const double expected=data->geom_xpos[g*3+nativeRow]+(axis<0?0:data->geom_xmat[g*9+nativeRow*3+axis]);
                    maximum=std::max(maximum,std::abs(out[row]+rotated[row]-expected));
                }
            }
        }
    };
    compare(before);
    for(int step=0;step<4;++step) {
        mju_zero(data->ctrl,model->nu);
        for(int j=0;j<29;++j)data->ctrl[j<22?j:j+7]=trace[step*87+58+j];
        mj_step(model,data);
    }
    compare(after);assert(maximum<2e-6);
    int soles=0,meshes=0;
    for(uint32_t i=0;i<count;++i) {
        uint32_t type=0;float size[3]{};char name[256]{};
        assert(mb_physics_collision_shape(physics,i,&type,size,3,name,sizeof(name),error,sizeof(error))==MB_OK);
        assert(type==static_cast<uint32_t>(model->geom_type[geoms[i]]));
        if(type==6 && std::strstr(name,"ankle_roll")) {
            assert(std::abs(size[0]-.085)<1e-7 && std::abs(size[1]-.03)<1e-7 && std::abs(size[2]-.005)<1e-7);++soles;
        }
        assert(mb_physics_collision_triangles(physics,i,nullptr,0,&n,error,sizeof(error))==MB_OK);
        if(type==7) {
            assert(n>0 && n%9==0);std::vector<float> vertices(n);
            assert(mb_physics_collision_triangles(physics,i,vertices.data(),n-1,&n,error,sizeof(error))==MB_INVALID_ARGUMENT);
            assert(mb_physics_collision_triangles(physics,i,vertices.data(),vertices.size(),&n,error,sizeof(error))==MB_OK);
            for(float v:vertices)assert(std::isfinite(v));++meshes;
        }else assert(n==0);
        assert(mb_physics_collision_shape(physics,count,&type,size,3,name,sizeof(name),error,sizeof(error))==MB_INVALID_ARGUMENT);
    }
    assert(soles==2 && meshes>0);
    mj_deleteData(data);mj_deleteModel(model);
    std::cout<<"Collision geometry: "<<count<<" shapes, "<<meshes<<" hulls, two exact foot soles; world-frame max error "<<maximum<<"\n";
}
