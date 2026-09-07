#include <motionbricks/physics.h>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#if defined(MOTIONBRICKS_TEST_COLLISIONS)
void check_collision_geometry(mb_physics *,const char *,const float *,const float *);
#endif
int main(int argc,char ** argv) {
    char error[64]{};mb_sonic * model=nullptr;
    assert(mb_sonic_load(nullptr,nullptr,&model,error,sizeof(error))==MB_INVALID_ARGUMENT && !model);
    std::array<float,1762> enc{};std::array<float,994> dec{};std::array<float,64> tokens{};std::array<float,29> actions{};
    assert(mb_sonic_encode(nullptr,enc.data(),1762,tokens.data(),64,error,sizeof(error))==MB_INVALID_ARGUMENT);
    mb_physics * physics=nullptr;
    uint32_t collision_count=42;
    assert(mb_physics_collision_count(nullptr,&collision_count,error,sizeof(error))==MB_INVALID_ARGUMENT && collision_count==0);
    assert(mb_physics_collision_transforms(nullptr,nullptr,0,error,sizeof(error))==MB_INVALID_ARGUMENT);
    assert(mb_physics_reset(nullptr,error,sizeof(error))==MB_INVALID_ARGUMENT);
    assert(mb_physics_create(nullptr,nullptr,nullptr,&physics,error,sizeof(error))==MB_INVALID_ARGUMENT && !physics);
    mb_physics_free(nullptr);mb_sonic_free(nullptr);
    if(argc<2) return 0;
    assert(mb_sonic_load(argv[1],nullptr,&model,error,sizeof(error))==MB_OK);
    uint64_t size=9;
    assert(mb_sonic_layer(model,1,0,nullptr,0,&size,error,sizeof(error))==MB_INVALID_ARGUMENT);
    assert(mb_sonic_encode(model,enc.data(),1761,tokens.data(),64,error,sizeof(error))==MB_INVALID_ARGUMENT);
    enc[0]=1;assert(mb_sonic_encode(model,enc.data(),1762,tokens.data(),64,error,sizeof(error))==MB_INVALID_ARGUMENT);
    enc[0]=0;enc[12]=std::numeric_limits<float>::quiet_NaN();
    assert(mb_sonic_encode(model,enc.data(),1762,tokens.data(),64,error,sizeof(error))==MB_INVALID_ARGUMENT);
    enc[12]=std::numeric_limits<float>::max();
    assert(mb_sonic_encode(model,enc.data(),1762,tokens.data(),64,error,sizeof(error))==MB_INVALID_ARGUMENT);
    enc[12]=0;assert(mb_sonic_encode(model,enc.data(),1762,tokens.data(),64,error,sizeof(error))==MB_OK);
    assert(mb_sonic_layer(model,1,0,nullptr,0,&size,error,sizeof(error))==MB_OK && size==2048);
    assert(mb_sonic_layer(model,1,0,tokens.data(),64,&size,error,sizeof(error))==MB_INVALID_ARGUMENT);
    dec[7]=std::numeric_limits<float>::infinity();
    assert(mb_sonic_decode(model,dec.data(),994,actions.data(),29,error,sizeof(error))==MB_INVALID_ARGUMENT);
    // Regression: libFuzzer reached a GGML SiLU NaN assertion with huge but
    // finite floats. This is an input error, never a process-level abort.
    dec[7]=-std::numeric_limits<float>::max();
    assert(mb_sonic_decode(model,dec.data(),994,actions.data(),29,error,sizeof(error))==MB_INVALID_ARGUMENT);
    dec[7]=0;assert(mb_sonic_decode(model,dec.data(),994,actions.data(),29,error,sizeof(error))==MB_OK);
    if(argc>=4) {
        assert(mb_physics_create(model,argv[2],"missing-physical-config",&physics,error,sizeof(error))==MB_INVALID_FORMAT && !physics);
        assert(mb_physics_create(model,argv[2],argv[3],&physics,error,sizeof(error))==MB_OK);
        std::array<float,6> root{0,1,0,0,1,0};std::array<float,272> rotations{};
        std::array<float,90> actual{},reference{};std::array<double,1762> trace{};
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_INVALID_ARGUMENT);
        for(size_t i=3;i<rotations.size();i+=4) rotations[i]=1;
        root[0]=std::numeric_limits<float>::max();
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_INVALID_ARGUMENT);
        root[0]=0;
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_OK);
#if defined(MOTIONBRICKS_TEST_COLLISIONS)
        check_collision_geometry(physics,argv[2],root.data(),rotations.data());
        assert(mb_physics_reset(physics,error,sizeof(error))==MB_OK);
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_OK);
#endif
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_INVALID_ARGUMENT);
        assert(mb_physics_trace(physics,0,nullptr,0,&size,error,sizeof(error))==MB_INVALID_ARGUMENT);
        assert(mb_physics_step(physics,root.data(),6,rotations.data(),272,2,-1,actual.data(),90,reference.data(),90,error,sizeof(error))==MB_INVALID_ARGUMENT);
        assert(mb_physics_step(physics,root.data(),6,rotations.data(),272,2,0,actual.data(),90,reference.data(),90,error,sizeof(error))==MB_OK);
        assert(mb_physics_trace(physics,0,nullptr,0,&size,error,sizeof(error))==MB_OK && size==1762);
        assert(mb_physics_trace(physics,0,trace.data(),64,&size,error,sizeof(error))==MB_INVALID_ARGUMENT);
        assert(mb_physics_trace(physics,0,trace.data(),trace.size(),&size,error,sizeof(error))==MB_OK);
        for(float v:actual) assert(std::isfinite(v));
        double time=0;uint32_t fallen=0,contacts=0;
        assert(mb_physics_status(physics,&time,&fallen,&contacts,error,sizeof(error))==MB_OK && std::abs(time-.02)<1e-10);
        assert(mb_physics_reset(physics,error,sizeof(error))==MB_OK);
        assert(mb_physics_status(physics,&time,&fallen,&contacts,error,sizeof(error))==MB_OK && time==0 && fallen==0);
        assert(mb_physics_start(physics,root.data(),3,rotations.data(),136,error,sizeof(error))==MB_OK);
        mb_physics_free(physics);
    }
    mb_sonic_free(model);std::cout<<"SONIC ABI validation passed\n";
}
