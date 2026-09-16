#include <motionbricks/inference.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);return 1;} } while(0)
int main(void) {
    char e[64];mb_inference_request * r=NULL;uint64_t n=0,seed=0;
    float input[2424]={0},copy[2424];uint32_t flags[11],mode=99;
    CHECK(mb_inference_request_create(NULL,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_create(&r,e,sizeof e)==MB_OK);
    CHECK(mb_inference_request_get_seed(r,&seed,e,sizeof e)==MB_OK&&seed==0);
    CHECK(mb_inference_request_set_seed(r,UINT64_MAX,e,sizeof e)==MB_OK);
    CHECK(mb_inference_request_get_seed(r,&seed,e,sizeof e)==MB_OK&&seed==UINT64_MAX);
    CHECK(mb_inference_request_get_sampling_argmax(r,&mode,e,sizeof e)==MB_OK&&mode==0);
    CHECK(mb_inference_request_set_sampling_argmax(r,1,e,sizeof e)==MB_OK);
    CHECK(mb_inference_request_set_sampling_argmax(r,2,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_get_sampling_argmax(r,&mode,e,sizeof e)==MB_OK&&mode==1);
    const uint64_t sizes[3]={40,32,2424};
    for(uint32_t f=0;f<3;++f) {
        CHECK(mb_inference_request_get_features(r,f,NULL,0,&n,e,sizeof e)==MB_OK&&n==sizes[f]);
        CHECK(mb_inference_request_set_features(r,f,input,n,e,sizeof e)==MB_OK);
        input[0]=NAN;
        CHECK(mb_inference_request_set_features(r,f,input,n,e,sizeof e)==MB_INVALID_ARGUMENT);
        input[0]=INFINITY;
        CHECK(mb_inference_request_set_features(r,f,input,n,e,sizeof e)==MB_INVALID_ARGUMENT);
        input[0]=10001;
        CHECK(mb_inference_request_set_features(r,f,input,n,e,sizeof e)==MB_INVALID_ARGUMENT);
        input[0]=0;
        CHECK(mb_inference_request_set_features(r,f,input,UINT64_MAX,e,sizeof e)==MB_INVALID_ARGUMENT);
        CHECK(mb_inference_request_get_features(r,f,copy,n-1,&n,e,sizeof e)==MB_INVALID_ARGUMENT);
        CHECK(mb_inference_request_get_features(r,f,copy,2424,&n,e,sizeof e)==MB_OK);
        CHECK(memcmp(input,copy,(size_t)n*sizeof(float))==0);
    }
    for(uint32_t f=0;f<4;++f) {
        CHECK(mb_inference_request_get_mask(r,f,NULL,0,&n,e,sizeof e)==MB_OK&&n==(f==3?11:8));
        CHECK(mb_inference_request_get_mask(r,f,flags,11,&n,e,sizeof e)==MB_OK);
        CHECK(flags[3]==(f==1?0:1));
        flags[0]=2;
        CHECK(mb_inference_request_set_mask(r,f,flags,n,e,sizeof e)==MB_INVALID_ARGUMENT);
    }
    memset(flags,0,sizeof flags);
    CHECK(mb_inference_request_set_mask(r,3,flags,11,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_set_mask(r,0,flags,8,e,sizeof e)==MB_INVALID_ARGUMENT);
    flags[10]=1;CHECK(mb_inference_request_set_mask(r,3,flags,11,e,sizeof e)==MB_OK);
    CHECK(mb_inference_request_get_mask(r,3,flags,11,&n,e,sizeof e)==MB_OK&&flags[10]==1&&flags[0]==0);
    CHECK(mb_inference_request_set_features(r,3,input,40,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_get_features(r,99,copy,2424,&n,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_get_mask(r,99,flags,11,&n,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_get_seed(NULL,&seed,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_get_seed(r,NULL,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_set_features(NULL,0,input,40,e,sizeof e)==MB_INVALID_ARGUMENT);
    CHECK(mb_inference_request_set_boundary_poses(r,NULL,0,input,12,input,544,e,sizeof e)==MB_INVALID_ARGUMENT);
    mb_motion * output=(mb_motion *)(uintptr_t)1;
    CHECK(mb_model_infer(NULL,r,&output,e,sizeof e)==MB_INVALID_ARGUMENT&&output==NULL);
    CHECK(mb_model_infer(NULL,r,NULL,NULL,0)==MB_INVALID_ARGUMENT);
    {
        const mb_inference_request * requests[1]={r};
        mb_motion * outputs[1]={(mb_motion *)(uintptr_t)1};
        CHECK(mb_model_infer_batch(NULL,requests,1,outputs,e,sizeof e)==MB_INVALID_ARGUMENT);
        CHECK(outputs[0]==NULL);
    }
    char tiny[2]={'a','b'};
    CHECK(mb_inference_request_set_seed(NULL,0,tiny,1)==MB_INVALID_ARGUMENT&&tiny[0]==0&&tiny[1]=='b');
    CHECK(mb_inference_request_set_seed(r,0,e,sizeof e)==MB_OK&&e[0]==0);
    mb_inference_request_free(r);mb_inference_request_free(NULL);return 0;
}
