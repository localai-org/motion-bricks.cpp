package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"testing"
	"time"

	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
)

func TestStreamPoseBuffer(t *testing.T) {
	chrome, err := exec.LookPath("chromium")
	if err != nil {
		t.Skip("Chromium unavailable")
	}
	s := httptest.NewServer(http.FileServer(http.FS(webFiles)))
	defer s.Close()
	opts := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	opts = append(opts, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true))
	a, ac := chromedp.NewExecAllocator(context.Background(), opts...)
	defer ac()
	b, bc := chromedp.NewContext(a)
	defer bc()
	ctx, cancel := context.WithTimeout(b, 20*time.Second)
	defer cancel()
	var failures []string
	err = chromedp.Run(ctx, chromedp.Navigate(s.URL), chromedp.Evaluate(`(async()=>{
 const {PoseBuffer}=await import('/web/stream-client.js');const failures=[];
 const assert=(value,name)=>{if(!value)failures.push(name)};
 const pose=(i,epoch=1)=>({type:'frame',epoch,tick:i,time:i/50,root:[i/50,1,0],rotations:Array.from({length:136},(_,j)=>j%4===3?1:0),physics:false,paused:false});
 const p=new PoseBuffer();let next=0,previous=0,maxStep=0;
 // Deterministic 0..60ms delivery jitter, 50Hz producer, 60Hz rendering.
 for(let i=0;i<1200;i++){
   const now=i/60;
   while(next/50+((next*17)%7)/100<=now){p.push(pose(next++));}
   const sample=p.sample(1/60);if(!sample)continue;
   const x=sample.a.root[0]*(1-sample.alpha)+sample.b.root[0]*sample.alpha;
   assert(x>=previous-1e-9,'time moved backwards');maxStep=Math.max(maxStep,x-previous);previous=x;
 }
 assert(p.underruns===0,'jitter caused underrun');assert(maxStep<.022,'jitter caused render jump');assert(p.frames.length<20,'unbounded normal queue');
 const before=p.time;for(let i=0;i<60;i++)p.sample(1/60);
 assert(p.waiting&&p.underruns===1&&p.time>=before,'loss does not rebuffer');
 assert(p.time<=p.frames.at(-1).time,'extrapolated missing motion');
 p.push(pose(0,2));assert(p.epoch===2&&p.time===null,'reset not flushed');p.push(pose(999,1));assert(p.epoch===2&&p.frames.length===1,'old epoch resurrected');
 for(let i=1;i<500;i++)p.push(pose(i,2));assert(p.frames.length<=128,'hidden tab queue grew');
 for(const mutation of [f=>f.root[0]=NaN,f=>f.root[0]=Infinity,f=>f.root[1]=1e20,f=>f.rotations[3]=2,f=>f.rotations.pop(),f=>{f.physics=true;f.physical=[]}]){
   const f=pose(501,2);mutation(f);let rejected=false;try{p.push(f)}catch{rejected=true}assert(rejected,'invalid pose accepted');
 }
 const paused=pose(0,3);paused.paused=true;p.push(paused);for(let i=0;i<60;i++)p.sample(1/60);assert(p.time===0&&p.underruns===0,'paused clock advanced');
 for(const c of [[0,0,0,0,0,0,2],[NaN,0,0,0,0,0,1],[1,2]]) {
   let rejected=false;try{p.push({...pose(1,3),collision_transforms:c})}catch{rejected=true}assert(rejected,'invalid collision transform accepted');
 }
 const {CollisionGeometry}=await import('/web/collision-geometry.js');const THREE=await import('/web/vendor/three.module.min.js');
 const overlay=new CollisionGeometry([{type:6,name:'sole',size:[.085,.03,.005]}]);
 overlay.pose([0,.005,0,-.5,-.5,-.5,.5],[1,.005,0,-.5,-.5,-.5,.5],.5);
 overlay.group.updateMatrixWorld(true);const bounds=new THREE.Box3().setFromObject(overlay.meshes[0]);
 assert(Math.abs(bounds.min.x-.47)<1e-7&&Math.abs(bounds.max.x-.53)<1e-7,'sole width/interpolated position');
 assert(Math.abs(bounds.min.y)<1e-7&&Math.abs(bounds.max.y-.01)<1e-7,'sole height/basis');
 assert(Math.abs(bounds.min.z+.085)<1e-7&&Math.abs(bounds.max.z-.085)<1e-7,'sole length/basis');
 assert(overlay.meshes[0].material.opacity===.6&&!overlay.meshes[0].material.depthWrite,'collision transparency');
 assert(overlay.batchMaterials[0].uniforms.opacity.value===.6&&overlay.batchMaterials[1].uniforms.opacity.value===.5,'batched collision transparency');
 overlay.pose([],[],0);assert(!overlay.group.visible,'missing transforms rendered');overlay.dispose();
 const hull=new CollisionGeometry([{type:7,name:'hull',size:[1,1,1],vertices:[0,0,0,1,0,0,0,1,0,0,0,1],indices:[0,2,1,0,1,3,0,3,2,1,2,3]}]);
 assert(hull.meshes[0].geometry.index.count===12&&hull.meshes[0].geometry.attributes.position.count===4,'indexed hull changed topology');hull.dispose();
 let rejected=false;try{new CollisionGeometry([{type:7,size:[1,1,1],vertices:[0,0,0],indices:[0,1,2]}])}catch{rejected=true}assert(rejected,'out-of-range hull index accepted');
 return failures;
})()`, &failures, func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }))
	if err != nil {
		t.Fatal(err)
	}
	if len(failures) > 0 {
		t.Fatal(failures)
	}
}
