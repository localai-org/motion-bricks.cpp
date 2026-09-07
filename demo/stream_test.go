package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/chromedp/cdproto/page"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

func testMotion(frames int) *mb.Motion {
	m := &mb.Motion{Frames: uint64(frames), Joints: 34, Roots: make([]float32, frames*3), Rotations: make([]float32, frames*136)}
	for i := 0; i < frames; i++ {
		m.Roots[i*3] = float32(i) / 30
		m.Roots[i*3+1] = 1
		for j := 0; j < 34; j++ {
			m.Rotations[i*136+j*4+3] = 1
		}
	}
	return m
}
func TestStreamTimeline(t *testing.T) {
	r := referenceTimeline{motion: testMotion(60)}
	context := r.context(16)
	if context.Frames != 4 || math.Abs(float64(context.Roots[0])-16./30) > 1e-6 {
		t.Fatal(context)
	}
	old, _ := r.sample(16)
	next := testMotion(40)
	for i := 0; i < 40; i++ {
		next.Roots[i*3] += old[0] + .1
	}
	if err := r.replace(16, next, 5, true); err != nil {
		t.Fatal(err)
	}
	actual, _ := r.sample(16)
	if actual[0] != old[0] {
		t.Fatal("seam moved", old, actual)
	}
	if err := r.replace(4, testMotion(40), 5, true); err == nil {
		t.Fatal("accepted retroactive plan")
	}
	next = testMotion(10)
	next.Roots[6] = 100
	if validateStreamMotion(next) == nil {
		t.Fatal("accepted root teleport")
	}
	next = testMotion(10)
	next.Rotations[5] = float32(math.NaN())
	if validateStreamMotion(next) == nil {
		t.Fatal("accepted NaN")
	}
}

func TestStreamBrowser(t *testing.T) {
	root := os.Getenv("MOTIONBRICKS_STREAM_TEST_ROOT")
	if root == "" {
		t.Skip("set MOTIONBRICKS_STREAM_TEST_ROOT for native streaming QA")
	}
	for _, delay := range []int{0, 100} {
		t.Run(fmt.Sprint(delay), func(t *testing.T) {
			lib := filepath.Join(root, "build/debug/libmotionbricks.so")
			s, err := loadDemoServer(lib, filepath.Join(root, "generated/g1-f32"), filepath.Join(root, "generated/styles"), mb.DeviceVulkan)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			if err = s.loadKimodo(os.Getenv("MOTIONBRICKS_STREAM_KIMODO")); err != nil {
				t.Fatal(err)
			}
			p, err := mb.OpenPhysics(lib, filepath.Join(root, "generated/sonic/ggml/sonic-g1.gguf"), filepath.Join(root, "generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"), filepath.Join(root, "generated/sonic/ggml/g1.mbphysics"), mb.DeviceVulkan)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			triangles := 0
			for _, shape := range p.CollisionShapes() {
				triangles += len(shape.Indices) / 3
			}
			t.Logf("collision overlay: %d shapes, %d hull triangles", len(p.CollisionShapes()), triangles)
			h := newStreamHub(s, p)
			defer h.Close()
			endpoint := httptest.NewServer(h.routes(s.routes()))
			defer endpoint.Close()
			opts := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
			opts = append(opts, chromedp.ExecPath("chromium"), chromedp.Flag("no-sandbox", true), chromedp.Flag("enable-unsafe-swiftshader", true))
			a, ac := chromedp.NewExecAllocator(context.Background(), opts...)
			defer ac()
			b, bc := chromedp.NewContext(a)
			defer bc()
			ctx, cancel := context.WithTimeout(b, 90*time.Second)
			defer cancel()
			var exceptions []string
			var exceptionMu sync.Mutex
			recordError := func(message string) {
				exceptionMu.Lock()
				defer exceptionMu.Unlock()
				exceptions = append(exceptions, message)
			}
			chromedp.ListenTarget(ctx, func(e any) {
				if x, ok := e.(*runtime.EventExceptionThrown); ok {
					recordError(x.ExceptionDetails.Text)
				}
				if x, ok := e.(*runtime.EventConsoleAPICalled); ok && x.Type == runtime.APITypeError {
					for _, arg := range x.Args {
						recordError(string(arg.Value))
					}
				}
			})
			defer func() {
				exceptionMu.Lock()
				defer exceptionMu.Unlock()
				if len(exceptions) > 0 {
					t.Error(exceptions)
				}
			}()
			run := func(actions ...chromedp.Action) {
				t.Helper()
				if err := chromedp.Run(ctx, actions...); err != nil {
					var message string
					_ = chromedp.Run(ctx, chromedp.Text("#live-physics-status", &message))
					t.Fatalf("%v: %s", err, message)
				}
			}
			run(chromedp.ActionFunc(func(ctx context.Context) error {
				_, err := page.AddScriptToEvaluateOnNewDocument(fmt.Sprintf(`const RealSocket=window.WebSocket;window.WebSocket=class extends RealSocket{set onmessage(fn){super.onmessage=e=>setTimeout(()=>fn(e),%d);}};`, delay)).Do(ctx)
				return err
			}))
			run(chromedp.EmulateViewport(1100, 800), chromedp.Navigate(endpoint.URL+"/?qa=1"), chromedp.Poll(`document.documentElement.dataset.streamHolding==='true'`, nil))
			checkIdle := func() {
				t.Helper()
				// Wait for the displayed pose to settle, not the ahead-of-playback
				// server flag (especially after loading the exact collision hulls).
				run(chromedp.Poll(`window.__motionBricksStreamQA.snapshot().rendered?.holding===true`, nil))
				var report map[string]any
				run(chromedp.Evaluate(`new Promise(resolve=>setTimeout(()=>{const a=window.__motionBricksStreamQA.snapshot();setTimeout(()=>{const b=window.__motionBricksStreamQA.snapshot();resolve({distance:Math.hypot(...a.rendered.root.map((x,i)=>x-b.rendered.root[i])),elapsed:b.rendered.time-a.rendered.time,error:document.documentElement.dataset.streamError??'',holding:document.documentElement.dataset.streamHolding});},2000)},300))`, &report, func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }))
				t.Logf("settled idle: %v", report)
				if report["distance"].(float64) > 1e-6 || report["elapsed"].(float64) < 1.8 || report["error"] != "" || report["holding"] != "true" {
					t.Fatal("idle drift or simulation stopped")
				}
			}
			checkIdle()
			run(chromedp.Click("#live-physics"), chromedp.Poll(`document.documentElement.dataset.livePhysics==='true'`, nil), chromedp.Evaluate(`document.querySelector('[data-key="w"]').click()`, nil), chromedp.Sleep(time.Second))
			run(chromedp.Click("#show-collisions"), chromedp.Poll(`document.documentElement.dataset.collisionsVisible==='true'`, nil))
			var collisionCheck bool
			run(chromedp.Evaluate(`(()=>{const shapes=window.__motionBricksStreamQA.snapshot().rendered.collisions;return shapes.length>2&&shapes.filter(s=>s.name.includes('ankle_roll')).length===2&&shapes.every(s=>s.opacity===.6&&s.position.every(Number.isFinite)&&Math.abs(s.rotation.reduce((v,x)=>v+x*x,0)-1)<1e-4)})()`, &collisionCheck))
			if !collisionCheck {
				t.Fatal("collision overlay definitions/transforms invalid")
			}
			var report map[string]any
			run(chromedp.Evaluate(`new Promise(resolve=>{
 const start=performance.now(),initial=window.__motionBricksStreamQA.snapshot();
 let last=initial.rendered,frames=0,maxRootStep=0,maxPhysicalStep=0,maxReferenceStep=0,invalid=0;
 const displacement=(a,b)=>{let maximum=0;for(let i=0;i<a.length;i+=3)maximum=Math.max(maximum,Math.hypot(a[i]-b[i],a[i+1]-b[i+1],a[i+2]-b[i+2]));return maximum;};
 const sample=now=>{
   frames++;const s=window.__motionBricksStreamQA.snapshot(),v=s.rendered;
   if(v&&last){
     for(const x of [...v.root,...v.reference,...(v.physical??[])])if(!Number.isFinite(x)||Math.abs(x)>10000)invalid++;
     maxRootStep=Math.max(maxRootStep,displacement(v.root,last.root));
     maxReferenceStep=Math.max(maxReferenceStep,displacement(v.reference,last.reference));
     if(v.physical&&last.physical)maxPhysicalStep=Math.max(maxPhysicalStep,displacement(v.physical,last.physical));
   }
   last=v;if(now-start<8000){requestAnimationFrame(sample);return;}
   resolve({wall:(now-start)/1000,simulation:v.time-initial.rendered.time,render_fps:frames/((now-start)/1000),max_root_step:maxRootStep,max_physical_step:maxPhysicalStep,max_reference_step:maxReferenceStep,distance:displacement(v.root,initial.rendered.root),invalid,underruns:s.underruns-initial.underruns,queue:s.queue,error:document.documentElement.dataset.streamError??'',final:s});
 };requestAnimationFrame(sample);
})`, &report, func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }))
			data, _ := json.MarshalIndent(report, "", "  ")
			out := filepath.Join(root, "generated/sonic/ggml/stream-qa")
			_ = os.MkdirAll(out, 0755)
			_ = os.WriteFile(filepath.Join(out, fmt.Sprintf("browser-%d.json", delay)), data, 0600)
			t.Logf("delay=%d rate=%.3fx render=%.1f FPS root-step=%.4fm physical-step=%.4fm underruns=%.0f", delay, report["simulation"].(float64)/report["wall"].(float64), report["render_fps"], report["max_root_step"], report["max_physical_step"], report["underruns"])
			rate := report["simulation"].(float64) / report["wall"].(float64)
			if report["max_reference_step"].(float64) > .5 || report["distance"].(float64) < 1 {
				t.Fatal("reference joints discontinuous or walking did not progress")
			}
			if rate < .95 || rate > 1.05 || report["invalid"].(float64) != 0 || report["underruns"].(float64) > 1 || report["max_root_step"].(float64) > .25 || report["max_physical_step"].(float64) > .5 || report["error"] != "" {
				t.Fatal("stream performance/continuity gate failed")
			}
			if delay == 0 {
				var standing []byte
				run(chromedp.FullScreenshot(&standing, 90))
				_ = os.WriteFile(filepath.Join(out, "collision-standing.png"), standing, 0600)
				await := func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }
				// Stop a moving character, settle once, and keep stepping physics
				// against the held reference instead of repeatedly planning creep.
				run(chromedp.Evaluate(`document.querySelector('[data-key="w"]').click()`, nil), chromedp.Poll(`document.documentElement.dataset.streamHolding==='true'`, nil))
				checkIdle()
				run(chromedp.Evaluate(`document.querySelector('[data-key="w"]').click()`, nil), chromedp.Poll(`document.documentElement.dataset.streamHolding==='false'`, nil))
				// Invalid commands must be rejected before changing the trajectory.
				run(chromedp.Evaluate(`window.__motionBricksStreamQA.send('control',{style:'walk',move:[1e20,0],facing:[0,1]})`, nil), chromedp.Poll(`JSON.parse(document.documentElement.dataset.streamAck||'{}').state==='rejected'`, nil))
				run(chromedp.Click("#stream-pause"), chromedp.Poll(`document.querySelector('#stream-pause').textContent==='Resume'`, nil))
				var pausedDelta float64
				run(chromedp.Evaluate(`new Promise(resolve=>setTimeout(()=>{const t=Number(document.documentElement.dataset.livePhysicsTime);setTimeout(()=>resolve(Number(document.documentElement.dataset.livePhysicsTime)-t),300)},400))`, &pausedDelta, await))
				if pausedDelta != 0 {
					t.Fatalf("paused simulation advanced %g", pausedDelta)
				}
				run(chromedp.Click("#stream-pause"), chromedp.Poll(`document.querySelector('#stream-pause').textContent==='Pause'`, nil))
				var epoch float64
				run(chromedp.Evaluate(`window.__motionBricksStreamQA.snapshot().rendered.epoch`, &epoch))
				run(chromedp.Evaluate(`window.__motionBricksStreamQA.disconnect()`, nil), chromedp.Poll(`document.documentElement.dataset.streamConnected==='false'`, nil), chromedp.Poll(`document.documentElement.dataset.streamConnected==='true'`, nil))
				var resumedEpoch float64
				run(chromedp.Evaluate(`window.__motionBricksStreamQA.snapshot().rendered.epoch`, &resumedEpoch))
				if epoch != resumedEpoch {
					t.Fatal("reconnect reset the simulation")
				}
				// Full authored playback and return to the previous walking action.
				// Kinematic first: successful physical tracking is a separate gate.
				run(chromedp.Click("#live-physics"), chromedp.Poll(`document.documentElement.dataset.livePhysics==='false'`, nil))
				if len(s.clips) > 0 {
					run(chromedp.Evaluate(`(()=>{const select=document.querySelector('#kimodo-select');const wave=[...select.options].find(o=>o.text.toLowerCase().includes('wave'));if(wave)select.value=wave.value;window.__streamTransition={maxStep:0,invalid:0,frames:0,last:null,active:true};const observe=()=>{const t=window.__streamTransition;if(!t.active)return;const v=window.__motionBricksStreamQA.snapshot().rendered;if(v){if(t.last)t.maxStep=Math.max(t.maxStep,Math.hypot(...v.root.map((x,i)=>x-t.last[i])));for(const x of v.root)if(!Number.isFinite(x)||Math.abs(x)>10000)t.invalid++;t.last=v.root;t.frames++;}requestAnimationFrame(observe);};observe();document.querySelector('#kimodo-play').click();})()`, nil), chromedp.Poll(`document.documentElement.dataset.kimodoState==='playing'`, nil), chromedp.Poll(`document.documentElement.dataset.kimodoState==='inactive'`, nil))
					var transition map[string]any
					run(chromedp.Evaluate(`new Promise(resolve=>{const a=window.__motionBricksStreamQA.snapshot().rendered.root;setTimeout(()=>{const b=window.__motionBricksStreamQA.snapshot().rendered.root;window.__streamTransition.active=false;resolve({...window.__streamTransition,resumed_distance:Math.hypot(...b.map((x,i)=>x-a[i])),error:document.documentElement.dataset.streamError??''})},1000)})`, &transition, await))
					payload, _ := json.MarshalIndent(transition, "", "  ")
					_ = os.WriteFile(filepath.Join(out, "kimodo-transition.json"), payload, 0600)
					t.Logf("Kimodo transition: %s", payload)
					if transition["maxStep"].(float64) > .3 || transition["invalid"].(float64) != 0 || transition["resumed_distance"].(float64) < .1 || transition["error"] != "" {
						t.Fatal("Kimodo continuity/resume gate failed")
					}
				}
				var screenshot []byte
				run(chromedp.FullScreenshot(&screenshot, 90))
				_ = os.WriteFile(filepath.Join(out, "streaming.png"), screenshot, 0600)
				// Reset is the only operation which intentionally starts a new epoch.
				run(chromedp.Click("#live-physics-reset"), chromedp.Poll(fmt.Sprintf(`window.__motionBricksStreamQA.snapshot().rendered?.epoch>%g`, epoch), nil))
				if len(s.clips) > 0 {
					run(chromedp.Click("#live-physics"), chromedp.Poll(`document.documentElement.dataset.livePhysics==='true'`, nil), chromedp.Click("#kimodo-play"))
					var physicalClip map[string]any
					run(chromedp.Evaluate(`new Promise(resolve=>{
 const start=performance.now();let previous=null,maxStep=0,invalid=0,epoch=null,changed=false,fallTime=null;
 const observe=now=>{
   const v=window.__motionBricksStreamQA.snapshot().rendered;
   const physicsTime=Number(document.documentElement.dataset.streamPhysicsTime);
   if(document.documentElement.dataset.streamFallen==='true'&&fallTime===null)fallTime=physicsTime;
   if(v){
     if(epoch===null)epoch=v.epoch;else if(epoch!==v.epoch)changed=true;
     for(const x of [...v.root,...(v.physical??[])])if(!Number.isFinite(x)||Math.abs(x)>10000)invalid++;
     if(v.physical&&previous)for(let i=0;i<90;i+=3)maxStep=Math.max(maxStep,Math.hypot(v.physical[i]-previous[i],v.physical[i+1]-previous[i+1],v.physical[i+2]-previous[i+2]));
     previous=v.physical;
   }
   const error=document.documentElement.dataset.streamError??'';
   if(now-start<12000&&!error){requestAnimationFrame(observe);return;}
   resolve({maxStep,invalid,changed,error,fallTime,physics_after_fall:fallTime===null?0:physicsTime-fallTime,paused:document.querySelector('#stream-pause').textContent==='Resume',warning:document.querySelector('#live-physics-status').textContent});
 };requestAnimationFrame(observe);
})`, &physicalClip, await))
					payload, _ := json.MarshalIndent(physicalClip, "", "  ")
					_ = os.WriteFile(filepath.Join(out, "kimodo-physical.json"), payload, 0600)
					t.Logf("physical Kimodo: %s", payload)
					if physicalClip["invalid"].(float64) != 0 || physicalClip["changed"].(bool) || physicalClip["maxStep"].(float64) > .5 || physicalClip["error"] != "" || physicalClip["paused"].(bool) || physicalClip["physics_after_fall"].(float64) < 2 {
						t.Fatal("physical Kimodo safety gate failed")
					}
					run(chromedp.FullScreenshot(&screenshot, 90))
					_ = os.WriteFile(filepath.Join(out, "collision-geometry.png"), screenshot, 0600)
					run(chromedp.Click("#show-collisions"), chromedp.Poll(`document.documentElement.dataset.collisionsVisible==='false'`, nil))
				}
			}
		})
	}
}
