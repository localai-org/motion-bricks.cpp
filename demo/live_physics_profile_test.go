package main

// Diagnostic only: no changes to production scheduling. An isolated server
// avoids taking ownership of the user's running physical session.
import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

func TestLivePhysicsProfile(t *testing.T) {
	root := os.Getenv("MOTIONBRICKS_PROFILE_ROOT")
	if root == "" {
		t.Skip("set MOTIONBRICKS_PROFILE_ROOT for wall-clock profiling")
	}
	library := filepath.Join(root, "build/debug/libmotionbricks.so")
	s, err := loadDemoServer(library, filepath.Join(root, "generated/g1-f32"), filepath.Join(root, "generated/styles"), mb.DeviceVulkan)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	p, err := mb.OpenPhysics(library, filepath.Join(root, "generated/sonic/ggml/sonic-g1.gguf"), filepath.Join(root, "generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"), filepath.Join(root, "generated/sonic/ggml/g1.mbphysics"), mb.DeviceVulkan)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, latency := range []int{-1, 0, 50, 100} {
		t.Run(fmt.Sprintf("extra_rtt_%d_ms", latency), func(t *testing.T) {
			var mu sync.Mutex
			var requests []map[string]any
			base := livePhysicsRoutes(s.routes(), p)
			h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				start := time.Now()
				base.ServeHTTP(w, r)
				if r.Method == "POST" {
					mu.Lock()
					requests = append(requests, map[string]any{"path": r.URL.Path, "ms": float64(time.Since(start)) / 1e6, "request_bytes": r.ContentLength})
					mu.Unlock()
				}
			})
			endpoint := httptest.NewServer(h)
			defer func() { endpoint.Close(); p.Reset() }()
			opts := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
			opts = append(opts, chromedp.ExecPath("chromium"), chromedp.Flag("no-sandbox", true), chromedp.Flag("enable-unsafe-swiftshader", true))
			a, ac := chromedp.NewExecAllocator(context.Background(), opts...)
			defer ac()
			b, bc := chromedp.NewContext(a)
			defer bc()
			ctx, cancel := context.WithTimeout(b, 35*time.Second)
			defer cancel()
			run := func(actions ...chromedp.Action) {
				t.Helper()
				if err := chromedp.Run(ctx, actions...); err != nil {
					t.Fatal(err)
				}
			}
			run(chromedp.EmulateViewport(960, 700), chromedp.Navigate(endpoint.URL+"/?qa=1"), chromedp.Poll(`document.documentElement.dataset.testStatus==='ready'`, nil))
			run(chromedp.Evaluate(fmt.Sprintf(`(()=>{const original=window.fetch.bind(window);window.__profileRequests=[];window.fetch=async(...args)=>{const start=performance.now();const response=await original(...args);if(String(args[0]).startsWith('/api/') && args[1]?.method==='POST'){if(%d>0)await new Promise(r=>setTimeout(r,%d));window.__profileRequests.push({path:String(args[0]),start,end:performance.now(),ms:performance.now()-start,status:response.status,bytes:args[1].body?.length??0});}return response;};})()`, latency, latency), nil))
			if latency >= 0 {
				run(chromedp.Click("#live-physics"), chromedp.Poll(`Number(document.documentElement.dataset.livePhysicsTime)>1`, nil))
			}
			run(chromedp.Evaluate(`document.querySelector('[data-key="w"]').click()`, nil))
			var report map[string]any
			run(chromedp.Evaluate(`new Promise(resolve=>{const start=performance.now(),initial=Number(document.documentElement.dataset.livePhysicsTime||0);let previous=start,frames=0,lastTick=initial,lastChange=start;const intervals=[],tickGaps=[];const collect=now=>{frames++;intervals.push(now-previous);previous=now;const tick=Number(document.documentElement.dataset.livePhysicsTime||0);if(tick!==lastTick){tickGaps.push(now-lastChange);lastChange=now;lastTick=tick;}if(now-start<8000){requestAnimationFrame(collect);return;}const wall=(now-start)/1000;resolve({wall_seconds:wall,simulation_seconds:tick-initial,simulation_rate:(tick-initial)/wall,render_fps:frames/wall,raf_intervals_ms:intervals,simulation_update_gaps_ms:tickGaps,requests:window.__profileRequests.filter(r=>r.start>=start),status:document.querySelector('#live-physics-status').textContent});};requestAnimationFrame(collect);})`, &report, func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }))
			mu.Lock()
			report["server_requests"] = append([]map[string]any{}, requests...)
			mu.Unlock()
			report["extra_rtt_ms"] = latency
			out := filepath.Join(root, "generated/sonic/ggml/performance")
			if err := os.MkdirAll(out, 0755); err != nil {
				t.Fatal(err)
			}
			data, _ := json.MarshalIndent(report, "", "  ")
			if err := os.WriteFile(filepath.Join(out, fmt.Sprintf("browser-%d.json", latency)), data, 0600); err != nil {
				t.Fatal(err)
			}
			t.Logf("RTT +%dms: render %.1f FPS, simulation rate %.3fx, status %s", latency, report["render_fps"], report["simulation_rate"], report["status"])
			if latency >= 0 {
				run(chromedp.Click("#live-physics"), chromedp.Poll(`document.documentElement.dataset.livePhysics==='false'`, nil))
			}
		})
	}
}
