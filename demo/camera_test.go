package main

import (
	"context"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"testing"
	"time"
)

func TestCameraFollow(t *testing.T) {
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("Chromium unavailable")
		}
	}
	source, err := webFiles.ReadFile("web/camera-follow.js")
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/camera-follow.js" {
			w.Header().Set("Content-Type", "text/javascript")
			_, _ = w.Write(source)
		} else {
			w.Header().Set("Content-Type", "text/html")
			_, _ = w.Write([]byte("<!doctype html>"))
		}
	}))
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true))
	allocator, cancel := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancel()
	browser, closeBrowser := chromedp.NewContext(allocator)
	defer closeBrowser()
	ctx, stop := context.WithTimeout(browser, 20*time.Second)
	defer stop()
	var failures []string
	err = chromedp.Run(ctx, chromedp.Navigate(server.URL), chromedp.Evaluate(`(async()=>{
  const {CameraFollow}=await import('/camera-follow.js');const failures=[];
  const assert=(ok,name)=>{if(!ok)failures.push(name)};
  const simulate=hz=>{const f=new CameraFollow();f.reset([0,0,0]);for(let i=0;i<hz;i++)f.update([1,0.3,0],1/hz);return f.position};
  const a=simulate(30),b=simulate(60),c=simulate(144);
  assert(a.every((v,i)=>Math.abs(v-b[i])<1e-9&&Math.abs(v-c[i])<1e-9),'frame-rate independence');
  const f=new CameraFollow();f.reset([0,0.8,0]);const first=[...f.update([10,8,0],1/60)];
  assert(first[0]<=0.1&&first[1]-0.8<=0.8/60+1e-9,'bounded teleport response');
  let previous=[...f.position];
  for(let i=0;i<600;i++){const p=f.update([10,8,0],1/60);assert(Math.abs(p[1]-previous[1])<=0.8/60+1e-9,'vertical speed cap');previous=[...p]}
  assert(Math.abs(f.position[0]-10)<0.01,'catches up horizontally');
  const bounce=new CameraFollow();bounce.reset([0,0.8,0]);let min=Infinity,max=-Infinity;
  for(let i=0;i<600;i++){const y=bounce.update([0,0.8+0.06*Math.sin(i/60*2*Math.PI*8),0],1/60)[1];if(i>120){min=Math.min(min,y);max=Math.max(max,y)}}
  assert(max-min<0.005,'suppresses walking-height jitter');
  const before=[...bounce.position];bounce.update([1,2,3],0);assert(before.every((v,i)=>v===bounce.position[i]),'paused filter');
  bounce.reset([3,4,5]);assert(bounce.position[0]===3&&bounce.velocity.every(v=>v===0),'explicit camera reset');
  return failures;
 })()`, &failures, func(p *runtime.EvaluateParams) *runtime.EvaluateParams { return p.WithAwaitPromise(true) }))
	if err != nil {
		t.Fatal(err)
	}
	if len(failures) > 0 {
		t.Fatal(failures)
	}
}
