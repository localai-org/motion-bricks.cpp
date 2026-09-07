package main

import (
	"context"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLivePhysicsDisabled(t *testing.T) {
	h := livePhysicsRoutes(http.NotFoundHandler(), nil)
	for _, tc := range []struct {
		method, path string
		status       int
	}{{"GET", "/api/live-physics", 200}, {"POST", "/api/live-physics", 503}} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest(tc.method, tc.path, nil))
		if w.Code != tc.status {
			t.Fatal(w.Code)
		}
	}
}

func TestLivePhysicsBrowser(t *testing.T) {
	root := os.Getenv("MOTIONBRICKS_LIVE_TEST_ROOT")
	if root == "" {
		t.Skip("set MOTIONBRICKS_LIVE_TEST_ROOT for full native simulation QA")
	}
	library := filepath.Join(root, "build/debug/libmotionbricks.so")
	server, err := loadDemoServer(library, filepath.Join(root, "generated/g1-f32"), filepath.Join(root, "generated/styles"), mb.DeviceVulkan)
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	if err = server.loadKimodo(os.Getenv("MOTIONBRICKS_LIVE_TEST_KIMODO")); err != nil {
		t.Fatal(err)
	}
	p, err := mb.OpenPhysics(library, filepath.Join(root, "generated/sonic/ggml/sonic-g1.gguf"), filepath.Join(root, "generated/sonic/assets/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"), filepath.Join(root, "generated/sonic/ggml/g1.mbphysics"), mb.DeviceVulkan)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	endpoint := httptest.NewServer(livePhysicsRoutes(server.routes(), p))
	defer endpoint.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath("chromium"), chromedp.Flag("no-sandbox", true), chromedp.Flag("enable-unsafe-swiftshader", true))
	allocator, cancel := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancel()
	browser, closeBrowser := chromedp.NewContext(allocator)
	defer closeBrowser()
	ctx, stop := context.WithTimeout(browser, 45*time.Second)
	defer stop()
	var exceptions []string
	chromedp.ListenTarget(ctx, func(e any) {
		if v, ok := e.(*runtime.EventExceptionThrown); ok {
			exceptions = append(exceptions, v.ExceptionDetails.Text)
		}
	})
	run := func(actions ...chromedp.Action) {
		t.Helper()
		if err := chromedp.Run(ctx, actions...); err != nil {
			var message string
			_ = chromedp.Run(ctx, chromedp.Text("#live-physics-status", &message))
			t.Fatalf("%v: %s", err, message)
		}
	}
	run(chromedp.EmulateViewport(1440, 1000), chromedp.Navigate(endpoint.URL+"/?qa=1"), chromedp.Poll(`document.documentElement.dataset.testStatus==='ready'`, nil), chromedp.Click("#live-physics"), chromedp.Poll(`Number(document.documentElement.dataset.livePhysicsTime)>1`, nil))
	run(chromedp.Evaluate(`document.querySelector('[data-key="w"]').click()`, nil), chromedp.Poll(`Number(document.documentElement.dataset.livePhysicsTime)>3`, nil))
	if len(server.clips) > 0 {
		run(chromedp.Evaluate(`(()=>{const s=document.querySelector('#kimodo-select');const o=[...s.options].find(o=>o.text.toLowerCase().includes('wave'));if(o)s.value=o.value;document.querySelector('#kimodo-play').click()})()`, nil), chromedp.Poll(`document.documentElement.dataset.kimodoState==='playing'`, nil), chromedp.Poll(`Number(document.documentElement.dataset.livePhysicsTime)>4`, nil))
	}
	if output := os.Getenv("MOTIONBRICKS_QA_DIR"); output != "" {
		var png []byte
		run(chromedp.FullScreenshot(&png, 90))
		if err := os.MkdirAll(output, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(output, "live-sonic.png"), png, 0600); err != nil {
			t.Fatal(err)
		}
	}
	run(chromedp.Click("#live-physics"), chromedp.Poll(`document.documentElement.dataset.livePhysics==='false'`, nil))
	if len(exceptions) > 0 {
		t.Fatal(exceptions)
	}
}
