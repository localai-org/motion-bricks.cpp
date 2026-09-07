package main

import (
	"context"
	"encoding/json"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
)

func writePhysicsFixture(t *testing.T, directory string, invalid bool) {
	t.Helper()
	joints := make([]map[string]any, 30)
	positions := make([]float64, 2*30*3)
	for i := range joints {
		joints[i] = map[string]any{"name": "joint", "parent": i - 1, "position": []float64{0, .8 + float64(i)*.02, 0}}
		positions[i*3+1] = .8 + float64(i)*.02
		positions[(i+30)*3+1] = positions[i*3+1]
	}
	if invalid {
		joints[2]["parent"] = 2
	}
	doc := map[string]any{"format": "motionbricks-sonic-playback-v1", "recorded": true, "title": "Synthetic browser test",
		"frames": 2, "fps": 50, "times": []float64{0, .02}, "joints": joints,
		"actual_positions": positions, "reference_positions": positions, "root_error_m": []float64{0, 0},
		"body_error_m": []float64{0, 0}, "contacts": []int{1, 1},
		"diagnostics": map[string]any{"joint_rmse_rad": 0, "root_xy_final_error_m": 0}}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(directory, "native-walk.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestPhysicsRoutes(t *testing.T) {
	dir := t.TempDir()
	writePhysicsFixture(t, dir, false)
	handler, err := physicsRoutes(http.NotFoundHandler(), dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		path   string
		status int
	}{
		{"/api/physics", 200}, {"/api/physics/native-walk", 200}, {"/api/physics/unknown", 404}, {"/api/physics/%2Fetc%2Fpasswd", 404},
	} {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, httptest.NewRequest("GET", test.path, nil))
		if response.Code != test.status {
			t.Fatalf("%s: %d", test.path, response.Code)
		}
	}
	writePhysicsFixture(t, dir, true)
	if _, err := physicsRoutes(http.NotFoundHandler(), dir); err == nil {
		t.Fatal("accepted cyclic skeleton")
	}
}

func TestPhysicsBrowser(t *testing.T) {
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("Chromium unavailable")
		}
	}
	dir := os.Getenv("MOTIONBRICKS_PHYSICS_TEST_DIR")
	if dir == "" {
		dir = t.TempDir()
		writePhysicsFixture(t, dir, false)
	}
	static, err := fs.Sub(webFiles, "web")
	if err != nil {
		t.Fatal(err)
	}
	handler, err := physicsRoutes(http.FileServer(http.FS(static)), dir)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(handler)
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true), chromedp.Flag("enable-unsafe-swiftshader", true))
	allocator, cancel := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancel()
	browser, closeBrowser := chromedp.NewContext(allocator)
	defer closeBrowser()
	ctx, stop := context.WithTimeout(browser, 40*time.Second)
	defer stop()
	var exceptions []string
	chromedp.ListenTarget(ctx, func(event any) {
		if e, ok := event.(*runtime.EventExceptionThrown); ok {
			exceptions = append(exceptions, e.ExceptionDetails.Text)
		}
	})
	id := os.Getenv("MOTIONBRICKS_PHYSICS_TEST_ID")
	if id == "" {
		id = "native-walk"
	}
	if err := chromedp.Run(ctx, chromedp.EmulateViewport(1440, 1000), chromedp.Navigate(server.URL+"/?physics="+id),
		chromedp.Poll(`['ready','failed'].includes(document.documentElement.dataset.testStatus)`, nil)); err != nil {
		t.Fatal(err)
	}
	var status string
	if err := chromedp.Run(ctx, chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status)); err != nil {
		t.Fatal(err)
	}
	if status != "ready" {
		var message string
		_ = chromedp.Run(ctx, chromedp.Text("#test-result", &message))
		t.Fatal(message)
	}
	if id == "kimodo-failed" {
		var message string
		if err := chromedp.Run(ctx, chromedp.Text("#status", &message)); err != nil || !strings.Contains(message, "FAILED") {
			t.Fatalf("failed recording not labelled: %q %v", message, err)
		}
	}
	for _, sample := range []struct {
		name     string
		fraction float64
	}{{"start", 0}, {"middle", .5}, {"end", 1}} {
		fraction, _ := json.Marshal(sample.fraction)
		js := `(()=>{const s=document.querySelector('#replay-frame');s.value=Math.round(Number(s.max)*` + string(fraction) + `);s.dispatchEvent(new Event('input'));return Number(document.documentElement.dataset.physicsFrame)===Number(s.value)})()`
		var correct bool
		if err := chromedp.Run(ctx, chromedp.Evaluate(js, &correct)); err != nil || !correct {
			t.Fatalf("scrub %s: %v %v", sample.name, correct, err)
		}
		if output := os.Getenv("MOTIONBRICKS_QA_DIR"); output != "" {
			var screenshot []byte
			if err := chromedp.Run(ctx, chromedp.FullScreenshot(&screenshot, 90)); err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(output, 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(output, "sonic-"+sample.name+".png"), screenshot, 0600); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := chromedp.Run(ctx, chromedp.Click("#physics-overlay"), chromedp.Click("#physics-overlay"), chromedp.Click("#reset-camera"), chromedp.Click("#replay-toggle")); err != nil {
		t.Fatal(err)
	}
	if len(exceptions) > 0 {
		t.Fatal(exceptions)
	}
}
