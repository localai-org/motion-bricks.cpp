package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	cdpinput "github.com/chromedp/cdproto/input"
	cdplog "github.com/chromedp/cdproto/log"
	"github.com/chromedp/cdproto/runtime"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

func TestParseDevice(t *testing.T) {
	for name, expected := range map[string]mb.Device{"auto": mb.DeviceAuto, "cpu": mb.DeviceCPU, "vulkan": mb.DeviceVulkan, "CPU": mb.DeviceCPU} {
		actual, err := parseDevice(name)
		if err != nil || actual != expected {
			t.Fatalf("parseDevice(%q)=(%d,%v), want %d", name, actual, err, expected)
		}
	}
	if _, err := parseDevice("cuda"); err == nil {
		t.Fatal("unknown device was accepted")
	}
}

func TestNativeHTTPFlow(t *testing.T) {
	libraryPath, modelPath, stylesPath := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES")
	if libraryPath == "" || modelPath == "" || stylesPath == "" {
		t.Skip("native demo paths are not configured")
	}
	demo, err := loadDemoServer(libraryPath, modelPath, stylesPath, mb.DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	if directory := os.Getenv("MOTIONBRICKS_KIMODO_DIR"); directory != "" {
		if err = demo.loadKimodo(directory); err != nil {
			t.Fatal(err)
		}
	}
	server := httptest.NewServer(demo.routes())
	defer server.Close()
	response, err := http.Get(server.URL + "/api/meta")
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		FPS         int              `json:"fps"`
		Joints      []mb.Joint       `json:"joints"`
		Styles      []styleInfo      `json:"styles"`
		KimodoClips []kimodoClipInfo `json:"kimodo_clips"`
	}
	if err = json.NewDecoder(response.Body).Decode(&metadata); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK || metadata.FPS != 30 || len(metadata.Joints) != 34 || len(metadata.Styles) < 10 {
		t.Fatalf("invalid metadata: status=%d fps=%d joints=%d styles=%d", response.StatusCode, metadata.FPS, len(metadata.Joints), len(metadata.Styles))
	}
	post := func(path string, value any, output any) {
		t.Helper()
		body, _ := json.Marshal(value)
		reply, postErr := http.Post(server.URL+path, "application/json", bytes.NewReader(body))
		if postErr != nil {
			t.Fatal(postErr)
		}
		defer reply.Body.Close()
		if reply.StatusCode != http.StatusOK {
			var failure any
			_ = json.NewDecoder(reply.Body).Decode(&failure)
			t.Fatalf("%s: status=%d body=%v", path, reply.StatusCode, failure)
		}
		if err := json.NewDecoder(reply.Body).Decode(output); err != nil {
			t.Fatal(err)
		}
	}
	var initial planResponse
	post("/api/session", sessionRequest{Style: "walk"}, &initial)
	if initial.Session == "" || initial.Motion == nil || initial.Motion.Joints != 34 || initial.Motion.Frames < 24 ||
		initial.Targets == nil || initial.Targets.Frames != 4 || initial.Targets.Joints != 34 ||
		len(initial.Targets.Roots) != 12 || len(initial.Targets.Rotations) != 4*34*4 {
		t.Fatalf("invalid initial response: %+v", initial)
	}
	turnedStyle := ""
	for _, style := range metadata.Styles {
		if style.Name == "walk_zombie" {
			turnedStyle = style.Name
			break
		}
		if style.Name != "walk" && turnedStyle == "" {
			turnedStyle = style.Name
		}
	}
	if turnedStyle == "" {
		t.Fatal("no alternate upstream style is available")
	}
	var turned planResponse
	post("/api/plan", planRequest{Session: initial.Session, Style: turnedStyle, Move: [2]float32{1, 0}, Facing: [2]float32{1, 0}, Advance: 3, Seed: 77}, &turned)
	if turned.Style != turnedStyle || turned.Motion == nil || turned.Motion.Joints != 34 || len(turned.Motion.Rotations) != int(turned.Motion.Frames*34*4) ||
		turned.Targets == nil || turned.Targets.Frames != 4 || len(turned.Targets.Rotations) != 4*34*4 {
		t.Fatalf("invalid turned response: style=%q motion=%+v", turned.Style, turned.Motion)
	}
	if len(metadata.KimodoClips) > 0 {
		frame := 3
		var authored kimodoStartResponse
		post("/api/kimodo/start", kimodoStartRequest{Session: initial.Session, Clip: metadata.KimodoClips[0].ID, Advance: uint32(frame),
			CurrentRoot:      [3]float32{turned.Motion.Roots[frame*3], turned.Motion.Roots[frame*3+1], turned.Motion.Roots[frame*3+2]},
			CurrentRotations: append([]float32(nil), turned.Motion.Rotations[frame*34*4:(frame+1)*34*4]...)}, &authored)
		if authored.EntryFrames != kimodoEntryFrames || authored.Clip.ID != metadata.KimodoClips[0].ID || authored.Motion == nil ||
			authored.Motion.Frames != authored.Clip.Frames+kimodoEntryFrames || authored.Motion.Joints != 34 {
			t.Fatalf("invalid Kimodo start response: %+v", authored)
		}
		body, _ := json.Marshal(planRequest{Session: initial.Session, Style: "walk", Facing: [2]float32{0, 1}})
		blocked, postErr := http.Post(server.URL+"/api/plan", "application/json", bytes.NewReader(body))
		if postErr != nil {
			t.Fatal(postErr)
		}
		blocked.Body.Close()
		if blocked.StatusCode != http.StatusConflict {
			t.Fatalf("planning during Kimodo playback returned %d, want 409", blocked.StatusCode)
		}
		var resumed planResponse
		post("/api/kimodo/finish", kimodoFinishRequest{Session: initial.Session, Style: "walk", Facing: [2]float32{0, 1}, Seed: 91}, &resumed)
		if resumed.Motion == nil || resumed.Motion.Frames < 24 || resumed.Motion.Joints != 34 || resumed.Targets == nil || resumed.Targets.Frames != 4 {
			t.Fatalf("invalid MotionBricks resume response: %+v", resumed)
		}
	}
}

func TestReplayHTTPFlow(t *testing.T) {
	path := os.Getenv("MOTIONBRICKS_REPLAY")
	if path == "" {
		t.Skip("portable replay is not configured")
	}
	replay, err := loadReplayServer(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(replay.routes())
	defer server.Close()
	response, err := http.Get(server.URL + "/api/meta")
	if err != nil {
		t.Fatal(err)
	}
	var meta replayMetadata
	if err = json.NewDecoder(response.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK || meta.Runtime != "replay" || meta.Frames != 345 || meta.Joints != 30 || meta.Plans != 14 || meta.TargetFrames != 4 {
		t.Fatalf("invalid replay metadata: status=%d meta=%+v", response.StatusCode, meta)
	}
	response, err = http.Get(server.URL + "/api/replay")
	if err != nil {
		t.Fatal(err)
	}
	served, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK || !bytes.Equal(served, original) {
		t.Fatalf("served replay differs: status=%d served=%d original=%d", response.StatusCode, len(served), len(original))
	}
}

func TestComparisonHTTPFlow(t *testing.T) {
	path := os.Getenv("MOTIONBRICKS_COMPARISON")
	if path == "" {
		t.Skip("open-loop comparison is not configured")
	}
	comparison, err := loadComparisonServer(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(comparison.routes())
	defer server.Close()
	response, err := http.Get(server.URL + "/api/meta")
	if err != nil {
		t.Fatal(err)
	}
	var meta map[string]any
	if err = json.NewDecoder(response.Body).Decode(&meta); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK || meta["runtime"] != "comparison" || meta["plans"] != float64(14) ||
		meta["joints"] != float64(34) || meta["passed"] != true {
		t.Fatalf("invalid comparison metadata: status=%d meta=%v", response.StatusCode, meta)
	}
	response, err = http.Get(server.URL + "/api/comparison")
	if err != nil {
		t.Fatal(err)
	}
	served, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil || response.StatusCode != http.StatusOK || !bytes.Equal(served, original) {
		t.Fatalf("served comparison differs: status=%d err=%v", response.StatusCode, err)
	}
}

func TestHeadlessChrome(t *testing.T) {
	libraryPath, modelPath, stylesPath := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES")
	if libraryPath == "" || modelPath == "" || stylesPath == "" {
		t.Skip("native demo paths are not configured")
	}
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("chromium is not installed")
		}
	}
	demo, err := loadDemoServer(libraryPath, modelPath, stylesPath, mb.DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	if directory := os.Getenv("MOTIONBRICKS_KIMODO_DIR"); directory != "" {
		if err = demo.loadKimodo(directory); err != nil {
			t.Fatal(err)
		}
	}
	server := httptest.NewServer(demo.routes())
	defer server.Close()

	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options,
		chromedp.ExecPath(chrome),
		chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true),
		chromedp.Flag("use-angle", "swiftshader"),
		chromedp.Flag("enable-unsafe-swiftshader", true),
		chromedp.WindowSize(1280, 800),
	)
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	chromedp.ListenTarget(browser, func(event any) {
		switch event := event.(type) {
		case *cdplog.EventEntryAdded:
			t.Logf("browser log: %s", event.Entry.Text)
		case *runtime.EventConsoleAPICalled:
			t.Logf("browser console: %v", event.Args)
		case *runtime.EventExceptionThrown:
			t.Logf("browser exception: %s", event.ExceptionDetails.Text)
		}
	})
	ctx, cancel := context.WithTimeout(browser, 55*time.Second)
	defer cancel()

	waitFor := func(expression, description string) chromedp.Action {
		return chromedp.ActionFunc(func(ctx context.Context) error {
			deadline := time.Now().Add(25 * time.Second)
			for time.Now().Before(deadline) {
				var ready bool
				if evaluateErr := chromedp.Evaluate(expression, &ready).Do(ctx); evaluateErr != nil {
					return evaluateErr
				}
				if ready {
					return nil
				}
				time.Sleep(100 * time.Millisecond)
			}
			var diagnostic any
			_ = chromedp.Evaluate(`({status: document.documentElement.dataset.testStatus, sequence: document.documentElement.dataset.planSequence, moveX: document.documentElement.dataset.plannedMoveX, moveZ: document.documentElement.dataset.plannedMoveZ, scripts: [...document.scripts].map(s => ({src:s.src,type:s.type})), resources: performance.getEntriesByType("resource").map(r => r.name)})`, &diagnostic).Do(ctx)
			return fmt.Errorf("%w for %s: %#v", errors.New("timeout waiting for browser"), description, diagnostic)
		})
	}
	var initialScreenshot, kimodoScreenshot, jumpScreenshot, movingScreenshot, screenshot []byte
	var viewportCenter struct {
		X float64 `json:"x"`
		Y float64 `json:"y"`
	}
	var message string
	uploadPath := filepath.Join(t.TempDir(), "uploaded-walk.glb")
	if err = os.WriteFile(uploadPath, testAnimationGLB(t, demo.joints, "g1skel34"), 0600); err != nil {
		t.Fatal(err)
	}
	err = chromedp.Run(ctx,
		chromedp.ActionFunc(func(ctx context.Context) error {
			if enableErr := cdplog.Enable().Do(ctx); enableErr != nil {
				return enableErr
			}
			return runtime.Enable().Do(ctx)
		}),
		chromedp.Navigate(server.URL+"/"),
		waitFor(`document.documentElement.dataset.testStatus === "ready"`, "initial plan"),
		chromedp.SetUploadFiles(`#kimodo-file`, []string{uploadPath}, chromedp.ByQuery),
		waitFor(`document.querySelector('#kimodo-upload-status').textContent.startsWith('Imported uploaded-walk.glb') && document.querySelector('#kimodo-select').value.startsWith('upload-')`, "GLB upload and automatic selection"),
		chromedp.Click(`.pad button[data-key="w"]`, chromedp.ByQuery),
		waitFor(`Math.hypot(Number(document.documentElement.dataset.plannedMoveX), Number(document.documentElement.dataset.plannedMoveZ)) > 0.9`, "walking before imported animation"),
		chromedp.Evaluate(`(() => { const d = document.documentElement.dataset; d.resumeTestX = d.plannedMoveX; d.resumeTestZ = d.plannedMoveZ; })()`, nil),
		chromedp.FullScreenshot(&initialScreenshot, 90),
		chromedp.Click(`#kimodo-play`, chromedp.ByQuery),
		waitFor(`document.documentElement.dataset.kimodoState === "playing" && document.documentElement.dataset.controlsLocked === "true" && document.querySelector('#jump').disabled && [...document.querySelectorAll('.pad button')].every(button => button.disabled)`, "non-interruptible Kimodo playback"),
		chromedp.Sleep(500*time.Millisecond),
		chromedp.FullScreenshot(&kimodoScreenshot, 90),
		waitFor(`document.documentElement.dataset.kimodoState === "complete" && document.documentElement.dataset.controlsLocked === "false" && Number(document.documentElement.dataset.planSequence) >= 2`, "Kimodo exit context and MotionBricks resume"),
		waitFor(`document.documentElement.dataset.plannedMoveX === document.documentElement.dataset.resumeTestX && document.documentElement.dataset.plannedMoveZ === document.documentElement.dataset.resumeTestZ && document.querySelector('.pad button[data-key="w"]').getAttribute('aria-pressed') === 'true'`, "walking action restored after imported animation"),
		chromedp.Click(`#jump`, chromedp.ByQuery),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 2 && document.documentElement.dataset.plannedJump === "true" && Number(document.documentElement.dataset.plannedSpeed) === 5 && Number(document.documentElement.dataset.jumpTargetDistance) > 4 && Number(document.documentElement.dataset.jumpTargetVelocity) > 3`, "jump speed override and distant high-velocity standard keyframes"),
		chromedp.Sleep(650*time.Millisecond),
		waitFor(`document.documentElement.dataset.plannedJump === "true" && document.querySelector('#jump').disabled`, "jump survives the 16-frame walking replan"),
		chromedp.Click(`#show-all-targets`, chromedp.ByQuery),
		chromedp.Sleep(100*time.Millisecond),
		chromedp.FullScreenshot(&jumpScreenshot, 90),
		waitFor(`document.documentElement.dataset.plannedJump === "false" && !document.querySelector('#jump').disabled`, "normal planning resumes after the complete jump"),
		chromedp.Navigate(server.URL+"/"),
		waitFor(`document.documentElement.dataset.testStatus === "ready"`, "fresh movement session"),
		chromedp.Click(`.pad button[data-key="w"]`, chromedp.ByQuery),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 2 && Math.abs(Number(document.documentElement.dataset.plannedMoveX) + Math.sin(Number(document.documentElement.dataset.cameraYaw))) < 1e-5 && Math.abs(Number(document.documentElement.dataset.plannedMoveZ) + Math.cos(Number(document.documentElement.dataset.cameraYaw))) < 1e-5`, "camera-forward pad-button plan"),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 3 && document.documentElement.dataset.plannedAdvance === "16" && document.documentElement.dataset.controllerCadenceFrames === "16" && Number(document.documentElement.dataset.plannedMoveX) !== 0`, "sustained movement across the 16-frame controller cadence"),
		chromedp.FullScreenshot(&movingScreenshot, 90),
		chromedp.Click(`.pad button[data-key="w"]`, chromedp.ByQuery),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 4 && document.documentElement.dataset.plannedMoveX === "0" && document.documentElement.dataset.plannedMoveZ === "0"`, "pad-button stop plan"),
		chromedp.Evaluate(`(() => { const bounds = document.querySelector('#viewport').getBoundingClientRect(); return {x: bounds.left + bounds.width / 2, y: bounds.top + bounds.height / 2}; })()`, &viewportCenter),
		chromedp.ActionFunc(func(ctx context.Context) error {
			if err := cdpinput.DispatchMouseEvent(cdpinput.MousePressed, viewportCenter.X, viewportCenter.Y).WithButton(cdpinput.Left).WithButtons(1).WithClickCount(1).Do(ctx); err != nil {
				return err
			}
			if err := cdpinput.DispatchMouseEvent(cdpinput.MouseMoved, viewportCenter.X+120, viewportCenter.Y).WithButton(cdpinput.Left).WithButtons(1).Do(ctx); err != nil {
				return err
			}
			return cdpinput.DispatchMouseEvent(cdpinput.MouseReleased, viewportCenter.X+120, viewportCenter.Y).WithButton(cdpinput.Left).Do(ctx)
		}),
		waitFor(`Math.abs(Number(document.documentElement.dataset.cameraYaw) - 0.68) > 0.5`, "camera orbit"),
		chromedp.Evaluate(`dispatchEvent(new KeyboardEvent("keydown", {key:"d", bubbles:true}))`, nil),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 5 && Math.abs(Number(document.documentElement.dataset.plannedMoveX) - Math.cos(Number(document.documentElement.dataset.cameraYaw))) < 1e-5 && Math.abs(Number(document.documentElement.dataset.plannedMoveZ) + Math.sin(Number(document.documentElement.dataset.cameraYaw))) < 1e-5`, "camera-right keyboard plan after orbit"),
		chromedp.Evaluate(`dispatchEvent(new KeyboardEvent("keyup", {key:"d", bubbles:true}))`, nil),
		waitFor(`Number(document.documentElement.dataset.planSequence) >= 6 && document.documentElement.dataset.plannedMoveX === "0" && document.documentElement.dataset.plannedMoveZ === "0"`, "keyboard stop plan"),
		chromedp.Navigate(server.URL+"/?test=1"),
		waitFor(`document.documentElement.dataset.testStatus === "passed" || document.documentElement.dataset.testStatus === "failed"`, "style-and-turn self-test"),
		chromedp.Text("#test-result", &message, chromedp.ByQuery),
		chromedp.FullScreenshot(&screenshot, 90),
	)
	if err != nil {
		t.Fatal(err)
	}
	var status string
	if err = chromedp.Run(ctx, chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status)); err != nil {
		t.Fatal(err)
	}
	if status != "passed" {
		t.Fatalf("browser self-test status=%q: %s", status, message)
	}
	if len(initialScreenshot) < 10_000 || len(kimodoScreenshot) < 10_000 || len(jumpScreenshot) < 10_000 || len(movingScreenshot) < 10_000 || len(screenshot) < 10_000 {
		t.Fatalf("rendered screenshots are unexpectedly small: initial=%d kimodo=%d jump=%d moving=%d final=%d", len(initialScreenshot), len(kimodoScreenshot), len(jumpScreenshot), len(movingScreenshot), len(screenshot))
	}
	artifact := os.Getenv("MOTIONBRICKS_SCREENSHOT")
	if artifact == "" {
		artifact = filepath.Join(t.TempDir(), "motionbricks-demo.png")
	}
	extension := filepath.Ext(artifact)
	if extension == "" {
		extension = ".png"
	}
	base := strings.TrimSuffix(artifact, filepath.Ext(artifact))
	artifacts := []struct {
		path string
		data []byte
	}{
		{path: base + "-initial" + extension, data: initialScreenshot},
		{path: base + "-kimodo" + extension, data: kimodoScreenshot},
		{path: base + "-jump" + extension, data: jumpScreenshot},
		{path: base + "-moving" + extension, data: movingScreenshot},
		{path: artifact, data: screenshot},
	}
	for _, item := range artifacts {
		if len(item.data) == 0 {
			continue
		}
		if err = os.WriteFile(item.path, item.data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%s; screenshots: %s, %s, %s, %s, %s (final %s)", message, artifacts[0].path, artifacts[1].path,
		artifacts[2].path, artifacts[3].path, artifacts[4].path, fmt.Sprintf("%d bytes", len(screenshot)))
}

func TestHeadlessReplay(t *testing.T) {
	path := os.Getenv("MOTIONBRICKS_REPLAY")
	if path == "" {
		t.Skip("portable replay is not configured")
	}
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("chromium is not installed")
		}
	}
	replay, err := loadReplayServer(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(replay.routes())
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true), chromedp.Flag("use-angle", "swiftshader"),
		chromedp.Flag("enable-unsafe-swiftshader", true), chromedp.WindowSize(1280, 800))
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	ctx, cancel := context.WithTimeout(browser, 30*time.Second)
	defer cancel()
	wait := chromedp.ActionFunc(func(ctx context.Context) error {
		deadline := time.Now().Add(25 * time.Second)
		for time.Now().Before(deadline) {
			var status string
			if evaluateErr := chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status).Do(ctx); evaluateErr != nil {
				return evaluateErr
			}
			if status == "passed" {
				return nil
			}
			if status == "failed" {
				var message string
				_ = chromedp.Text("#test-result", &message, chromedp.ByQuery).Do(ctx)
				return fmt.Errorf("browser replay self-test failed: %s", message)
			}
			time.Sleep(100 * time.Millisecond)
		}
		return errors.New("timeout waiting for browser replay self-test")
	})
	var screenshot []byte
	if err = chromedp.Run(ctx, chromedp.Navigate(server.URL+"/?test=1"), wait, chromedp.FullScreenshot(&screenshot, 90)); err != nil {
		t.Fatal(err)
	}
	var values map[string]string
	if err = chromedp.Run(ctx, chromedp.Evaluate(`({runtime:document.documentElement.dataset.runtime,frames:document.documentElement.dataset.replayFrames,joints:document.documentElement.dataset.animatedJoints,plans:document.documentElement.dataset.replayPlans,targets:document.documentElement.dataset.targetFrames,visible:document.documentElement.dataset.visibleTargets,camera:document.documentElement.dataset.cameraSubject})`, &values)); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"runtime": "replay", "frames": "345", "joints": "30", "plans": "14", "targets": "4", "visible": "4", "camera": "animated"}
	for key, expected := range want {
		if values[key] != expected {
			t.Fatalf("browser replay %s=%q, want %q (all=%v)", key, values[key], expected, values)
		}
	}
	if len(screenshot) < 10_000 {
		t.Fatalf("replay screenshot is unexpectedly small: %d bytes", len(screenshot))
	}
	artifact := os.Getenv("MOTIONBRICKS_REPLAY_SCREENSHOT")
	if artifact == "" {
		artifact = filepath.Join(t.TempDir(), "motionbricks-replay.png")
	}
	if err = os.WriteFile(artifact, screenshot, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("replay screenshot: %s (%d bytes)", artifact, len(screenshot))
}

func TestHeadlessComparison(t *testing.T) {
	path := os.Getenv("MOTIONBRICKS_COMPARISON")
	if path == "" {
		t.Skip("open-loop comparison is not configured")
	}
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("chromium is not installed")
		}
	}
	comparison, err := loadComparisonServer(path)
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(comparison.routes())
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true), chromedp.Flag("use-angle", "swiftshader"),
		chromedp.Flag("enable-unsafe-swiftshader", true), chromedp.WindowSize(1280, 800))
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	ctx, cancel := context.WithTimeout(browser, 30*time.Second)
	defer cancel()
	wait := chromedp.ActionFunc(func(ctx context.Context) error {
		deadline := time.Now().Add(25 * time.Second)
		for time.Now().Before(deadline) {
			var status string
			if evaluateErr := chromedp.Evaluate(`document.documentElement.dataset.testStatus`, &status).Do(ctx); evaluateErr != nil {
				return evaluateErr
			}
			if status == "passed" {
				return nil
			}
			if status == "failed" {
				return errors.New("browser comparison self-test failed")
			}
			time.Sleep(100 * time.Millisecond)
		}
		return errors.New("timeout waiting for browser comparison self-test")
	})
	var screenshot []byte
	if err = chromedp.Run(ctx, chromedp.Navigate(server.URL+"/?test=1"), wait, chromedp.FullScreenshot(&screenshot, 90)); err != nil {
		t.Fatal(err)
	}
	var values map[string]string
	if err = chromedp.Run(ctx, chromedp.Evaluate(`({runtime:document.documentElement.dataset.runtime,passed:document.documentElement.dataset.comparisonPassed,plans:document.documentElement.dataset.comparisonPlans,joints:document.documentElement.dataset.animatedJoints,vectors:document.documentElement.dataset.errorVectors,showcase:document.documentElement.dataset.comparisonShowcase,showcaseFrames:document.documentElement.dataset.comparisonShowcaseFrames})`, &values)); err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"runtime": "comparison", "passed": "true", "plans": "14", "joints": "34", "vectors": "34",
		"showcase": "Forward walk,Right turn,Zombie walk", "showcaseFrames": "176"}
	for key, expected := range want {
		if values[key] != expected {
			t.Fatalf("comparison %s=%q, want %q", key, values[key], expected)
		}
	}
	if len(screenshot) < 10_000 {
		t.Fatalf("comparison screenshot is unexpectedly small: %d", len(screenshot))
	}
	artifact := os.Getenv("MOTIONBRICKS_COMPARISON_SCREENSHOT")
	if artifact == "" {
		artifact = filepath.Join(t.TempDir(), "motionbricks-comparison.png")
	}
	if err = os.WriteFile(artifact, screenshot, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("comparison screenshot: %s (%d bytes)", artifact, len(screenshot))
}
