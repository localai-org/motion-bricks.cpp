package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
	"os"
	"testing"
	"time"
)

func TestSOMATargetPlacement(t *testing.T) {
	style := &somaStyle{Speed: 2, VelocityScale: .9, TargetHorizon: 1.3, Local: make([]float32, 28)}
	context := make([]float32, 28)
	for f := 0; f < 4; f++ {
		style.Local[f*7+2] = float32(f) / 30
		style.Local[f*7+6] = 1
		context[f*7] = 10
		context[f*7+2] = 20
		context[f*7+6] = 1
	}
	placed := placeSOMATarget(style, context, 1, [2]float32{1, 0}, [2]float32{1, 0})
	if math.Abs(float64(placed[0]-12.34)) > 1e-5 || placed[2] != 20 || math.Abs(float64(placed[21]-12.52)) > 1e-5 {
		t.Fatalf("incorrect placement %v", placed)
	}
	if math.Abs(float64(placed[4]-float32(math.Sqrt(.5)))) > 1e-6 {
		t.Fatal("facing did not rotate root")
	}
	if style.Local[0] != 0 {
		t.Fatal("mutated style")
	}
	stopped := placeSOMATarget(style, context, 1, [2]float32{}, [2]float32{1, 0})
	for f := 0; f < 4; f++ {
		if stopped[f*7] != 10 || stopped[f*7+2] != 20 {
			t.Fatal("stopped target drifts")
		}
	}
	partial := placeSOMATarget(style, context, 1, [2]float32{.5, 0}, [2]float32{1, 0})
	if math.Abs(float64(partial[0]-11.17)) > 1e-5 || math.Abs(float64(partial[21]-11.26)) > 1e-5 {
		t.Fatal("partial movement did not scale target distance and velocity")
	}
}

func TestSOMAStreamPlanner(t *testing.T) {
	library, bundle, assets := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_SOMA_MODEL"), os.Getenv("MOTIONBRICKS_SOMA_ASSETS")
	if library == "" || bundle == "" || assets == "" {
		t.Skip("SOMA assets not configured")
	}
	device := mb.DeviceCPU
	if os.Getenv("MOTIONBRICKS_DEVICE") == "vulkan" {
		device = mb.DeviceVulkan
	}
	s, err := loadSOMAServer(library, bundle, assets, device)
	if err != nil {
		t.Fatal(err)
	}
	defer s.model.Close()
	if s.controllerAssets != nil {
		t.Skip("controller assets covered by TestUpstreamController")
	}
	idle := s.style("idle")
	world := somaWorldPositions(idle.Local, s.assets.Skeleton.Parents)
	for i, v := range idle.World {
		if math.Abs(float64(world[i]-v)) > 2e-5 {
			t.Fatalf("FK differs from upstream at %d: %g vs %g", i, world[i], v)
		}
	}
	p := somaStreamPlanner{s}
	current, err := p.plan(nil, planRequest{Style: "idle", Facing: [2]float32{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	commands := []planRequest{
		{Style: "walk", Move: [2]float32{0, 1}, Facing: [2]float32{0, 1}},
		{Style: "walk", Move: [2]float32{1, 0}, Facing: [2]float32{1, 0}},
		{Style: "walk_zombie", Move: [2]float32{0, -1}, Facing: [2]float32{0, -1}},
		{Style: "idle", Facing: [2]float32{0, -1}},
	}
	for _, style := range s.assets.Styles {
		if style.Name != "idle" {
			commands = append(commands, planRequest{Style: style.Name, Move: [2]float32{0, 1}, Facing: [2]float32{0, 1}})
		}
	}
	for _, command := range commands {
		timeline := referenceTimeline{motion: current}
		// Replan within the current clip, not just from its endpoint.
		at := min(16, int(current.Frames)-1)
		context := timeline.context(at + p.contextOffset())
		next, err := p.plan(context, command)
		if err != nil {
			t.Fatal(err)
		}
		if err = validateStreamMotion(next); err != nil {
			t.Fatal(err)
		}
		if next.Joints != 77 || len(next.LocalPositions) != int(next.Frames*next.Joints)*3 {
			t.Fatal("incomplete SOMA pose")
		}
		oldRoot, oldRot := timeline.sample(float64(at))
		for i, v := range next.Roots[:3] {
			if math.Abs(float64(v-oldRoot[i])) > 2e-5 {
				t.Fatal("context boundary shifted")
			}
		}
		for j := 0; j < 77; j++ {
			dot := 0.0
			for k := 0; k < 4; k++ {
				dot += float64(next.Rotations[j*4+k]) * float64(oldRot[j*4+k])
			}
			if math.Abs(math.Abs(dot)-1) > 2e-5 {
				t.Fatalf("context rotation shifted at joint %d: dot=%g", j, dot)
			}
		}
		oldPositions := sampleLocalPositions(current, float64(at))
		for i, v := range next.LocalPositions[:231] {
			if math.Abs(float64(v-oldPositions[i])) > 2e-5 {
				t.Fatal("context translations shifted")
			}
		}
		dx, dz := next.Roots[len(next.Roots)-3]-next.Roots[0], next.Roots[len(next.Roots)-1]-next.Roots[2]
		if command.Move == [2]float32{} {
			if math.Hypot(float64(dx), float64(dz)) > .05 {
				t.Fatal("idle target moved")
			}
		} else if dx*command.Move[0]+dz*command.Move[1] <= 0 {
			t.Fatal("wrong movement direction")
		}
		// Exercise the same replacement/blending machinery used for G1.
		if err = timeline.replace(at, next, 4, true); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s: %d joints, replanned at frame %d, delta %.3f %.3f", command.Style, next.Joints, at, dx, dz)
		current = timeline.motion
	}

}

func TestDualModelBrowser(t *testing.T) {
	url, chrome := os.Getenv("MOTIONBRICKS_DEMO_URL"), os.Getenv("MOTIONBRICKS_CHROME")
	if url == "" || chrome == "" {
		t.Skip("live dual-model demo and Chromium required")
	}
	opts := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	opts = append(opts, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true), chromedp.Flag("use-gl", "angle"), chromedp.Flag("use-angle", "swiftshader"), chromedp.Flag("enable-unsafe-swiftshader", true))
	alloc, cancel := chromedp.NewExecAllocator(context.Background(), opts...)
	defer cancel()
	browser, closeBrowser := chromedp.NewContext(alloc)
	defer closeBrowser()
	ctx, stop := context.WithTimeout(browser, 90*time.Second)
	defer stop()
	switchModel := func(model string) chromedp.Action {
		return chromedp.ActionFunc(func(ctx context.Context) error {
			_, err := chromedp.RunResponse(ctx, chromedp.Evaluate(fmt.Sprintf(`document.querySelector('#model-select').value=%q;document.querySelector('#model-select').dispatchEvent(new Event('change'));`, model), nil))
			return err
		})
	}
	var turnLatency float64
	var result string
	var png []byte
	err := chromedp.Run(ctx, chromedp.EmulateViewport(1280, 900), chromedp.Navigate(url+"/?model=soma&qa=1"),
		chromedp.Poll(`document.documentElement.dataset.testStatus==='ready'`, nil),
		chromedp.Poll(`window.__motionBricksQA?.snapshot()?.joints===77&&document.documentElement.dataset.streamConnected==='true'`, nil),
		chromedp.Click(`.pad button[data-key="w"]`),
		chromedp.Poll(`JSON.parse(document.documentElement.dataset.streamAck||'{}').state==='applied'`, nil),
		chromedp.Poll(`(()=>{const m=window.__motionBricksQA.motion();return Math.hypot(m.roots.at(-3)-m.roots[0],m.roots.at(-1)-m.roots[2])>.001})()`, nil),
		chromedp.Evaluate(`JSON.stringify({models:[...document.querySelector('#model-select').options].map(o=>o.value),joints:window.__motionBricksQA.snapshot().joints,kimodo:document.querySelector('#kimodo-controls').hidden,physics:document.querySelector('#live-physics').disabled})`, &result),
		chromedp.Evaluate(`window.__turnStart=performance.now();window.__turnSeq=JSON.parse(document.documentElement.dataset.streamAck||'{}').seq||0;`, nil),
		chromedp.Click(`.pad button[data-key="a"]`),
		chromedp.Poll(`(()=>{const ack=JSON.parse(document.documentElement.dataset.streamAck||'{}');return ack.seq>window.__turnSeq&&ack.state==='applied'&&Number(document.documentElement.dataset.streamPlaybackTime)>=ack.time})()`, nil),
		chromedp.Evaluate(`performance.now()-window.__turnStart`, &turnLatency),
		chromedp.ScrollIntoView("#model-select"),
		chromedp.FullScreenshot(&png, 90),
		chromedp.Evaluate(`document.querySelector("#style-select").value="zombie_1d";document.querySelector("#style-select").dispatchEvent(new Event("change"))`, nil),
		chromedp.Poll(`document.querySelector("#plan-info").textContent.includes("zombie")`, nil),
		chromedp.Click(`.pad button[data-key="a"]`),
		chromedp.Sleep(2*time.Second),
		chromedp.Poll(`document.documentElement.dataset.streamError===''`, nil),
		switchModel("g1"),
		chromedp.Poll(`new URL(location.href).searchParams.get('model')==='g1'&&document.documentElement.dataset.testStatus==='ready'`, nil),
		chromedp.Poll(`window.__motionBricksQA?.snapshot()?.joints===34`, nil),
		switchModel("soma"),
		chromedp.Poll(`new URL(location.href).searchParams.get('model')==='soma'&&window.__motionBricksQA?.snapshot()?.joints===77`, nil))
	if err != nil {
		var detail string
		_ = chromedp.Run(ctx, chromedp.Evaluate(`document.body.innerText`, &detail))
		t.Fatalf("%v: %s", err, detail)
	}
	if os.Getenv("MOTIONBRICKS_SOMA_ALL_STYLES") != "" {
		var styles []string
		if err := chromedp.Run(ctx, chromedp.Evaluate(`Array.from(document.querySelector('#style-select').options,o=>o.value)`, &styles), chromedp.Click(`.pad button[data-key="w"]`)); err != nil {
			t.Fatal(err)
		}
		if len(styles) != 9 {
			t.Fatalf("expected nine upstream SOMA locomotion modes, got %v", styles)
		}
		for _, style := range styles {
			if err := chromedp.Run(ctx,
				chromedp.Evaluate(fmt.Sprintf(`window.__styleSeq=JSON.parse(document.documentElement.dataset.streamAck||'{}').seq||0;(()=>{const select=document.querySelector('#style-select');select.value=%q;select.dispatchEvent(new Event('change'));})()`, style), nil),
				chromedp.Poll(`(()=>{const ack=JSON.parse(document.documentElement.dataset.streamAck||'{}');return ack.seq>window.__styleSeq&&ack.state==='applied'&&Number(document.documentElement.dataset.streamPlaybackTime)>=ack.time})()`, nil)); err != nil {
				t.Fatalf("style %s: %v", style, err)
			}
			t.Logf("rendered style: %s", style)
		}
	}
	var observed struct {
		Models          []string `json:"models"`
		Joints          int      `json:"joints"`
		Kimodo, Physics bool
	}
	if err = json.Unmarshal([]byte(result), &observed); err != nil {
		t.Fatal(err)
	}
	if len(observed.Models) != 2 || observed.Models[0] != "g1" || observed.Models[1] != "soma" || observed.Joints != 77 || !observed.Kimodo || !observed.Physics {
		t.Fatalf("incorrect model controls: %s", result)
	}
	t.Log(result)
	t.Logf("direction change reached rendered playback in %.0f ms", turnLatency)
	if turnLatency > 1200 {
		t.Fatalf("direction change too slow: %.0f ms", turnLatency)
	}
	if len(png) < 10000 {
		t.Fatal("empty screenshot")
	}
	if path := os.Getenv("MOTIONBRICKS_SOMA_SCREENSHOT"); path != "" {
		if err = os.WriteFile(path, png, 0600); err != nil {
			t.Fatal(err)
		}
	}
}

// somaWorldPositions applies the full local skeleton, including authored child
// translations and nonidentity reference rotations (unlike the G1 rest rig).
func somaWorldPositions(local []float32, parents []int32) []float32 {
	n := len(parents)
	world := make([]float32, len(local))
	positions := make([]float32, 0, len(local)/7*3)
	for f := 0; f < len(local)/(n*7); f++ {
		for j, parent := range parents {
			at := (f*n + j) * 7
			p := local[at : at+7]
			out := world[at : at+7]
			if parent < 0 {
				copy(out, p)
			} else {
				base := (f*n + int(parent)) * 7
				b := world[base : base+7]
				x, y, z, w := b[3], b[4], b[5], b[6]
				tx, ty, tz := 2*(y*p[2]-z*p[1]), 2*(z*p[0]-x*p[2]), 2*(x*p[1]-y*p[0])
				out[0] = b[0] + p[0] + w*tx + y*tz - z*ty
				out[1] = b[1] + p[1] + w*ty + z*tx - x*tz
				out[2] = b[2] + p[2] + w*tz + x*ty - y*tx
				qx, qy, qz, qw := p[3], p[4], p[5], p[6]
				out[3] = w*qx + x*qw + y*qz - z*qy
				out[4] = w*qy - x*qz + y*qw + z*qx
				out[5] = w*qz + x*qy - y*qx + z*qw
				out[6] = w*qw - x*qx - y*qy - z*qz
			}
			positions = append(positions, out[:3]...)
		}
	}
	return positions
}
