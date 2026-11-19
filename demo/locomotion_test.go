package main

import (
	"encoding/json"
	"fmt"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func controllerPoseError(a, b []float32) (float64, float64) {
	pmax, qmax := 0., 0.
	if len(a) != len(b) {
		return math.Inf(1), math.Inf(1)
	}
	for i := 0; i < len(a); i += 7 {
		p := 0.
		dot, aa, bb := 0., 0., 0.
		for k := 0; k < 3; k++ {
			d := float64(a[i+k]) - float64(b[i+k])
			p += d * d
		}
		for k := 3; k < 7; k++ {
			x, y := float64(a[i+k]), float64(b[i+k])
			dot += x * y
			aa += x * x
			bb += y * y
		}
		pmax = math.Max(pmax, math.Sqrt(p))
		qmax = math.Max(qmax, 2*math.Acos(math.Min(1, math.Abs(dot)/math.Sqrt(aa*bb)))*180/math.Pi)
	}
	return pmax, qmax
}
func TestUpstreamController(t *testing.T) {
	root := os.Getenv("MOTIONBRICKS_CONTROLLER_CAPTURE")
	if root == "" {
		t.Skip("set MOTIONBRICKS_CONTROLLER_CAPTURE")
	}
	a, e := loadLocomotionAssets(filepath.Join(root, "controller-assets.json"))
	if e != nil {
		t.Fatal(e)
	}
	var manifest struct {
		Oracle struct{ Precision string }
		Arrays map[string]controllerArray
		Frames []struct {
			Tick                                            int
			Mode                                            string
			Move, Facing                                    []float32
			Fast                                            bool
			Speed                                           float64
			Velocity, CurrentFacing, Ground, ActualVelocity []float32 `json:"-"`
			PlaybackTime                                    float64   `json:"playback_time"`
		}
		Events []struct {
			Tick, Tokens, Frames int
			Allowed              []int
			Early                int `json:"transition_early_by_frames"`
		}
	}
	b, e := os.ReadFile(filepath.Join(root, "manifest.json"))
	if e != nil {
		t.Fatal(e)
	}
	if e = json.Unmarshal(b, &manifest); e != nil {
		t.Fatal(e)
	}
	if manifest.Oracle.Precision == "" {
		manifest.Oracle.Precision = "fp32"
	}
	if manifest.Oracle.Precision != "fp32" && manifest.Oracle.Precision != "mixed-fp16" {
		t.Fatal("unsupported oracle precision")
	}
	read := func(name string) []float32 {
		t.Helper()
		out, e := readControllerArray(root, manifest.Arrays[name])
		if e != nil {
			t.Fatal(e)
		}
		return out
	}
	var contract map[string]struct {
		Position float64 `json:"position_m"`
		Rotation float64 `json:"rotation_deg"`
		Speed    float64 `json:"speed_mps"`
	}
	limitsData, err := os.ReadFile("../reference/soma-controller-tolerances.json")
	if err != nil {
		t.Fatal(err)
	}
	// Read only the three numeric acceptance records; other contract fields remain documentation.
	var raw map[string]json.RawMessage
	if err = json.Unmarshal(limitsData, &raw); err != nil {
		t.Fatal(err)
	}
	contract = make(map[string]struct {
		Position float64 `json:"position_m"`
		Rotation float64 `json:"rotation_deg"`
		Speed    float64 `json:"speed_mps"`
	})
	for _, key := range []string{"oracle_playback", "native_feedback"} {
		v := contract[key]
		if err = json.Unmarshal(raw[key], &v); err != nil {
			t.Fatal(err)
		}
		contract[key] = v
	}
	limits := contract["oracle_playback"]
	played := read("poses")
	var native *somaServer
	if os.Getenv("MOTIONBRICKS_CONTROLLER_NATIVE") == "1" {
		var err error
		native, err = loadSOMAServer(os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_SOMA_MODEL"), filepath.Join(root, "controller-assets.json"), mb.DeviceVulkan)
		if err != nil {
			t.Fatal(err)
		}
		defer native.model.Close()
		limits = contract["native_feedback"]
	}
	nativePrecision := uint32(0)
	if manifest.Oracle.Precision == "mixed-fp16" {
		nativePrecision = 3
	}
	tick, event := 0, 0
	maxContext, maxTarget, maxPose, maxAngle, maxSpeed := 0., 0., 0., 0., 0.
	completedTicks := 0
	generationTicksMatch := true
	defer func() {
		if path := os.Getenv("MOTIONBRICKS_CONTROLLER_REPORT"); path != "" {
			report := map[string]any{"events": event, "frames": len(manifest.Frames), "context_position_m": maxContext, "target_position_m": maxTarget, "playback_position_m": maxPose, "rotation_deg": maxAngle, "speed_mps": maxSpeed, "native": native != nil, "limits": limits, "precision": manifest.Oracle.Precision, "completed_ticks": completedTicks, "failed": t.Failed(), "passed": !t.Failed() && completedTicks == len(manifest.Frames) && event == len(manifest.Events) && maxContext <= limits.Position && maxTarget <= limits.Position && maxPose <= limits.Position && maxAngle <= limits.Rotation && maxSpeed <= limits.Speed, "generation_ticks_match_checked_prefix": generationTicksMatch, "complete": completedTicks == len(manifest.Frames), "generation_ticks_match": generationTicksMatch && event == len(manifest.Events)}
			b, _ := json.MarshalIndent(report, "", "  ")
			if e := os.WriteFile(path, b, 0644); e != nil {
				t.Fatal(e)
			}
		}
	}()
	var c *locomotionController
	c = newLocomotionController(a, func(context, target []float32, allowed []int) ([]float32, error) {
		if event >= len(manifest.Events) {
			return nil, fmt.Errorf("extra generation tick %d", tick)
		}
		ev := manifest.Events[event]
		if ev.Tick != tick {
			generationTicksMatch = false
			return nil, fmt.Errorf("event %d tick %d expected %d speed %g", event, tick, ev.Tick, c.movement.speed)
		}
		if fmt.Sprint(allowed) != fmt.Sprint(ev.Allowed) {
			return nil, fmt.Errorf("allowed mismatch")
		}
		p, q := controllerPoseError(context, read(fmt.Sprintf("event_%04d_context", event)))
		maxContext = math.Max(maxContext, p)
		maxAngle = math.Max(maxAngle, q)
		p, q = controllerPoseError(target, read(fmt.Sprintf("event_%04d_target", event)))
		maxTarget = math.Max(maxTarget, p)
		maxAngle = math.Max(maxAngle, q)
		if p > limits.Position || q > limits.Rotation {
			return nil, fmt.Errorf("event %d target position %g rotation %g speed %g", event, p, q, c.movement.speed)
		}
		output := read(fmt.Sprintf("event_%04d_motion", event))
		event++
		if native != nil {
			got, err := native.controllerGeneratorPrecision(context, target, allowed, nativePrecision)
			if err != nil {
				return nil, err
			}
			if dir := os.Getenv("MOTIONBRICKS_CONTROLLER_TRACE"); dir != "" {
				if err := os.MkdirAll(dir, 0755); err != nil {
					return nil, err
				}
				data, _ := json.Marshal(map[string]any{"tick": tick, "context": context, "target": target, "output": got, "allowed": allowed})
				if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("event-%03d.json", event-1)), data, 0644); err != nil {
					return nil, err
				}
			}
			if len(got) != len(output) {
				return nil, fmt.Errorf("event %d native duration differs", event-1)
			}
			return got, nil
		}
		return output, nil
	})

	for _, f := range manifest.Frames {
		tick = f.Tick
		input := locomotionInput{Mode: f.Mode, Fast: f.Fast, Facing: groundVector{f.Facing[0], f.Facing[1]}}
		if len(f.Move) > 0 {
			input.Move = groundVector{f.Move[0], f.Move[1]}
		}
		before := event
		if e = c.Step(input, .02); e != nil {
			t.Fatal(e)
		}

		if before != event && c.early != manifest.Events[event-1].Early {
			t.Fatalf("tick %d transition early differs", tick)
		}
		p, q := controllerPoseError(c.pose, played[tick*a.Joints*7:(tick+1)*a.Joints*7])
		completedTicks = tick + 1
		maxPose = math.Max(maxPose, p)
		maxAngle = math.Max(maxAngle, q)
		maxSpeed = math.Max(maxSpeed, math.Abs(c.movement.speed-f.Speed))
		if p > limits.Position || q > limits.Rotation {
			t.Fatalf("tick %d pose position %g rotation %g", tick, p, q)
		}
	}

	t.Logf("events=%d context=%g target=%g playback=%g rotation=%g speed=%g", event, maxContext, maxTarget, maxPose, maxAngle, maxSpeed)
	if event != len(manifest.Events) || maxContext > limits.Position || maxTarget > limits.Position || maxPose > limits.Position || maxAngle > limits.Rotation || maxSpeed > limits.Speed {
		t.Fatal("controller parity limits exceeded")
	}
}
