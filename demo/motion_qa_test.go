package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/chromedp/chromedp"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

type qaMotion struct {
	Frames      int       `json:"frames"`
	Joints      int       `json:"joints"`
	Roots       []float64 `json:"roots"`
	Rotations   []float64 `json:"rotations"`
	EntryFrames int       `json:"entry_frames"`
	ClipFrames  int       `json:"clip_frames"`
}

type qaSnapshot struct {
	Label          string    `json:"label"`
	Frame          int       `json:"frame"`
	Root           []float64 `json:"root"`
	Rotations      []float64 `json:"rotations"`
	JointPositions []float64 `json:"joint_positions"`
	MotionFrames   int       `json:"motion_frames"`
	Joints         int       `json:"joints"`
	KimodoState    string    `json:"kimodo_state"`
	ControlsLocked bool      `json:"controls_locked"`
	EntryFrames    int       `json:"entry_frames"`
	ClipFrames     int       `json:"clip_frames"`
	ClipID         string    `json:"clip_id"`
	Screenshot     string    `json:"screenshot,omitempty"`
}

type qaStats struct {
	MaxRootStepMetres        float64 `json:"max_root_step_metres"`
	MaxRootSpeedMPS          float64 `json:"max_root_speed_mps"`
	MaxRootAccelerationMPS2  float64 `json:"max_root_acceleration_mps2"`
	MaxJointAngleStepDegrees float64 `json:"max_joint_angle_step_degrees"`
	MaxJointWorldStepMetres  float64 `json:"max_joint_world_step_metres"`
	MaxJointWorldSpeedMPS    float64 `json:"max_joint_world_speed_mps"`
	MaxQuaternionNormError   float64 `json:"max_quaternion_norm_error"`
	MaxRootExcursionMetres   float64 `json:"max_root_excursion_metres"`
}

type qaCheck struct {
	Value  float64 `json:"value"`
	Limit  float64 `json:"limit"`
	Unit   string  `json:"unit"`
	Passed bool    `json:"passed"`
	Reason string  `json:"reason"`
}

type qaReport struct {
	Format       string             `json:"format"`
	CreatedAt    string             `json:"created_at"`
	Device       string             `json:"device"`
	Clip         kimodoClipInfo     `json:"clip"`
	Passed       bool               `json:"passed"`
	Component    map[string]qaStats `json:"component_statistics"`
	Checks       map[string]qaCheck `json:"checks"`
	Snapshots    []qaSnapshot       `json:"snapshots"`
	Observations []string           `json:"observations"`
}

func qaDistance(a, b []float64) float64 {
	var square float64
	for axis := 0; axis < 3; axis++ {
		difference := a[axis] - b[axis]
		square += difference * difference
	}
	return math.Sqrt(square)
}

func qaQuaternionAngle(a, b []float64) float64 {
	dot := math.Abs(a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3])
	return 2 * math.Acos(math.Min(1, dot)) * 180 / math.Pi
}

func qaQuaternionMultiply(a, b []float64) [4]float64 {
	return [4]float64{
		a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
		a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
		a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
		a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
	}
}

func qaRotate(q []float64, value [3]float64) [3]float64 {
	vector := []float64{value[0], value[1], value[2], 0}
	conjugate := []float64{-q[0], -q[1], -q[2], q[3]}
	left := qaQuaternionMultiply(q, vector)
	result := qaQuaternionMultiply(left[:], conjugate)
	return [3]float64{result[0], result[1], result[2]}
}

func qaWorldPositions(motion qaMotion, joints []mb.Joint) []float64 {
	positions := make([]float64, motion.Frames*motion.Joints*3)
	worldRotations := make([]float64, motion.Joints*4)
	for frame := 0; frame < motion.Frames; frame++ {
		for joint := 0; joint < motion.Joints; joint++ {
			rotation := motion.Rotations[(frame*motion.Joints+joint)*4 : (frame*motion.Joints+joint+1)*4]
			positionOffset := (frame*motion.Joints + joint) * 3
			if joints[joint].Parent < 0 {
				copy(positions[positionOffset:positionOffset+3], motion.Roots[frame*3:frame*3+3])
				copy(worldRotations[joint*4:joint*4+4], rotation)
				continue
			}
			parent := int(joints[joint].Parent)
			parentPosition := positions[(frame*motion.Joints+parent)*3 : (frame*motion.Joints+parent)*3+3]
			offset := [3]float64{
				float64(joints[joint].Position[0] - joints[parent].Position[0]),
				float64(joints[joint].Position[1] - joints[parent].Position[1]),
				float64(joints[joint].Position[2] - joints[parent].Position[2]),
			}
			rotated := qaRotate(worldRotations[parent*4:parent*4+4], offset)
			for axis := 0; axis < 3; axis++ {
				positions[positionOffset+axis] = parentPosition[axis] + rotated[axis]
			}
			world := qaQuaternionMultiply(worldRotations[parent*4:parent*4+4], rotation)
			copy(worldRotations[joint*4:joint*4+4], world[:])
		}
	}
	return positions
}

func qaMotionStats(motion qaMotion, joints []mb.Joint, first, end int) qaStats {
	first = max(0, first)
	end = min(motion.Frames, end)
	statistics := qaStats{}
	if first >= end {
		return statistics
	}
	positions := qaWorldPositions(motion, joints)
	startRoot := motion.Roots[first*3 : first*3+3]
	for frame := first; frame < end; frame++ {
		root := motion.Roots[frame*3 : frame*3+3]
		statistics.MaxRootExcursionMetres = max(statistics.MaxRootExcursionMetres, qaDistance(root, startRoot))
		for joint := 0; joint < motion.Joints; joint++ {
			q := motion.Rotations[(frame*motion.Joints+joint)*4 : (frame*motion.Joints+joint+1)*4]
			norm := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
			statistics.MaxQuaternionNormError = max(statistics.MaxQuaternionNormError, math.Abs(norm-1))
		}
		if frame == first {
			continue
		}
		previousRoot := motion.Roots[(frame-1)*3 : (frame-1)*3+3]
		rootStep := qaDistance(root, previousRoot)
		statistics.MaxRootStepMetres = max(statistics.MaxRootStepMetres, rootStep)
		for joint := 0; joint < motion.Joints; joint++ {
			currentQ := motion.Rotations[(frame*motion.Joints+joint)*4 : (frame*motion.Joints+joint+1)*4]
			previousQ := motion.Rotations[((frame-1)*motion.Joints+joint)*4 : ((frame-1)*motion.Joints+joint+1)*4]
			statistics.MaxJointAngleStepDegrees = max(statistics.MaxJointAngleStepDegrees, qaQuaternionAngle(currentQ, previousQ))
			currentPosition := positions[(frame*motion.Joints+joint)*3 : (frame*motion.Joints+joint)*3+3]
			previousPosition := positions[((frame-1)*motion.Joints+joint)*3 : ((frame-1)*motion.Joints+joint)*3+3]
			statistics.MaxJointWorldStepMetres = max(statistics.MaxJointWorldStepMetres, qaDistance(currentPosition, previousPosition))
		}
		if frame >= first+2 {
			previousPreviousRoot := motion.Roots[(frame-2)*3 : (frame-2)*3+3]
			var velocityChange float64
			for axis := 0; axis < 3; axis++ {
				value := root[axis] - 2*previousRoot[axis] + previousPreviousRoot[axis]
				velocityChange += value * value
			}
			statistics.MaxRootAccelerationMPS2 = max(statistics.MaxRootAccelerationMPS2, math.Sqrt(velocityChange)*kimodoFPS*kimodoFPS)
		}
	}
	statistics.MaxRootSpeedMPS = statistics.MaxRootStepMetres * kimodoFPS
	statistics.MaxJointWorldSpeedMPS = statistics.MaxJointWorldStepMetres * kimodoFPS
	return statistics
}

func qaFromMotion(motion *mb.Motion) qaMotion {
	result := qaMotion{Frames: int(motion.Frames), Joints: int(motion.Joints), Roots: make([]float64, len(motion.Roots)), Rotations: make([]float64, len(motion.Rotations))}
	for index, value := range motion.Roots {
		result.Roots[index] = float64(value)
	}
	for index, value := range motion.Rotations {
		result.Rotations[index] = float64(value)
	}
	return result
}

func qaPoseAngle(left qaMotion, leftFrame int, right qaMotion, rightFrame int) float64 {
	maximum := 0.0
	for joint := 0; joint < left.Joints; joint++ {
		leftQ := left.Rotations[(leftFrame*left.Joints+joint)*4 : (leftFrame*left.Joints+joint+1)*4]
		rightQ := right.Rotations[(rightFrame*right.Joints+joint)*4 : (rightFrame*right.Joints+joint+1)*4]
		maximum = max(maximum, qaQuaternionAngle(leftQ, rightQ))
	}
	return maximum
}

func TestKimodoMotionQA(t *testing.T) {
	directory := os.Getenv("MOTIONBRICKS_KIMODO_DIR")
	artifactDirectory := os.Getenv("MOTIONBRICKS_MOTION_QA_DIR")
	libraryPath, modelPath, stylesPath := os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES")
	if directory == "" || artifactDirectory == "" || libraryPath == "" || modelPath == "" || stylesPath == "" {
		t.Skip("set native paths, MOTIONBRICKS_KIMODO_DIR, and MOTIONBRICKS_MOTION_QA_DIR to run observational motion QA")
	}
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("chromium is not installed")
		}
	}
	deviceName := os.Getenv("MOTIONBRICKS_MOTION_QA_DEVICE")
	if deviceName == "" {
		deviceName = "cpu"
	}
	device, err := parseDevice(deviceName)
	if err != nil {
		t.Fatal(err)
	}
	demo, err := loadDemoServer(libraryPath, modelPath, stylesPath, device)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	if err = demo.loadKimodo(directory); err != nil {
		t.Fatal(err)
	}
	if len(demo.clips) == 0 {
		t.Fatal("no compatible Kimodo G1 clips were loaded")
	}
	clipID := os.Getenv("MOTIONBRICKS_MOTION_QA_CLIP")
	if clipID == "" {
		clips := append([]kimodoClipInfo(nil), demo.clips...)
		sort.Slice(clips, func(i, j int) bool { return clips[i].Frames < clips[j].Frames })
		clipID = clips[0].ID
	}
	clip := demo.kimodo[clipID]
	if clip == nil {
		t.Fatalf("unknown QA clip %q", clipID)
	}
	if err = os.MkdirAll(artifactDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(demo.routes())
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true), chromedp.Flag("use-angle", "swiftshader"),
		chromedp.Flag("enable-unsafe-swiftshader", true), chromedp.WindowSize(1280, 800))
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	ctx, cancel := context.WithTimeout(browser, 60*time.Second)
	defer cancel()
	waitFor := func(expression, description string) chromedp.Action {
		return chromedp.ActionFunc(func(ctx context.Context) error {
			deadline := time.Now().Add(30 * time.Second)
			for time.Now().Before(deadline) {
				var ready bool
				if evaluateErr := chromedp.Evaluate(expression, &ready).Do(ctx); evaluateErr != nil {
					return evaluateErr
				}
				if ready {
					return nil
				}
				time.Sleep(50 * time.Millisecond)
			}
			return fmt.Errorf("%w for %s", errors.New("timeout waiting for browser"), description)
		})
	}
	var initial qaMotion
	if err = chromedp.Run(ctx,
		chromedp.Navigate(server.URL+"/?qa=1"),
		waitFor(`document.documentElement.dataset.testStatus === "ready" && document.documentElement.dataset.motionQA === "available"`, "QA hook"),
		chromedp.Evaluate(`window.__motionBricksQA.pause(true)`, nil),
		chromedp.Evaluate(`window.__motionBricksQA.motion()`, &initial),
		chromedp.Evaluate(fmt.Sprintf(`document.querySelector('#kimodo-select').value = %q`, clipID), nil),
		chromedp.Click(`#kimodo-play`, chromedp.ByQuery),
		waitFor(`document.documentElement.dataset.kimodoState === "playing"`, "Kimodo playback"),
	); err != nil {
		t.Fatal(err)
	}
	var authored qaMotion
	if err = chromedp.Run(ctx, chromedp.Evaluate(`window.__motionBricksQA.motion()`, &authored)); err != nil {
		t.Fatal(err)
	}
	prefix := strings.ReplaceAll(filepath.Base(clipID), string(filepath.Separator), "-")
	report := qaReport{Format: "motionbricks-observational-motion-qa-v1", CreatedAt: time.Now().UTC().Format(time.RFC3339), Device: deviceName,
		Clip: clip.Info, Passed: true, Component: make(map[string]qaStats), Checks: make(map[string]qaCheck)}
	capture := func(label string, frame int) {
		t.Helper()
		var snapshot qaSnapshot
		var screenshot []byte
		if captureErr := chromedp.Run(ctx,
			chromedp.Evaluate(fmt.Sprintf(`window.__motionBricksQA.setFrame(%d)`, frame), nil),
			chromedp.Sleep(80*time.Millisecond),
			chromedp.Evaluate(`window.__motionBricksQA.snapshot()`, &snapshot),
			chromedp.FullScreenshot(&screenshot, 90)); captureErr != nil {
			t.Fatal(captureErr)
		}
		name := fmt.Sprintf("%s-%02d-%s.png", prefix, len(report.Snapshots), label)
		if writeErr := os.WriteFile(filepath.Join(artifactDirectory, name), screenshot, 0o600); writeErr != nil {
			t.Fatal(writeErr)
		}
		snapshot.Label, snapshot.Screenshot = label, name
		report.Snapshots = append(report.Snapshots, snapshot)
	}
	entry, sourceFrames := authored.EntryFrames, authored.ClipFrames
	capture("entry-start", 0)
	capture("entry-middle", entry/2)
	capture("entry-last", entry-1)
	capture("authored-first", entry)
	capture("authored-quarter", entry+(sourceFrames-1)/4)
	capture("authored-middle", entry+(sourceFrames-1)/2)
	capture("authored-three-quarter", entry+3*(sourceFrames-1)/4)
	capture("authored-near-end", authored.Frames-2)
	if err = chromedp.Run(ctx,
		chromedp.Evaluate(`window.__motionBricksQA.pause(false)`, nil),
		chromedp.Evaluate(fmt.Sprintf(`window.__motionBricksQA.setFrame(%d)`, authored.Frames-1), nil),
		waitFor(`document.documentElement.dataset.kimodoState === "complete"`, "MotionBricks exit plan"),
		chromedp.Evaluate(`window.__motionBricksQA.pause(true)`, nil),
	); err != nil {
		t.Fatal(err)
	}
	var exit qaMotion
	if err = chromedp.Run(ctx, chromedp.Evaluate(`window.__motionBricksQA.motion()`, &exit)); err != nil {
		t.Fatal(err)
	}
	capture("exit-first", 0)
	capture("exit-blend-last", min(3, exit.Frames-1))
	capture("exit-settled", min(8, exit.Frames-1))

	raw := qaFromMotion(&clip.Motion)
	report.Component["motionbricks_before"] = qaMotionStats(initial, demo.joints, 0, initial.Frames)
	report.Component["kimodo_source"] = qaMotionStats(raw, demo.joints, 0, raw.Frames)
	report.Component["kimodo_opening"] = qaMotionStats(raw, demo.joints, 0, min(8, raw.Frames))
	report.Component["stitched_sequence"] = qaMotionStats(authored, demo.joints, 0, authored.Frames)
	report.Component["motionbricks_after"] = qaMotionStats(exit, demo.joints, 0, exit.Frames)
	report.Component["entry_transition"] = qaMotionStats(authored, demo.joints, 0, entry+1)
	report.Component["kimodo_tail"] = qaMotionStats(raw, demo.joints, max(0, raw.Frames-8), raw.Frames)
	report.Component["exit_opening"] = qaMotionStats(exit, demo.joints, 0, min(8, exit.Frames))

	addCheck := func(name string, value, limit float64, unit, reason string) {
		passed := !math.IsNaN(value) && !math.IsInf(value, 0) && value <= limit
		report.Checks[name] = qaCheck{Value: value, Limit: limit, Unit: unit, Passed: passed, Reason: reason}
		report.Passed = report.Passed && passed
	}
	before, source, stitched := report.Component["motionbricks_before"], report.Component["kimodo_source"], report.Component["stitched_sequence"]
	sourceOpening := report.Component["kimodo_opening"]
	entryStats, tail, opening := report.Component["entry_transition"], report.Component["kimodo_tail"], report.Component["exit_opening"]
	addCheck("quaternion_norm", stitched.MaxQuaternionNormError, 1e-3, "absolute error", "all stitched local rotations remain unit quaternions")
	addCheck("root_speed_absolute", stitched.MaxRootSpeedMPS, 15, "m/s", "reject map-scale single-frame root teleports")
	addCheck("root_acceleration_absolute", stitched.MaxRootAccelerationMPS2, 120, "m/s^2", "reject implausible one-frame root direction changes")
	addCheck("joint_speed_absolute", stitched.MaxJointWorldSpeedMPS, 25, "m/s", "reject explosive whole-body or limb movement")
	addCheck("joint_angle_absolute", stitched.MaxJointAngleStepDegrees, 120, "degrees/frame", "reject single-frame local-joint flips")
	addCheck("root_excursion", stitched.MaxRootExcursionMetres, source.MaxRootExcursionMetres+0.5, "m", "alignment may translate/rotate but must preserve the source trajectory scale")
	addCheck("entry_root_step_component_envelope", entryStats.MaxRootStepMetres,
		max(0.15, 3*max(before.MaxRootStepMetres, sourceOpening.MaxRootStepMetres)), "m/frame", "entry must stay near the opening displacement envelope of its components")
	addCheck("entry_joint_step_component_envelope", entryStats.MaxJointAngleStepDegrees,
		max(12, 3*max(before.MaxJointAngleStepDegrees, sourceOpening.MaxJointAngleStepDegrees)), "degrees/frame", "entry slerp must not introduce a much faster joint step than its opening components imply")
	addCheck("entry_joint_world_step_component_envelope", entryStats.MaxJointWorldStepMetres,
		max(0.20, 3*max(before.MaxJointWorldStepMetres, sourceOpening.MaxJointWorldStepMetres)), "m/frame", "entry must not sweep limbs implausibly far in one frame")
	entryStartGap := qaDistance(initial.Roots[:3], authored.Roots[:3])
	addCheck("entry_start_root_gap", entryStartGap, 0.05, "m", "authored playback starts at the visible MotionBricks root")
	exitRootGap := qaDistance(authored.Roots[(authored.Frames-1)*3:], exit.Roots[:3])
	addCheck("exit_root_gap", exitRootGap, max(0.25, 4*tail.MaxRootStepMetres), "m", "first regenerated root remains near the closing Kimodo context")
	exitPoseGap := qaPoseAngle(authored, authored.Frames-1, exit, 0)
	addCheck("exit_pose_gap", exitPoseGap, max(60, 3*max(tail.MaxJointAngleStepDegrees, before.MaxJointAngleStepDegrees)), "degrees", "context blend prevents an unexplained whole-pose snap")
	addCheck("exit_root_speed_absolute", opening.MaxRootSpeedMPS, 15, "m/s", "reject a teleport in the regenerated opening")
	addCheck("exit_root_acceleration_absolute", opening.MaxRootAccelerationMPS2, 120, "m/s^2", "reject explosive regenerated root acceleration")
	addCheck("exit_joint_speed_absolute", opening.MaxJointWorldSpeedMPS, 25, "m/s", "reject explosive regenerated limb movement")
	addCheck("exit_joint_angle_absolute", opening.MaxJointAngleStepDegrees, 120, "degrees/frame", "reject regenerated single-frame joint flips")
	if report.Passed {
		report.Observations = append(report.Observations, "No numeric teleport, quaternion corruption, single-frame limb explosion, or transition-boundary excursion exceeded the configured component-relative and absolute limits.")
	} else {
		if !report.Checks["entry_joint_step_component_envelope"].Passed || !report.Checks["entry_joint_world_step_component_envelope"].Passed {
			report.Observations = append(report.Observations, "The entry blend moves limbs substantially faster than either source component; inspect the entry screenshots for sweeping, floor penetration, or implausible inversion.")
		}
		if !report.Checks["exit_root_gap"].Passed || !report.Checks["exit_root_speed_absolute"].Passed {
			report.Observations = append(report.Observations, "The MotionBricks restart leaves the closing Kimodo root envelope and must be treated as a teleport, not accepted as a new baseline.")
		}
		if !report.Checks["exit_pose_gap"].Passed || !report.Checks["exit_joint_angle_absolute"].Passed {
			report.Observations = append(report.Observations, "The exit pose contains an abrupt local-joint discontinuity; inspect the exit screenshots for folding or limb flips.")
		}
	}
	reportPath := filepath.Join(artifactDirectory, prefix+"-report.json")
	data, marshalErr := json.MarshalIndent(report, "", "  ")
	if marshalErr != nil {
		t.Fatal(marshalErr)
	}
	data = append(data, '\n')
	if err = os.WriteFile(reportPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	for name, check := range report.Checks {
		t.Logf("%-38s %9.4f <= %-9.4f %-14s pass=%v", name, check.Value, check.Limit, check.Unit, check.Passed)
	}
	t.Logf("motion QA report: %s (%d screenshots)", reportPath, len(report.Snapshots))
	if !report.Passed {
		t.Fatalf("motion QA bounds failed; inspect %s", reportPath)
	}
}
