package main

import (
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
	"os"
	"testing"
)

// Measure actual foot clearance, not merely a requested target speed.
func TestJumpFootClearance(t *testing.T) {
	if os.Getenv("MOTIONBRICKS_LIB") == "" || os.Getenv("MOTIONBRICKS_MODEL") == "" || os.Getenv("MOTIONBRICKS_STYLES") == "" {
		t.Skip("native demo paths are not configured")
	}
	demo, err := loadDemoServer(os.Getenv("MOTIONBRICKS_LIB"), os.Getenv("MOTIONBRICKS_MODEL"), os.Getenv("MOTIONBRICKS_STYLES"), mb.DeviceCPU)
	if err != nil {
		t.Fatal(err)
	}
	defer demo.Close()
	for _, speed := range []float32{2, 5} {
		agent, err := demo.model.NewAgent()
		if err != nil {
			t.Fatal(err)
		}
		if err = agent.Reset(demo.styles["walk"]); err != nil {
			t.Fatal(err)
		}
		item := &session{agent: agent}
		for i := 0; i < 4; i++ {
			_, err = demo.commandPlan(item, demo.styles["walk"], planRequest{Move: [2]float32{0, 1}, Facing: [2]float32{0, 1}, Advance: 16, Seed: 10})
			if err != nil {
				t.Fatal(err)
			}
		}
		motion, err := demo.commandPlan(item, demo.styles["walk"], planRequest{Move: [2]float32{0, 1}, Facing: [2]float32{0, 1}, Advance: 16, Speed: &speed, Seed: 10})
		if err != nil {
			t.Fatal(err)
		}
		sample := qaMotion{Frames: int(motion.Frames), Joints: 34}
		for _, x := range motion.Roots {
			sample.Roots = append(sample.Roots, float64(x))
		}
		for _, x := range motion.Rotations {
			sample.Rotations = append(sample.Rotations, float64(x))
		}
		positions := qaWorldPositions(sample, demo.joints)
		maxHeight, maxClearance, apex := 0.0, -math.MaxFloat64, 0
		for f := 0; f < sample.Frames; f++ {
			maxHeight = math.Max(maxHeight, sample.Roots[f*3+1])
			clearance := math.Inf(1)
			for _, j := range []int{6, 7, 13, 14} {
				clearance = math.Min(clearance, positions[(f*34+j)*3+1])
			}
			if clearance > maxClearance {
				maxClearance = clearance
				apex = f
			}
		}
		t.Logf("speed=%.1f frames=%d rootHeight=%.3f bothFeetClearance=%.3f apex=%d", speed, motion.Frames, maxHeight, maxClearance, apex)
		if speed == 5 && (maxClearance < 0.12 || maxHeight > 1.2 || apex <= 16) {
			t.Errorf("expected a bounded airborne phase after the normal walking replan boundary")
		}
		agent.Close()
	}
}
