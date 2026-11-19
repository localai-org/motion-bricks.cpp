// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Go adaptation of the upstream locomotion runtime; see the revision below.

package main

// Locomotion policy follows NVlabs/motionbricks app/runtime/common (Apache-2.0),
// revision 54fc563aeaaf819a753d72cc9fdc6713989cdd43. It has no transport or GPU
// dependency: generation is injected, including recorded outputs in parity tests.
import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

type controllerArray struct {
	File   string
	Shape  []int
	SHA256 string
}
type locomotionMode struct {
	Name, Label   string
	Directional   bool
	MaxSpeed      float64 `json:"max_speed"`
	DefaultSpeed  float64 `json:"default_speed"`
	MinimumSpeed  float64 `json:"minimum_speed"`
	VelocityScale float32 `json:"velocity_scale"`
	PositionScale float32 `json:"position_scale"`
	Early         int
	Allowed       []int
	Values        [][2]float32
	Targets       controllerArray
	poses         []float32
}
type locomotionClip struct {
	Name           string
	Rate, Duration float64
	SourceGround   [7]float32 `json:"source_ground"`
	Poses          controllerArray
	poses          []float32
}
type locomotionAssets struct {
	Format         string
	SourceRevision string `json:"source_revision"`
	Joints         int
	Names          []string
	Parents        []int32
	Reference      []float32
	ModelToPose    []uint32   `json:"model_to_pose"`
	ModelToRuntime [7]float32 `json:"model_to_runtime"`
	Modes          []*locomotionMode
	Clips          []*locomotionClip
}

func readControllerArray(root string, a controllerArray) ([]float32, error) {
	if filepath.Base(a.File) != a.File {
		return nil, fmt.Errorf("invalid controller array path")
	}
	b, e := os.ReadFile(filepath.Join(root, a.File))
	if e != nil {
		return nil, e
	}
	hash := sha256.Sum256(b)
	n := 1
	for _, v := range a.Shape {
		if v <= 0 || v > 1000000 {
			return nil, fmt.Errorf("invalid controller array shape")
		}
		n *= v
	}
	if len(b) != n*4 || hex.EncodeToString(hash[:]) != a.SHA256 {
		return nil, fmt.Errorf("controller array integrity: %s", a.File)
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			return nil, fmt.Errorf("nonfinite controller array")
		}
	}
	return out, nil
}
func loadLocomotionAssets(path string) (*locomotionAssets, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return nil, e
	}
	a := new(locomotionAssets)
	if e = json.Unmarshal(b, a); e != nil {
		return nil, e
	}
	if a.SourceRevision != "54fc563aeaaf819a753d72cc9fdc6713989cdd43" || a.Format != "motionbricks-controller-assets-v1" || a.Joints < 30 || len(a.Reference) != a.Joints*7 || len(a.Modes) != 9 || len(a.Clips) != 2 {
		return nil, fmt.Errorf("invalid controller assets")
	}
	seen := map[string]bool{}
	for _, m := range a.Modes {
		if m.Name == "" || seen[m.Name] || m.MaxSpeed <= 0 || m.DefaultSpeed <= 0 || m.DefaultSpeed > m.MaxSpeed || m.MinimumSpeed < 0 || m.MinimumSpeed > m.DefaultSpeed || m.Early < 0 || m.Early > 17 || len(m.Allowed) == 0 || len(m.Values) < 2 || m.Directional && len(m.Values) != 5 {
			return nil, fmt.Errorf("invalid locomotion mode")
		}
		seen[m.Name] = true
		for _, n := range m.Allowed {
			if n < 6 || n > 16 {
				return nil, fmt.Errorf("invalid allowed duration")
			}
		}
		m.poses, e = readControllerArray(filepath.Dir(path), m.Targets)
		if e != nil {
			return nil, e
		}
		if len(m.poses) != len(m.Values)*4*a.Joints*7 {
			return nil, fmt.Errorf("invalid mode targets")
		}
	}
	for _, c := range a.Clips {
		if c.Rate <= 0 || c.Duration <= 0 {
			return nil, fmt.Errorf("invalid idle timing")
		}
		c.poses, e = readControllerArray(filepath.Dir(path), c.Poses)
		if e != nil {
			return nil, e
		}
		if len(c.poses) < 2*a.Joints*7 || len(c.poses)%(a.Joints*7) != 0 {
			return nil, fmt.Errorf("invalid idle poses")
		}
	}
	if a.Clips[0].Name != "standing" || a.Clips[1].Name != "crouching" || !seen["linear_1d"] || !seen["directional_2d"] || !seen["crouch_directional_2d"] {
		return nil, fmt.Errorf("missing controller modes/clips")
	}
	return a, nil
}

type groundVector [2]float32

func groundLength(v groundVector) float64 {
	return float64(float32(math.Sqrt(float64(v[0]*v[0] + v[1]*v[1]))))
}
func groundUnit(v, fallback groundVector) groundVector {
	n := float32(groundLength(v))
	if n <= 1e-6 {
		return fallback
	}
	return groundVector{v[0] / n, v[1] / n}
}
func groundScale(v groundVector, s float64) groundVector {
	return groundVector{v[0] * float32(s), v[1] * float32(s)}
}
func groundDot(a, b groundVector) float64 { return float64(a[0]*b[0] + a[1]*b[1]) }
func spring(x, v, goal, half, dt float64) (float64, float64) {
	y := 2 * math.Ln2 / (half + 1e-5)
	j0 := x - goal
	j1 := v + j0*y
	e := math.Exp(-y * dt)
	x = e*(j0+j1*dt) + goal
	v = e * (v - j1*y*dt)
	if math.Abs(x-goal) <= 1e-4 && math.Abs(v) <= 1e-4 {
		return goal, 0
	}
	return x, v
}
func wrapAngle(x float64) float64 { return x - 2*math.Pi*math.Floor((x+math.Pi)/(2*math.Pi)) }
func angleSpring(x, v float64, dir groundVector, half, dt float64) (float64, float64) {
	goal := x + wrapAngle(math.Atan2(float64(dir[1]), float64(dir[0]))-x)
	x, v = spring(x, v, goal, half, dt)
	return wrapAngle(x), v
}
func angleVector(x float64) groundVector {
	return groundVector{float32(math.Cos(x)), float32(math.Sin(x))}
}

type locomotionInput struct {
	Mode         string
	Move, Facing groundVector
	Fast         bool
}
type movementControl struct {
	inputSpeed, speed, speedRate, angle, angleRate, facingAngle, facingRate float64
	inputDirection, desiredFacing, lastFacing, facing, direction, velocity  groundVector
}

func newMovementControl() movementControl {
	v := groundVector{1, 0}
	return movementControl{inputDirection: v, desiredFacing: v, lastFacing: v, facing: v, direction: v}
}
func (m *movementControl) reset() { m.speed = 0; m.speedRate = 0; m.velocity = groundVector{} }
func (m *movementControl) realign(v groundVector, mode *locomotionMode) {
	m.speed = groundLength(v)
	m.speedRate = 0
	m.direction = groundUnit(v, m.facing)
	m.angle = math.Atan2(float64(m.direction[1]), float64(m.direction[0]))
	m.angleRate = 0
	m.velocity = groundScale(m.direction, m.speed)
	if !mode.Directional {
		m.facing = m.direction
		m.facingAngle = m.angle
		m.facingRate = 0
	}
}
func (m *movementControl) update(c locomotionInput, mode, previous *locomotionMode, dt float64) {
	facing := groundUnit(c.Facing, m.desiredFacing)
	amount := groundLength(c.Move)
	if amount <= 1e-6 {
		if mode.Directional || m.speed <= 0 {
			m.desiredFacing = facing
		}
		m.inputSpeed = 0
	} else {
		m.inputDirection = groundUnit(c.Move, m.inputDirection)
		m.desiredFacing = m.inputDirection
		if mode.Directional {
			m.desiredFacing = facing
		}
		m.lastFacing = m.desiredFacing
		base := mode.DefaultSpeed
		if c.Fast {
			base = mode.MaxSpeed
		}
		m.inputSpeed = math.Max(base*math.Min(amount, 1), mode.MinimumSpeed)
	}
	if previous != nil {
		if mode.Directional {
			m.facing = m.desiredFacing
			m.facingAngle = math.Atan2(float64(m.facing[1]), float64(m.facing[0]))
			m.facingRate = 0
		}
		if mode.MaxSpeed <= previous.MaxSpeed {
			s := 1 - .25*math.Min(math.Max(m.speed/mode.MaxSpeed, 0), 1)
			m.speed = math.Min(m.speed*s, mode.MaxSpeed)
			m.speedRate *= s
		}
	}
	m.angle, m.angleRate = angleSpring(m.angle, m.angleRate, m.inputDirection, .2, dt)
	m.direction = angleVector(m.angle)
	if !mode.Directional {
		m.facing = m.direction
		m.facingAngle = m.angle
		m.facingRate = m.angleRate
	} else {
		m.facingAngle, m.facingRate = angleSpring(m.facingAngle, m.facingRate, m.desiredFacing, .15, dt)
		m.facing = angleVector(m.facingAngle)
	}
	goal := m.inputSpeed
	if groundDot(m.direction, m.inputDirection) < .92 {
		goal *= 1 - .25*math.Min(math.Max(m.speed/mode.MaxSpeed, 0), 1)
	}
	half := .4
	if goal > m.speed {
		half = .6
	}
	m.speed, m.speedRate = spring(m.speed, m.speedRate, goal, half, dt)
	if m.speed > mode.MaxSpeed {
		m.speed = mode.MaxSpeed
		m.speedRate = math.Min(m.speedRate, 0)
	}
	m.velocity = groundScale(m.direction, m.speed)
}

// Local transforms use XYZW quaternions. Only the root is in runtime Z-up space.
func controllerQuatMul(a, b []float32) [4]float32 {
	return [4]float32{a[3]*b[0] + b[3]*a[0] + a[1]*b[2] - a[2]*b[1], a[3]*b[1] + b[3]*a[1] + a[2]*b[0] - a[0]*b[2], a[3]*b[2] + b[3]*a[2] + a[0]*b[1] - a[1]*b[0], a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]}
}
func controllerRotate(q, p []float32) [3]float32 {
	// Warp quat_rotate: v*(2*w*w-1) + cross(q.xyz,v)*2*w + q.xyz*dot(q.xyz,v)*2.
	d := 2 * (q[0]*p[0] + q[1]*p[1] + q[2]*p[2])
	s := 2*q[3]*q[3] - 1
	return [3]float32{p[0]*s + q[0]*d + (q[1]*p[2]-q[2]*p[1])*q[3]*2, p[1]*s + q[1]*d + (q[2]*p[0]-q[0]*p[2])*q[3]*2, p[2]*s + q[2]*d + (q[0]*p[1]-q[1]*p[0])*q[3]*2}
}
func controllerTransform(a, b []float32) [7]float32 {
	p := controllerRotate(a[3:], b[:3])
	q := controllerQuatMul(a[3:], b[3:])
	return [7]float32{p[0] + a[0], p[1] + a[1], p[2] + a[2], q[0], q[1], q[2], q[3]}
}
func controllerInverse(t [7]float32) [7]float32 {
	q := []float32{-t[3], -t[4], -t[5], t[6]}
	p := controllerRotate(q, []float32{-t[0], -t[1], -t[2]})
	return [7]float32{p[0], p[1], p[2], q[0], q[1], q[2], q[3]}
}
func controllerYaw(angle float64) [7]float32 {
	h := float32(angle) * .5
	return [7]float32{0, 0, 0, 0, 0, float32(math.Sin(float64(h))), float32(math.Cos(float64(h)))}
}
func controllerBlend(a, b []float32, t float32) []float32 {
	out := make([]float32, len(a))
	for j := 0; j < len(a); j += 7 {
		for k := 0; k < 3; k++ {
			out[j+k] = a[j+k]*(1-t) + b[j+k]*t
		}
		q := controllerSlerp(a[j+3:j+7], b[j+3:j+7], t)
		copy(out[j+3:j+7], q[:])
	}
	return out
}
func controllerSlerp(a, b []float32, t float32) [4]float32 {
	// Warp uses relative quaternion axis-angle interpolation (not acos(dot)).
	inv := []float32{-a[0], -a[1], -a[2], a[3]}
	q := controllerQuatMul(inv, b)
	if q[3] < 0 {
		for i := range q {
			q[i] = -q[i]
		}
	}
	length := float32(math.Sqrt(float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2])))
	if length < 1e-8 {
		return [4]float32{a[0], a[1], a[2], a[3]}
	}
	angle := float32(2 * math.Atan2(float64(length), float64(q[3])))
	h := angle * t * .5
	s := float32(math.Sin(float64(h))) / length
	step := []float32{q[0] * s, q[1] * s, q[2] * s, float32(math.Cos(float64(h)))}
	return controllerQuatMul(a, step)
}
func controllerSample(data []float32, width int, t, rate float64) []float32 {
	n := len(data) / width
	x := float32(t) * float32(rate)
	lo := max(0, min(int(math.Floor(float64(x))), n-1))
	hi := min(lo+1, n-1)
	b := float32(0)
	if lo < n-1 {
		b = max(0, min(x-float32(lo), 1))
	}
	return controllerBlend(data[lo*width:(lo+1)*width], data[hi*width:(hi+1)*width], b)
}
func controllerGround(data []float32, width int, t, rate float64, offset *[7]float32) groundVector {
	n := len(data) / width
	x := t * rate
	lo := max(0, min(int(math.Floor(x)), n-1))
	hi := min(lo+1, n-1)
	blend := 0.
	if lo < n-1 {
		blend = float64(float32(math.Max(0, math.Min(x-float64(lo), 1))))
	}
	var p [3]float32
	for k := range p {
		p[k] = data[lo*width+k]*float32(1-blend) + data[hi*width+k]*float32(blend)
	}
	if offset != nil {
		v := controllerRotate(offset[3:], p[:])
		for k := range p {
			p[k] = v[k] + offset[k]
		}
	}
	return groundVector{p[0], p[1]}
}

type locomotionGenerate func(context, target []float32, allowed []int) ([]float32, error)
type locomotionController struct {
	assets                                     *locomotionAssets
	mode                                       *locomotionMode
	movement                                   movementControl
	generate                                   locomotionGenerate
	pose, output, target                       []float32
	history                                    [][]float32
	historyDT                                  [32]float32
	ground, actualVelocity, previousGeneration groundVector
	idle, clipPlaying                          bool
	clip                                       *locomotionClip
	clipOffset                                 [7]float32
	time, clipTime, duration                   float64
	early                                      int
	generations                                int
}

func newLocomotionController(a *locomotionAssets, g locomotionGenerate) *locomotionController {
	c := &locomotionController{assets: a, mode: a.Modes[0], movement: newMovementControl(), generate: g, idle: true, history: make([][]float32, 32)}
	for _, m := range a.Modes {
		if m.Name == "linear_1d" {
			c.mode = m
		}
	}
	c.clip = a.Clips[0]
	c.alignClip(c.ground, c.movement.lastFacing)
	c.startClip()
	for i := range c.history {
		c.history[i] = append([]float32(nil), c.pose...)
	}
	return c
}
func (c *locomotionController) alignClip(ground, facing groundVector) {
	yaw := controllerYaw(math.Atan2(float64(facing[1]), float64(facing[0])) + math.Pi/2)
	q := controllerQuatMul(yaw[3:], c.clip.SourceGround[3:])
	target := [7]float32{ground[0], ground[1], 0, q[0], q[1], q[2], q[3]}
	inverse := controllerInverse(c.clip.SourceGround)
	align := controllerTransform(target[:], inverse[:])
	c.clipOffset = controllerTransform(align[:], c.assets.ModelToRuntime[:])
}
func (c *locomotionController) sampleClip(t float64) []float32 {
	p := controllerSample(c.clip.poses, c.assets.Joints*7, t, c.clip.Rate)
	root := controllerTransform(c.clipOffset[:], p[:7])
	copy(p[:7], root[:])
	return p
}
func (c *locomotionController) startClip() {
	c.clipPlaying = true
	c.clipTime = 0
	c.pose = c.sampleClip(0)
	c.ground = controllerGround(c.clip.poses, c.assets.Joints*7, 0, c.clip.Rate, &c.clipOffset)
}
func (c *locomotionController) context() []float32 {
	var times [32]float32
	for i := 30; i >= 0; i-- {
		times[i] = times[i+1] - c.historyDT[i]
	}
	out := []float32{}
	for i := 0; i < 4; i++ {
		t := -.1 + float64(i)/30
		upper := 0
		for upper < 31 && float64(times[upper]) <= t {
			upper++
		}
		lower := max(upper-1, 0)
		delta := float64(times[upper] - times[lower])
		b := float32(0)
		if delta > 0 {
			b = float32(math.Max(0, math.Min((t-float64(times[lower]))/delta, 1)))
		} else {
			lower = upper
		}
		out = append(out, controllerBlend(c.history[lower], c.history[upper], b)...)
	}
	return out
}
func (c *locomotionController) generateTarget(target []float32, allowed []int, early int) error {
	output, e := c.generate(c.context(), target, allowed)
	if e != nil {
		return e
	}
	if len(output)%(c.assets.Joints*7) != 0 || len(output) < 4*c.assets.Joints*7 {
		return fmt.Errorf("invalid controller generation")
	}
	c.output = output
	c.target = target
	c.time = 0
	c.early = early
	c.duration = float64(len(output)/(c.assets.Joints*7)-1-early) / 30
	c.clipPlaying = false
	c.generations++
	c.pose = controllerSample(output, c.assets.Joints*7, 0, 30)
	c.ground = controllerGround(output, c.assets.Joints*7, 0, 30, nil)
	return nil
}
func (c *locomotionController) enterIdle() error {
	c.idle = true
	c.movement.reset()
	c.clip = c.assets.Clips[0]
	if c.mode.Name == "crouch_directional_2d" {
		c.clip = c.assets.Clips[1]
	}
	g := groundVector{c.ground[0] + c.actualVelocity[0], c.ground[1] + c.actualVelocity[1]}
	c.alignClip(g, c.movement.lastFacing)
	var target []float32
	for i := 0; i < 4; i++ {
		target = append(target, c.sampleClip(float64(i)/30)...)
	}
	return c.generateTarget(target, []int{8, 9, 10, 11, 12, 13, 14}, 0)
}
func (c *locomotionController) generateMovement() error {
	target := c.blendTarget()
	if e := c.generateTarget(target, c.mode.Allowed, c.mode.Early); e != nil {
		return e
	}
	c.previousGeneration = c.movement.velocity
	return nil
}
func (c *locomotionController) enterMovement() error {
	c.idle = false
	c.movement.realign(c.actualVelocity, c.mode)
	return c.generateMovement()
}
func (c *locomotionController) needsGeneration() bool {
	m := &c.movement
	p := groundLength(c.previousGeneration)
	delta := m.speed - p
	remaining := p
	if delta > 0 {
		remaining = c.mode.MaxSpeed - p
	}
	if (m.speed > 0 && p <= 1e-6) || math.Abs(delta) > math.Max(remaining*.3, c.mode.MaxSpeed*.1) {
		return true
	}
	return p > 1e-6 && groundDot(m.direction, groundScale(c.previousGeneration, 1/p)) < .92
}
func (c *locomotionController) Step(input locomotionInput, dt float64) error {
	previous := c.mode
	for _, m := range c.assets.Modes {
		if m.Name == input.Mode {
			c.mode = m
		}
	}
	changed := previous != c.mode
	var old *locomotionMode
	if changed {
		old = previous
	}
	c.movement.update(input, c.mode, old, dt)
	stanceChanged := (previous.Name == "crouch_directional_2d") != (c.mode.Name == "crouch_directional_2d")
	if stanceChanged {
		var e error
		if c.movement.inputSpeed > 0 {
			e = c.enterMovement()
		} else {
			e = c.enterIdle()
		}
		if e != nil {
			return e
		}
		changed = false
	}
	if c.idle {
		if c.movement.inputSpeed > 0 {
			if e := c.enterMovement(); e != nil {
				return e
			}
		} else if !c.clipPlaying && c.duration-c.time <= dt {
			c.startClip()
		}
	} else {
		if c.movement.inputSpeed <= 0 {
			if e := c.enterIdle(); e != nil {
				return e
			}
		} else if changed || c.duration-c.time <= dt || c.needsGeneration() {
			if e := c.generateMovement(); e != nil {
				return e
			}
		}
	}
	previousGround := c.ground
	if c.clipPlaying {
		c.clipTime = math.Mod(c.clipTime+dt, c.clip.Duration)
		c.pose = c.sampleClip(c.clipTime)
		c.ground = controllerGround(c.clip.poses, c.assets.Joints*7, c.clipTime, c.clip.Rate, &c.clipOffset)
	} else {
		c.time += dt
		c.pose = controllerSample(c.output, c.assets.Joints*7, c.time, 30)
		c.ground = controllerGround(c.output, c.assets.Joints*7, c.time, 30, nil)
	}
	c.actualVelocity = groundVector{(c.ground[0] - previousGround[0]) / float32(dt), (c.ground[1] - previousGround[1]) / float32(dt)}
	copy(c.history, c.history[1:])
	c.history[31] = append([]float32(nil), c.pose...)
	c.historyDT[31] = float32(dt)
	copy(c.historyDT[:], c.historyDT[1:])
	c.historyDT[31] = 0
	return nil
}
func (c *locomotionController) blendTarget() []float32 {
	m := c.mode
	movement := &c.movement
	width := c.assets.Joints * 7
	size := 4 * width
	speed := groundLength(movement.velocity)
	var out []float32
	if !m.Directional {
		v := float32(speed / m.MaxSpeed)
		a, b := 0, 0
		t := float32(0)
		for i := 1; i < len(m.Values); i++ {
			a, b = i-1, i
			if v <= m.Values[i][0] {
				t = max(0, min((v-m.Values[a][0])/(m.Values[b][0]-m.Values[a][0]), 1))
				break
			}
			t = 1
		}
		out = controllerBlend(m.poses[a*size:(a+1)*size], m.poses[b*size:(b+1)*size], t)
	} else {
		v := groundScale(movement.velocity, 1/m.MaxSpeed)
		x := max(float32(-1), min(float32(1), float32(groundDot(v, movement.facing))))
		y := max(float32(-1), min(float32(1), float32(groundDot(v, groundVector{-movement.facing[1], movement.facing[0]}))))
		weights := make([]float32, 5)
		ix, iy := 1, 3
		if x < 0 {
			ix = 2
			x = -x
		}
		if y < 0 {
			iy = 4
			y = -y
		}
		sum := max(float32(1), x+y)
		weights[ix] = x / sum
		weights[iy] = y / sum
		weights[0] = max(0, 1-weights[ix]-weights[iy])
		out = make([]float32, size)
		ws := float32(0)
		for i, w := range weights {
			if w <= 0 {
				continue
			}
			ws += w
			for k := range out {
				out[k] += m.poses[i*size+k] * w
			}
		}
		missing := max(0, 1-ws)
		for i := 0; i < size; i += 7 {
			for k := 0; k < 7; k++ {
				out[i+k] += c.assets.Reference[(i%width)+k] * missing
			}
			q := out[i+3 : i+7]
			n := float32(math.Sqrt(float64(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])))
			for k := range q {
				q[k] /= n
			}
		}
	}
	dx, dy := out[3*width]-out[0], out[3*width+1]-out[1]
	source := float32(math.Sqrt(float64(dx*dx+dy*dy))) * 10
	if source > 1e-6 {
		scale := float32(speed*float64(m.VelocityScale)) / source
		for f := 1; f < 4; f++ {
			for k := 0; k < 2; k++ {
				at := f*width + k
				out[at] = out[k] + (out[at]-out[k])*scale
			}
		}
	}
	placement := controllerYaw(math.Atan2(float64(movement.facing[1]), float64(movement.facing[0])))
	placement[0] = c.ground[0] + movement.velocity[0]*m.VelocityScale*m.PositionScale
	placement[1] = c.ground[1] + movement.velocity[1]*m.VelocityScale*m.PositionScale
	for f := 0; f < 4; f++ {
		at := f * width
		root := controllerTransform(placement[:], out[at:at+7])
		copy(out[at:at+7], root[:])
	}
	return out
}
