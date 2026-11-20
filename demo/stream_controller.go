package main

import (
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
)

// streamStepper is a stateful policy serviced by the existing stream worker.
// Transport, ownership, sequencing and the 50 Hz clock remain in streamHub.
// A step can generate before sampling; the hub waits without advancing its clock.
type streamStepper interface {
	step(planRequest) (*mb.Motion, error)
}
type streamStepperFactory interface{ newStepper() streamStepper }
type somaControllerStepper struct {
	server     *somaServer
	controller *locomotionController
}

func controllerConvertRoot(local []float32, joints int, offset [7]float32) []float32 {
	out := append([]float32(nil), local...)
	for at := 0; at < len(out); at += joints * 7 {
		root := controllerTransform(offset[:], out[at:at+7])
		copy(out[at:at+7], root[:])
	}
	return out
}
func (s *somaServer) controllerGenerator(context, target []float32, allowed []int) ([]float32, error) {
	return s.controllerGeneratorPrecision(context, target, allowed, 0)
}
func (s *somaServer) controllerGeneratorPrecision(context, target []float32, allowed []int, precision uint32) ([]float32, error) {
	a := s.controllerAssets
	inverse := controllerInverse(a.ModelToRuntime)
	context = controllerConvertRoot(context, a.Joints, inverse)
	target = controllerConvertRoot(target, a.Joints, inverse)
	var mask uint32
	for _, n := range allowed {
		mask |= 1 << uint(n-6)
	}
	s.mu.Lock()
	motion, e := s.model.GenerateAlignedPrecision(context, target, mask, math.Pi/2, precision)
	s.mu.Unlock()
	if e != nil {
		return nil, e
	}
	return controllerConvertRoot(motion.Local, a.Joints, a.ModelToRuntime), nil
}
func (p somaStreamPlanner) newStepper() streamStepper {
	if p.server.controllerAssets == nil {
		return nil
	}
	return &somaControllerStepper{server: p.server, controller: newLocomotionController(p.server.controllerAssets, p.server.controllerGenerator)}
}
func (p *somaControllerStepper) step(command planRequest) (*mb.Motion, error) {
	c := p.controller
	before := c.generations
	// Browser movement is X/Z in model space; runtime ground is X/-Z.
	input := locomotionInput{Mode: command.Style, Move: groundVector{command.Move[0], -command.Move[1]}, Facing: groundVector{command.Facing[0], -command.Facing[1]}, Fast: command.Fast}
	if e := c.Step(input, .02); e != nil {
		return nil, e
	}
	inverse := controllerInverse(c.assets.ModelToRuntime)
	motion := somaLocalMotion(controllerConvertRoot(c.pose, c.assets.Joints, inverse), c.assets.Joints)
	if before != c.generations {
		target := somaLocalMotion(controllerConvertRoot(c.target, c.assets.Joints, inverse), c.assets.Joints)
		motion.Targets = &mb.Keyframes{Frames: 4, Joints: target.Joints, Roots: target.Roots, Rotations: target.Rotations, LocalPositions: target.LocalPositions}
	}
	return motion, nil
}
