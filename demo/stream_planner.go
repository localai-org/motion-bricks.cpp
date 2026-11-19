package main

import (
	"errors"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

// streamPlanner adapts model inference only. Scheduling, ownership, buffering,
// supersession, blending and playback stay in streamHub/referenceTimeline.
type streamPlanner interface {
	styles() []styleInfo
	plan(*mb.Motion, planRequest) (*mb.Motion, error)
	contextOffset() int
	actions() bool
	clip(string) *kimodoClip
}
type g1StreamPlanner struct{ server *demoServer }

func (p g1StreamPlanner) styles() []styleInfo { return p.server.ordered }
func (p g1StreamPlanner) contextOffset() int  { return 0 }
func (p g1StreamPlanner) actions() bool       { return true }
func (p g1StreamPlanner) clip(id string) *kimodoClip {
	p.server.mu.Lock()
	defer p.server.mu.Unlock()
	return p.server.kimodo[id]
}
func (p g1StreamPlanner) plan(context *mb.Motion, command planRequest) (*mb.Motion, error) {
	style, err := p.server.style(command.Style)
	if err != nil {
		return nil, err
	}
	agent, err := p.server.model.NewAgent()
	if err != nil {
		return nil, err
	}
	defer agent.Close()
	if context == nil {
		err = agent.Reset(style)
	} else {
		err = agent.SetContext(context.Roots, context.Rotations, context.Frames)
	}
	if err != nil {
		return nil, err
	}
	return p.server.commandPlan(&session{agent: agent}, style, command)
}

type somaStreamPlanner struct{ server *somaServer }

func (p somaStreamPlanner) styles() []styleInfo {
	out := []styleInfo{}
	if p.server.controllerAssets != nil {
		for _, m := range p.server.controllerAssets.Modes {
			out = append(out, styleInfo{Name: m.Name, Speed: float32(m.DefaultSpeed)})
		}
		return out
	}
	for _, s := range p.server.assets.Styles {
		out = append(out, styleInfo{Name: s.Name, Speed: s.Speed})
	}
	return out
}
func (p somaStreamPlanner) contextOffset() int      { return -3 }
func (p somaStreamPlanner) actions() bool           { return false }
func (p somaStreamPlanner) clip(string) *kimodoClip { return nil }
func (p somaStreamPlanner) plan(context *mb.Motion, command planRequest) (*mb.Motion, error) {
	s := p.server
	style := s.style(command.Style)
	if style == nil {
		return nil, errors.New("unknown SOMA style")
	}
	n := len(s.assets.Skeleton.Parents)
	local := append([]float32(nil), s.style("idle").Local...)
	if context != nil {
		if err := validateStreamMotion(context); err != nil {
			return nil, err
		}
		if context.Frames != 4 || int(context.Joints) != n || len(context.LocalPositions) != 4*n*3 {
			return nil, errors.New("SOMA context requires four complete local poses")
		}
		for f := 0; f < 4; f++ {
			for j := 0; j < n; j++ {
				at := (f*n + j) * 7
				copy(local[at:at+3], context.LocalPositions[(f*n+j)*3:])
				copy(local[at+3:at+7], context.Rotations[(f*n+j)*4:])
				if j == 0 {
					copy(local[at:at+3], context.Roots[f*3:])
				}
			}
		}
	} else {
		x, z := local[0], local[2]
		for f := 0; f < 4; f++ {
			local[f*n*7] -= x
			local[f*n*7+2] -= z
		}
	}
	target := placeSOMATarget(style, local, n, command.Move, command.Facing)
	s.mu.Lock()
	motion, err := s.model.GenerateAlignedPrecision(local, target, 2047, 0, 0)
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	out := somaLocalMotion(motion.Local, n)
	targetMotion := somaLocalMotion(target, n)
	out.Targets = &mb.Keyframes{Frames: 4, Joints: uint64(n), Roots: targetMotion.Roots, Rotations: targetMotion.Rotations, LocalPositions: targetMotion.LocalPositions}
	return out, nil
}
func somaLocalMotion(local []float32, n int) *mb.Motion {
	out := &mb.Motion{Frames: uint64(len(local) / (n * 7)), Joints: uint64(n)}
	for f := 0; f < int(out.Frames); f++ {
		for j := 0; j < n; j++ {
			at := (f*n + j) * 7
			if j == 0 {
				out.Roots = append(out.Roots, local[at:at+3]...)
			}
			out.Rotations = append(out.Rotations, local[at+3:at+7]...)
			out.LocalPositions = append(out.LocalPositions, local[at:at+3]...)
		}
	}
	return out
}
