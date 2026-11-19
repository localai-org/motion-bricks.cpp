package main

// All timeline indices are absolute 30 Hz reference frames; physical time is
// a separate integer 50 Hz tick counter. Replacement never changes the past.
import (
	"errors"
	"fmt"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
)

type referenceTimeline struct {
	base   int
	motion *mb.Motion
}

func validateStreamMotion(m *mb.Motion) error { return validateStreamPoseSequence(m, 4) }
func validateStreamPoseSequence(m *mb.Motion, minFrames uint64) error {
	if m == nil || (m.Joints < 1 || m.Joints > 256) || m.Frames < minFrames || m.Frames > 18016 || len(m.Roots) != int(m.Frames)*3 || len(m.Rotations) != int(m.Frames*m.Joints)*4 {
		return errors.New("invalid reference shape")
	}
	if len(m.LocalPositions) > 0 {
		if len(m.LocalPositions) != int(m.Frames*m.Joints)*3 || !finite(m.LocalPositions...) {
			return errors.New("invalid local position shape or values")
		}
		for _, v := range m.LocalPositions {
			if math.Abs(float64(v)) > 10000 {
				return errors.New("extreme local position")
			}
		}
	}
	for i, v := range m.Roots {
		if !finite(v) || math.Abs(float64(v)) > 10000 {
			return fmt.Errorf("invalid/extreme reference root at %d", i)
		}
		if i >= 3 && math.Abs(float64(v-m.Roots[i-3])) > .7 {
			return fmt.Errorf("reference root discontinuity at frame %d", i/3)
		}
	}
	for i := 0; i < len(m.Rotations); i += 4 {
		n := 0.
		for _, v := range m.Rotations[i : i+4] {
			n += float64(v) * float64(v)
		}
		if math.IsNaN(n) || math.IsInf(n, 0) || math.Abs(n-1) > .02 {
			return fmt.Errorf("invalid reference quaternion at %d", i/4)
		}
	}
	if m.Targets != nil {
		if m.Targets.Frames != 4 || m.Targets.Joints != m.Joints {
			return errors.New("invalid target frame count")
		}
		return validateStreamMotion(&mb.Motion{Frames: m.Targets.Frames, Joints: m.Targets.Joints, Roots: m.Targets.Roots, Rotations: m.Targets.Rotations, LocalPositions: m.Targets.LocalPositions})
	}
	return nil
}

func sampleStreamMotion(m *mb.Motion, frame float64) ([]float32, []float32) {
	x := math.Max(0, math.Min(float64(m.Frames-1), frame))
	i := int(x)
	j := min(i+1, int(m.Frames-1))
	u := float32(x - float64(i))
	root := make([]float32, 3)
	width := int(m.Joints) * 4
	rot := make([]float32, width)
	for k := range root {
		root[k] = m.Roots[i*3+k] + u*(m.Roots[j*3+k]-m.Roots[i*3+k])
	}
	for k := 0; k < width; k += 4 {
		q := slerpQuaternion(m.Rotations[i*width+k:i*width+k+4], m.Rotations[j*width+k:j*width+k+4], u)
		copy(rot[k:], q[:])
	}
	return root, rot
}

func (r *referenceTimeline) sample(frame float64) ([]float32, []float32) {
	return sampleStreamMotion(r.motion, frame-float64(r.base))
}
func (r *referenceTimeline) end() int { return r.base + int(r.motion.Frames) - 1 }
func (r *referenceTimeline) context(at int) *mb.Motion {
	m := &mb.Motion{Frames: 4, Joints: r.motion.Joints}
	for i := 0; i < 4; i++ {
		root, rot := r.sample(float64(at + i))
		m.Roots = append(m.Roots, root...)
		m.Rotations = append(m.Rotations, rot...)
		m.LocalPositions = append(m.LocalPositions, sampleLocalPositions(r.motion, float64(at+i-r.base))...)
	}
	return m
}

func (r *referenceTimeline) replace(at int, m *mb.Motion, now float64, blend bool) error {
	if err := validateStreamMotion(m); err != nil {
		return err
	}
	if r.motion == nil {
		r.base = at
		r.motion = m
		return nil
	}
	if m.Joints != r.motion.Joints || (len(m.LocalPositions) > 0) != (len(r.motion.LocalPositions) > 0) {
		return errors.New("replacement pose layout differs")
	}
	if float64(at) < now+1 {
		return errors.New("late plan: seam already committed to playback")
	}
	// Deployment crossfade, separate from inference parity. First new sample
	// is exactly the old reference at the seam; ease into the planned result.
	width := int(m.Joints) * 4
	if blend {
		for i := 0; i < min(8, int(m.Frames)); i++ {
			root, rot := r.sample(float64(at + i))
			u := float32(i) / 7
			u = u * u * (3 - 2*u)
			for k := 0; k < 3; k++ {
				m.Roots[i*3+k] = root[k] + u*(m.Roots[i*3+k]-root[k])
			}
			positions := sampleLocalPositions(r.motion, float64(at+i-r.base))
			for k, v := range positions {
				offset := i*len(positions) + k
				m.LocalPositions[offset] = v + u*(m.LocalPositions[offset]-v)
			}
			for k := 0; k < width; k += 4 {
				q := slerpQuaternion(rot[k:k+4], m.Rotations[i*width+k:i*width+k+4], u)
				copy(m.Rotations[i*width+k:], q[:])
			}
		}
	}
	if err := validateStreamMotion(m); err != nil {
		return err
	}
	keep := max(0, int(math.Floor(now))-4)
	out := &mb.Motion{Frames: uint64(at-keep) + m.Frames, Joints: r.motion.Joints, Targets: m.Targets}
	if at < keep || out.Frames > 18100 {
		return errors.New("reference buffer bound exceeded")
	}
	for i := keep; i < at; i++ {
		root, rot := r.sample(float64(i))
		out.Roots = append(out.Roots, root...)
		out.Rotations = append(out.Rotations, rot...)
		out.LocalPositions = append(out.LocalPositions, sampleLocalPositions(r.motion, float64(i-r.base))...)
	}
	out.Roots = append(out.Roots, m.Roots...)
	out.Rotations = append(out.Rotations, m.Rotations...)
	out.LocalPositions = append(out.LocalPositions, m.LocalPositions...)
	if err := validateStreamMotion(out); err != nil {
		return err
	}
	r.base = keep
	r.motion = out
	return nil
}

func (r *referenceTimeline) window(frame float64) (*mb.Motion, float64) {
	base := int(math.Floor(frame))
	m := &mb.Motion{Frames: 40, Joints: r.motion.Joints}
	for i := 0; i < 40; i++ {
		root, rot := r.sample(float64(base + i))
		m.Roots = append(m.Roots, root...)
		m.Rotations = append(m.Rotations, rot...)
		m.LocalPositions = append(m.LocalPositions, sampleLocalPositions(r.motion, float64(base+i-r.base))...)
	}
	return m, frame - float64(base)
}

func validatePhysicalFrame(p mb.PhysicalFrame, previous []float32) error {
	if len(p.CollisionTransforms)%7 != 0 || len(p.CollisionTransforms) > 1024*7 {
		return errors.New("collision transform shape")
	}
	for _, v := range p.CollisionTransforms {
		if !finite(v) || math.Abs(float64(v)) > 10000 {
			return errors.New("invalid collision transform")
		}
	}
	if len(p.Actual) != 90 || len(p.Reference) != 90 {
		return errors.New("physical output shape")
	}
	for i, v := range p.Actual {
		if !finite(v) || math.Abs(float64(v)) > 10000 {
			return fmt.Errorf("invalid/extreme physical output at joint %d", i/3)
		}
		if len(previous) == 90 && math.Abs(float64(v-previous[i])) > 1 {
			return fmt.Errorf("physical discontinuity at joint %d", i/3)
		}
	}
	for _, v := range p.Reference {
		if !finite(v) || math.Abs(float64(v)) > 10000 {
			return errors.New("invalid physical reference FK")
		}
	}
	return nil
}

func sampleLocalPositions(m *mb.Motion, frame float64) []float32 {
	if len(m.LocalPositions) == 0 {
		return nil
	}
	at := math.Max(0, math.Min(float64(m.Frames-1), frame))
	first := int(at)
	next := min(first+1, int(m.Frames-1))
	u := float32(at - float64(first))
	width := int(m.Joints) * 3
	out := make([]float32, width)
	for k := range out {
		a, b := m.LocalPositions[first*width+k], m.LocalPositions[next*width+k]
		out[k] = a + u*(b-a)
	}
	return out
}
