package motionbricks

import (
	"errors"
	"github.com/ebitengine/purego"
	"unsafe"
)

type ModelKind uint32

const (
	ModelG1   ModelKind = 0
	ModelSOMA ModelKind = 1
)

type RetargetSkeleton struct {
	Names     []string  `json:"names"`
	Parents   []int32   `json:"parents"`
	Reference []float32 `json:"reference"`
	Mapping   []uint32  `json:"mapping"`
	RightLeg  uint32    `json:"right_leg"`
	LeftLeg   uint32    `json:"left_leg"`
}
type TransformMotion struct {
	Frames, Joints, Tokens uint32
	Local, World           []float32
}

// InferenceRequest owns an opaque request. Its model must outlive it.
// Serialize calls sharing a request or model.
type InferenceRequest struct {
	model        *Model
	handle       uintptr
	joints       uint32
	free         func(uintptr)
	setSkeleton  func(uintptr, uint32, unsafe.Pointer, unsafe.Pointer, uint64, unsafe.Pointer, uint64, uint32, uint32, unsafe.Pointer, uint64) uint32
	setBoundary  func(uintptr, uint32, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	setPrecision func(uintptr, uint32, unsafe.Pointer, uint64) uint32
	setHeading   func(uintptr, float32, unsafe.Pointer, uint64) uint32
	setMask      func(uintptr, uint32, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	infer        func(uintptr, uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	tokens       func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	copyMotion   func(uintptr, uint32, unsafe.Pointer, uint64, unsafe.Pointer, unsafe.Pointer, uint64) uint32
}

func (m *Model) NewInferenceRequest() (*InferenceRequest, error) {
	if m == nil || m.handle == 0 {
		return nil, errors.New("model is closed")
	}
	r := &InferenceRequest{model: m}
	var create func(unsafe.Pointer, unsafe.Pointer, uint64) uint32
	for _, x := range []struct {
		name string
		fn   any
	}{
		{"mb_inference_request_create", &create}, {"mb_inference_request_free", &r.free},
		{"mb_inference_request_set_skeleton", &r.setSkeleton}, {"mb_inference_request_set_boundary_transforms", &r.setBoundary},
		{"mb_inference_request_set_precision", &r.setPrecision}, {"mb_inference_request_set_heading", &r.setHeading},
		{"mb_inference_request_set_mask", &r.setMask}, {"mb_model_infer", &r.infer},
		{"mb_motion_get_token_count", &r.tokens}, {"mb_motion_copy_field", &r.copyMotion},
	} {
		address, e := lookupSymbol(m.library.handle, x.name)
		if e != nil {
			return nil, e
		}
		purego.RegisterFunc(x.fn, address)
	}
	b := make([]byte, errorBufferSize)
	if e := m.library.check("create inference request", create(unsafe.Pointer(&r.handle), errorPointer(b), uint64(len(b))), b); e != nil {
		return nil, e
	}
	return r, nil
}
func (r *InferenceRequest) Close() {
	if r != nil && r.handle != 0 {
		r.free(r.handle)
		r.handle = 0
	}
}
func (r *InferenceRequest) SetSkeleton(s RetargetSkeleton) error {
	if r == nil || r.handle == 0 {
		return errors.New("request is closed")
	}
	n := len(s.Parents)
	if n < 30 || n > 256 || len(s.Reference) != n*7 || len(s.Mapping) != 30 {
		return errors.New("retarget skeleton requires 30..256 parents, joints*7 reference values and 30 mapping entries")
	}
	b := make([]byte, errorBufferSize)
	e := r.model.library.check("set skeleton", r.setSkeleton(r.handle, uint32(n), unsafe.Pointer(&s.Parents[0]), unsafe.Pointer(&s.Reference[0]), uint64(len(s.Reference)), unsafe.Pointer(&s.Mapping[0]), uint64(len(s.Mapping)), s.RightLeg, s.LeftLeg, errorPointer(b), uint64(len(b))), b)
	if e == nil {
		r.joints = uint32(n)
	}
	return e
}
func (r *InferenceRequest) GenerateAlignedPrecision(context, target []float32, allowed uint32, heading float32, precision uint32) (*TransformMotion, error) {
	if r == nil || r.handle == 0 || r.model.handle == 0 {
		return nil, errors.New("request or model is closed")
	}
	if len(context) == 0 || len(target) == 0 {
		return nil, errors.New("context and target must not be empty")
	}
	if allowed == 0 || allowed > 2047 {
		return nil, errors.New("allowed duration bits must be a nonempty subset of bits 0..10")
	}
	b := make([]byte, errorBufferSize)
	n := uint64(len(b))
	e := errorPointer(b)
	l := r.model.library
	var masks [11]uint32
	for i := range masks {
		masks[i] = (allowed >> i) & 1
	}
	calls := []func() uint32{
		func() uint32 {
			return r.setBoundary(r.handle, 0, unsafe.Pointer(&context[0]), uint64(len(context)), e, n)
		},
		func() uint32 {
			return r.setBoundary(r.handle, 1, unsafe.Pointer(&target[0]), uint64(len(target)), e, n)
		},
		func() uint32 { return r.setMask(r.handle, 3, unsafe.Pointer(&masks[0]), 11, e, n) },
		func() uint32 { return r.setPrecision(r.handle, precision, e, n) }, func() uint32 { return r.setHeading(r.handle, heading, e, n) },
	}
	for _, call := range calls {
		if err := l.check("configure inference", call(), b); err != nil {
			return nil, err
		}
	}
	var h uintptr
	if err := l.check("infer", r.infer(r.model.handle, r.handle, unsafe.Pointer(&h), e, n), b); err != nil {
		return nil, err
	}
	defer l.motionFree(h)
	var frames, joints uint64
	var tokens uint32
	for _, call := range []func() uint32{
		func() uint32 { return l.motionFrames(h, unsafe.Pointer(&frames), e, n) }, func() uint32 { return l.motionJoints(h, unsafe.Pointer(&joints), e, n) }, func() uint32 { return r.tokens(h, unsafe.Pointer(&tokens), e, n) },
	} {
		if err := l.check("motion shape", call(), b); err != nil {
			return nil, err
		}
	}
	if frames > 58 || joints != uint64(r.joints) {
		return nil, errors.New("invalid inferred motion shape")
	}
	out := &TransformMotion{Frames: uint32(frames), Joints: uint32(joints), Tokens: tokens}
	for field, dst := range []*[]float32{&out.Local, &out.World} {
		var count uint64
		if err := l.check("motion size", r.copyMotion(h, uint32(field), nil, 0, unsafe.Pointer(&count), e, n), b); err != nil {
			return nil, err
		}
		if count != frames*joints*7 {
			return nil, errors.New("inferred transform count does not match shape")
		}
		*dst = make([]float32, count)
		if count > 0 {
			if err := l.check("motion copy", r.copyMotion(h, uint32(field), unsafe.Pointer(&(*dst)[0]), count, unsafe.Pointer(&count), e, n), b); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}
