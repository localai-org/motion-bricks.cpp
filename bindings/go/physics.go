package motionbricks

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/ebitengine/purego"
	"math"
	"unsafe"
)

// Physics owns an independent SONIC model and one physical session. Callers
// serialize calls. Closing a session before its SONIC model is mandatory.
type Physics struct {
	library             *Library
	sonic, session      uintptr
	started             bool
	scene, config       []byte
	create              func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	free                func(uintptr)
	reset               func(uintptr, unsafe.Pointer, uint64) uint32
	sonicFree           func(uintptr)
	start               func(uintptr, unsafe.Pointer, uint64, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	step                func(uintptr, unsafe.Pointer, uint64, unsafe.Pointer, uint64, uint32, float64, unsafe.Pointer, uint64, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	skeleton            func(uintptr, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	status              func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	collisionCount      func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	collisionShape      func(uintptr, uint32, unsafe.Pointer, unsafe.Pointer, uint64, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	collisionTriangles  func(uintptr, uint32, unsafe.Pointer, uint64, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	collisionTransforms func(uintptr, unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	collisionShapes     []CollisionShape
	collisionShapesID   string
	engineVersion       string
}
type CollisionShape struct {
	Type     uint32    `json:"type"`
	Name     string    `json:"name"`
	Size     []float32 `json:"size"`
	Vertices []float32 `json:"vertices,omitempty"`
	Indices  []uint32  `json:"indices,omitempty"`
}

// CollisionShapes returns immutable compiled collision geometry for this model.
func (p *Physics) CollisionShapes() []CollisionShape { return p.collisionShapes }
func (p *Physics) EngineVersion() string             { return p.engineVersion }
func (p *Physics) CollisionShapesID() string         { return p.collisionShapesID }

type PhysicalFrame struct {
	Time                float64   `json:"time"`
	Actual              []float32 `json:"actual"`
	Reference           []float32 `json:"reference"`
	Fallen              bool      `json:"fallen"`
	Contacts            uint32    `json:"contacts"`
	CollisionTransforms []float32 `json:"collision_transforms,omitempty"`
}

func OpenPhysics(library, model, scene, config string, device Device) (*Physics, error) {
	l, err := Open(library)
	if err != nil {
		return nil, err
	}
	p := &Physics{library: l}
	ok := false
	defer func() {
		if !ok {
			p.Close()
		}
	}()
	if p.scene, err = cString(scene); err != nil {
		return nil, err
	}
	if p.config, err = cString(config); err != nil {
		return nil, err
	}
	var load func(unsafe.Pointer, uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	var engineVersion func(unsafe.Pointer, uint64, unsafe.Pointer, uint64) uint32
	for _, entry := range []struct {
		name   string
		target any
	}{
		{"mb_sonic_load", &load}, {"mb_sonic_free", &p.sonicFree}, {"mb_physics_create", &p.create}, {"mb_physics_free", &p.free}, {"mb_physics_reset", &p.reset},
		{"mb_physics_engine_version", &engineVersion},
		{"mb_physics_start", &p.start}, {"mb_physics_step", &p.step}, {"mb_physics_skeleton", &p.skeleton}, {"mb_physics_status", &p.status},
		{"mb_physics_collision_count", &p.collisionCount}, {"mb_physics_collision_shape", &p.collisionShape},
		{"mb_physics_collision_triangles", &p.collisionTriangles}, {"mb_physics_collision_transforms", &p.collisionTransforms},
	} {
		address, e := lookupSymbol(l.handle, entry.name)
		if e != nil {
			return nil, e
		}
		purego.RegisterFunc(entry.target, address)
	}
	buffer := make([]byte, errorBufferSize)
	version := make([]byte, 32)
	if err = l.check("MuJoCo version", engineVersion(unsafe.Pointer(&version[0]), uint64(len(version)), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	for _, b := range version {
		if b == 0 {
			break
		}
		p.engineVersion += string(b)
	}
	var options uintptr
	if err = l.check("options", l.optionsCreate(unsafe.Pointer(&options), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	defer l.optionsFree(options)
	if err = l.check("device", l.optionsSetDevice(options, uint32(device), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	path, err := cString(model)
	if err != nil {
		return nil, err
	}
	if err = l.check("SONIC load", load(unsafe.Pointer(&path[0]), options, unsafe.Pointer(&p.sonic), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	// Compile the robot once during startup, not in the real-time tick loop.
	if err = l.check("physics prepare", p.create(p.sonic, unsafe.Pointer(&p.scene[0]), unsafe.Pointer(&p.config[0]), unsafe.Pointer(&p.session), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	var count uint32
	if err = l.check("collision count", p.collisionCount(p.session, unsafe.Pointer(&count), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if count > 1024 {
		return nil, errors.New("too many collision shapes")
	}
	for i := uint32(0); i < count; i++ {
		shape := CollisionShape{Size: make([]float32, 3)}
		name := make([]byte, 256)
		if err = l.check("collision shape", p.collisionShape(p.session, i, unsafe.Pointer(&shape.Type), unsafe.Pointer(&shape.Size[0]), 3, unsafe.Pointer(&name[0]), uint64(len(name)), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
			return nil, err
		}
		for n, v := range name {
			if v == 0 {
				shape.Name = string(name[:n])
				break
			}
		}
		if shape.Type < 2 || shape.Type > 7 {
			return nil, fmt.Errorf("unsupported robot collision shape %d", shape.Type)
		}
		var floats uint64
		if err = l.check("collision mesh count", p.collisionTriangles(p.session, i, nil, 0, unsafe.Pointer(&floats), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
			return nil, err
		}
		if floats > 1000000 || floats%9 != 0 {
			return nil, errors.New("invalid collision triangle count")
		}
		if floats > 0 {
			shape.Vertices = make([]float32, floats)
			if err = l.check("collision mesh", p.collisionTriangles(p.session, i, unsafe.Pointer(&shape.Vertices[0]), floats, unsafe.Pointer(&floats), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
				return nil, err
			}
			// Lossless indexing: preserve every compiled hull triangle while
			// avoiding duplicated vertices on the wire and in the renderer.
			triangles := shape.Vertices
			shape.Vertices = nil
			vertices := make(map[[3]float32]uint32)
			for at := 0; at < len(triangles); at += 3 {
				key := [3]float32{triangles[at], triangles[at+1], triangles[at+2]}
				index, found := vertices[key]
				if !found {
					index = uint32(len(shape.Vertices) / 3)
					vertices[key] = index
					shape.Vertices = append(shape.Vertices, key[:]...)
				}
				shape.Indices = append(shape.Indices, index)
			}
		}
		p.collisionShapes = append(p.collisionShapes, shape)
	}
	definitions, err := json.Marshal(p.collisionShapes)
	if err != nil {
		return nil, err
	}
	p.collisionShapesID = fmt.Sprintf("%x", sha256.Sum256(definitions))
	ok = true
	return p, nil
}
func (p *Physics) Close() {
	if p == nil {
		return
	}
	if p.session != 0 {
		p.free(p.session)
		p.session = 0
		p.started = false
	}
	if p.sonic != 0 {
		p.sonicFree(p.sonic)
		p.sonic = 0
	}
	if p.library != nil {
		p.library.Close()
		p.library = nil
	}
}
func (p *Physics) Reset() {
	if p.session != 0 {
		b := make([]byte, errorBufferSize)
		_ = p.reset(p.session, errorPointer(b), uint64(len(b)))
		p.started = false
	}
}
func (p *Physics) Start(m *Motion, frame float64) ([]int32, error) {
	if err := validPhysicalMotion(m, frame); err != nil {
		return nil, err
	}
	if p.started {
		return nil, errors.New("physical session already started")
	}
	b := make([]byte, errorBufferSize)
	i := int(math.Min(float64(m.Frames-1), math.Floor(frame)))
	if err := p.library.check("physics start", p.start(p.session, unsafe.Pointer(&m.Roots[i*3]), 3, unsafe.Pointer(&m.Rotations[i*136]), 136, errorPointer(b), uint64(len(b))), b); err != nil {
		p.Reset()
		return nil, err
	}
	parents := make([]int32, 30)
	if err := p.library.check("physics skeleton", p.skeleton(p.session, unsafe.Pointer(&parents[0]), 30, errorPointer(b), uint64(len(b))), b); err != nil {
		p.Reset()
		return nil, err
	}
	p.started = true
	return parents, nil
}
func validPhysicalMotion(m *Motion, frame float64) error {
	if m == nil || m.Frames < 2 || m.Frames > 1800 || m.Joints != 34 || uint64(len(m.Roots)) != m.Frames*3 || uint64(len(m.Rotations)) != m.Frames*136 || math.IsNaN(frame) || math.IsInf(frame, 0) || frame < 0 || frame > 3600 {
		return errors.New("invalid physical reference shape/time")
	}
	return nil
}
func (p *Physics) Step(m *Motion, frame float64) (PhysicalFrame, error) {
	out := PhysicalFrame{Actual: make([]float32, 90), Reference: make([]float32, 90)}
	if !p.started {
		return out, fmt.Errorf("physical session not started")
	}
	if err := validPhysicalMotion(m, frame); err != nil {
		return out, err
	}
	b := make([]byte, errorBufferSize)
	if err := p.library.check("physics step", p.step(p.session, unsafe.Pointer(&m.Roots[0]), uint64(len(m.Roots)), unsafe.Pointer(&m.Rotations[0]), uint64(len(m.Rotations)), uint32(m.Frames), frame/30, unsafe.Pointer(&out.Actual[0]), 90, unsafe.Pointer(&out.Reference[0]), 90, errorPointer(b), uint64(len(b))), b); err != nil {
		return out, err
	}
	var fallen uint32
	err := p.library.check("physics status", p.status(p.session, unsafe.Pointer(&out.Time), unsafe.Pointer(&fallen), unsafe.Pointer(&out.Contacts), errorPointer(b), uint64(len(b))), b)
	if err != nil {
		return out, err
	}
	out.CollisionTransforms = make([]float32, len(p.collisionShapes)*7)
	if len(out.CollisionTransforms) > 0 {
		err = p.library.check("collision transforms", p.collisionTransforms(p.session, unsafe.Pointer(&out.CollisionTransforms[0]), uint64(len(out.CollisionTransforms)), errorPointer(b), uint64(len(b))), b)
	}
	out.Fallen = fallen != 0
	return out, err
}
