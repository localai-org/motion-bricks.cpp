//go:build linux || darwin || freebsd

// Package motionbricks provides a PureGo wrapper around the stable opaque C ABI.
package motionbricks

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/ebitengine/purego"
)

type Device uint32

const (
	DeviceAuto   Device = 0
	DeviceCPU    Device = 1
	DeviceVulkan Device = 2
)

const errorBufferSize = 1024

type Library struct {
	handle                   uintptr
	abiVersion               func() uint32
	statusString             func(uint32) uintptr
	optionsCreate            func(unsafe.Pointer, unsafe.Pointer, uint64) uint32
	optionsFree              func(uintptr)
	optionsSetDevice         func(uintptr, uint32, unsafe.Pointer, uint64) uint32
	modelLoad                func(unsafe.Pointer, uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	modelFree                func(uintptr)
	modelJointCount          func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	modelJointName           func(uintptr, uint32, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	modelJointParent         func(uintptr, uint32, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	modelNeutralPosition     func(uintptr, uint32, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	styleLoad                func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	styleFree                func(uintptr)
	styleName                func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	styleSpeed               func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	agentCreate              func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	agentFree                func(uintptr)
	agentReset               func(uintptr, uintptr, unsafe.Pointer, uint64) uint32
	agentSetContext          func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64, uint64, unsafe.Pointer, uint64) uint32
	agentPlan                func(uintptr, uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	agentAdvance             func(uintptr, uint32, unsafe.Pointer, uint64) uint32
	commandCreate            func(unsafe.Pointer, unsafe.Pointer, uint64) uint32
	commandFree              func(uintptr)
	commandSetStyle          func(uintptr, uintptr, unsafe.Pointer, uint64) uint32
	commandSetMovement       func(uintptr, float32, float32, float32, unsafe.Pointer, uint64) uint32
	commandSetFacing         func(uintptr, float32, float32, float32, unsafe.Pointer, uint64) uint32
	commandSetSpeed          func(uintptr, float32, unsafe.Pointer, uint64) uint32
	commandSetSamplingArgmax func(uintptr, uint32, unsafe.Pointer, uint64) uint32
	commandSetWorldTarget    func(uintptr, float32, float32, float32, float32, uint32, unsafe.Pointer, uint64) uint32
	commandSetSeed           func(uintptr, uint64, unsafe.Pointer, uint64) uint32
	motionFree               func(uintptr)
	motionFrames             func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionJoints             func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionRoots              func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionRotations          func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionTargetFrames       func(uintptr, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionTargetRoots        func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
	motionTargetRotations    func(uintptr, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, uint64) uint32
}

func Open(path string) (*Library, error) {
	handle, err := purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
	if err != nil {
		return nil, err
	}
	library := &Library{handle: handle}
	register := func(target any, name string) { purego.RegisterLibFunc(target, handle, name) }
	register(&library.abiVersion, "mb_abi_version")
	register(&library.statusString, "mb_status_string")
	register(&library.optionsCreate, "mb_runtime_options_create")
	register(&library.optionsFree, "mb_runtime_options_free")
	register(&library.optionsSetDevice, "mb_runtime_options_set_device")
	register(&library.modelLoad, "mb_model_load")
	register(&library.modelFree, "mb_model_free")
	register(&library.modelJointCount, "mb_model_get_joint_count")
	register(&library.modelJointName, "mb_model_get_joint_name")
	register(&library.modelJointParent, "mb_model_get_joint_parent")
	register(&library.modelNeutralPosition, "mb_model_get_neutral_joint_position")
	register(&library.styleLoad, "mb_style_load")
	register(&library.styleFree, "mb_style_free")
	register(&library.styleName, "mb_style_get_name")
	register(&library.styleSpeed, "mb_style_get_speed")
	register(&library.agentCreate, "mb_agent_create")
	register(&library.agentFree, "mb_agent_free")
	register(&library.agentReset, "mb_agent_reset")
	register(&library.agentSetContext, "mb_agent_set_context")
	register(&library.agentPlan, "mb_agent_plan")
	register(&library.agentAdvance, "mb_agent_advance")
	register(&library.commandCreate, "mb_command_create")
	register(&library.commandFree, "mb_command_free")
	register(&library.commandSetStyle, "mb_command_set_style")
	register(&library.commandSetMovement, "mb_command_set_movement_direction")
	register(&library.commandSetFacing, "mb_command_set_facing_direction")
	register(&library.commandSetSpeed, "mb_command_set_target_speed")
	register(&library.commandSetSamplingArgmax, "mb_command_set_sampling_argmax")
	register(&library.commandSetWorldTarget, "mb_command_set_world_target")
	register(&library.commandSetSeed, "mb_command_set_seed")
	register(&library.motionFree, "mb_motion_free")
	register(&library.motionFrames, "mb_motion_get_frame_count")
	register(&library.motionJoints, "mb_motion_get_joint_count")
	register(&library.motionRoots, "mb_motion_get_root_translations")
	register(&library.motionRotations, "mb_motion_get_local_rotations_xyzw")
	register(&library.motionTargetFrames, "mb_motion_get_target_frame_count")
	register(&library.motionTargetRoots, "mb_motion_get_target_root_translations")
	register(&library.motionTargetRotations, "mb_motion_get_target_local_rotations_xyzw")
	if version := library.abiVersion(); version != 1 {
		library.Close()
		return nil, fmt.Errorf("motionbricks ABI version %d is unsupported", version)
	}
	return library, nil
}

func (l *Library) Close() error {
	if l == nil || l.handle == 0 {
		return nil
	}
	err := purego.Dlclose(l.handle)
	l.handle = 0
	return err
}

func cString(value string) ([]byte, error) {
	for _, char := range value {
		if char == 0 {
			return nil, errors.New("string contains NUL")
		}
	}
	return append([]byte(value), 0), nil
}

func goString(pointer uintptr) string {
	if pointer == 0 {
		return ""
	}
	const maximum = 1 << 20
	bytes := make([]byte, 0, 64)
	for index := uintptr(0); index < maximum; index++ {
		value := *(*byte)(unsafe.Pointer(pointer + index))
		if value == 0 {
			return string(bytes)
		}
		bytes = append(bytes, value)
	}
	return ""
}

func errorPointer(buffer []byte) unsafe.Pointer { return unsafe.Pointer(&buffer[0]) }

func (l *Library) check(operation string, status uint32, buffer []byte) error {
	if status == 0 {
		return nil
	}
	message := ""
	for index, value := range buffer {
		if value == 0 {
			message = string(buffer[:index])
			break
		}
	}
	if message == "" {
		message = goString(l.statusString(status))
	}
	return fmt.Errorf("%s: %s", operation, message)
}

type Model struct {
	library *Library
	handle  uintptr
}
type Style struct {
	library *Library
	handle  uintptr
	Name    string
	Speed   float32
}
type Agent struct {
	model  *Model
	handle uintptr
}
type Command struct {
	library *Library
	handle  uintptr
}

type Joint struct {
	Name     string     `json:"name"`
	Parent   int32      `json:"parent"`
	Position [3]float32 `json:"position"`
}
type Motion struct {
	Frames    uint64     `json:"frames"`
	Joints    uint64     `json:"joints"`
	Roots     []float32  `json:"roots"`
	Rotations []float32  `json:"rotations"`
	Targets   *Keyframes `json:"-"`
}
type Keyframes struct {
	Frames    uint64    `json:"frames"`
	Joints    uint64    `json:"joints"`
	Roots     []float32 `json:"roots"`
	Rotations []float32 `json:"rotations"`
}

func (l *Library) LoadModel(path string, device Device) (*Model, error) {
	pathBytes, err := cString(path)
	if err != nil {
		return nil, err
	}
	buffer := make([]byte, errorBufferSize)
	var options uintptr
	if err = l.check("create runtime options", l.optionsCreate(unsafe.Pointer(&options), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	defer l.optionsFree(options)
	if err = l.check("select device", l.optionsSetDevice(options, uint32(device), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	var handle uintptr
	status := l.modelLoad(unsafe.Pointer(&pathBytes[0]), options, unsafe.Pointer(&handle), errorPointer(buffer), uint64(len(buffer)))
	if err = l.check("load model", status, buffer); err != nil {
		return nil, err
	}
	return &Model{library: l, handle: handle}, nil
}

func (m *Model) Close() {
	if m != nil && m.handle != 0 {
		m.library.modelFree(m.handle)
		m.handle = 0
	}
}

func (m *Model) Skeleton() ([]Joint, error) {
	buffer := make([]byte, errorBufferSize)
	var count uint32
	if err := m.library.check("get joint count", m.library.modelJointCount(m.handle, unsafe.Pointer(&count), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	joints := make([]Joint, count)
	for index := uint32(0); index < count; index++ {
		var name uintptr
		if err := m.library.check("get joint name", m.library.modelJointName(m.handle, index, unsafe.Pointer(&name), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
			return nil, err
		}
		joints[index].Name = goString(name)
		if err := m.library.check("get joint parent", m.library.modelJointParent(m.handle, index, unsafe.Pointer(&joints[index].Parent), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
			return nil, err
		}
		position := &joints[index].Position
		if err := m.library.check("get neutral position", m.library.modelNeutralPosition(m.handle, index, unsafe.Pointer(&position[0]), unsafe.Pointer(&position[1]), unsafe.Pointer(&position[2]), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
			return nil, err
		}
	}
	return joints, nil
}

func (m *Model) LoadStyle(path string) (*Style, error) {
	pathBytes, err := cString(path)
	if err != nil {
		return nil, err
	}
	buffer := make([]byte, errorBufferSize)
	var handle uintptr
	if err = m.library.check("load style", m.library.styleLoad(m.handle, unsafe.Pointer(&pathBytes[0]), unsafe.Pointer(&handle), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	style := &Style{library: m.library, handle: handle}
	var name uintptr
	if err = m.library.check("get style name", m.library.styleName(handle, unsafe.Pointer(&name), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		style.Close()
		return nil, err
	}
	style.Name = goString(name)
	if err = m.library.check("get style speed", m.library.styleSpeed(handle, unsafe.Pointer(&style.Speed), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		style.Close()
		return nil, err
	}
	return style, nil
}

func (s *Style) Close() {
	if s != nil && s.handle != 0 {
		s.library.styleFree(s.handle)
		s.handle = 0
	}
}

func (m *Model) NewAgent() (*Agent, error) {
	buffer := make([]byte, errorBufferSize)
	var handle uintptr
	if err := m.library.check("create agent", m.library.agentCreate(m.handle, unsafe.Pointer(&handle), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	return &Agent{model: m, handle: handle}, nil
}
func (a *Agent) Close() {
	if a != nil && a.handle != 0 {
		a.model.library.agentFree(a.handle)
		a.handle = 0
	}
}
func (a *Agent) Reset(style *Style) error {
	buffer := make([]byte, errorBufferSize)
	return a.model.library.check("reset agent", a.model.library.agentReset(a.handle, style.handle, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (a *Agent) Advance(frames uint32) error {
	buffer := make([]byte, errorBufferSize)
	return a.model.library.check("advance agent", a.model.library.agentAdvance(a.handle, frames, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (a *Agent) SetContext(roots, rotations []float32, frames uint64) error {
	if frames < 4 || len(roots) != int(frames*3) || len(rotations) != int(frames*34*4) {
		return errors.New("invalid G1 context shape")
	}
	buffer := make([]byte, errorBufferSize)
	return a.model.library.check("set agent context", a.model.library.agentSetContext(a.handle, unsafe.Pointer(&roots[0]), unsafe.Pointer(&rotations[0]), frames, 34, errorPointer(buffer), uint64(len(buffer))), buffer)
}

func (l *Library) NewCommand() (*Command, error) {
	buffer := make([]byte, errorBufferSize)
	var handle uintptr
	if err := l.check("create command", l.commandCreate(unsafe.Pointer(&handle), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	return &Command{library: l, handle: handle}, nil
}
func (c *Command) Close() {
	if c != nil && c.handle != 0 {
		c.library.commandFree(c.handle)
		c.handle = 0
	}
}
func (c *Command) SetStyle(style *Style) error {
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set command style", c.library.commandSetStyle(c.handle, style.handle, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (c *Command) SetMovement(x, y, z float32) error {
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set movement", c.library.commandSetMovement(c.handle, x, y, z, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (c *Command) SetFacing(x, y, z float32) error {
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set facing", c.library.commandSetFacing(c.handle, x, y, z, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (c *Command) SetSpeed(speed float32) error {
	if math.IsNaN(float64(speed)) {
		return errors.New("speed is NaN")
	}
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set speed", c.library.commandSetSpeed(c.handle, speed, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (c *Command) SetSeed(seed uint64) error {
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set seed", c.library.commandSetSeed(c.handle, seed, errorPointer(buffer), uint64(len(buffer))), buffer)
}

// SetSamplingArgmax selects deterministic diagnostics; false (the default)
// uses upstream-style temperature-1 Gumbel sampling with the command seed.
func (c *Command) SetSamplingArgmax(enabled bool) error {
	var flag uint32
	if enabled {
		flag = 1
	}
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set sampling", c.library.commandSetSamplingArgmax(c.handle, flag, errorPointer(buffer), uint64(len(buffer))), buffer)
}
func (c *Command) SetWorldTarget(x, y, z, heading float32, enabled bool) error {
	var flag uint32
	if enabled {
		flag = 1
	}
	buffer := make([]byte, errorBufferSize)
	return c.library.check("set world target", c.library.commandSetWorldTarget(c.handle, x, y, z, heading, flag, errorPointer(buffer), uint64(len(buffer))), buffer)
}

func (a *Agent) Plan(command *Command) (*Motion, error) {
	buffer := make([]byte, errorBufferSize)
	var handle uintptr
	if err := a.model.library.check("plan motion", a.model.library.agentPlan(a.handle, command.handle, unsafe.Pointer(&handle), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	defer a.model.library.motionFree(handle)
	motion := &Motion{}
	var rootsPointer, rotationsPointer, targetRootsPointer, targetRotationsPointer uintptr
	var rootsCount, rotationsCount, targetRootsCount, targetRotationsCount uint64
	if err := a.model.library.check("get frame count", a.model.library.motionFrames(handle, unsafe.Pointer(&motion.Frames), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if err := a.model.library.check("get joint count", a.model.library.motionJoints(handle, unsafe.Pointer(&motion.Joints), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if err := a.model.library.check("get roots", a.model.library.motionRoots(handle, unsafe.Pointer(&rootsPointer), unsafe.Pointer(&rootsCount), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if err := a.model.library.check("get rotations", a.model.library.motionRotations(handle, unsafe.Pointer(&rotationsPointer), unsafe.Pointer(&rotationsCount), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	targets := &Keyframes{Joints: motion.Joints}
	if err := a.model.library.check("get target frame count", a.model.library.motionTargetFrames(handle, unsafe.Pointer(&targets.Frames), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if err := a.model.library.check("get target roots", a.model.library.motionTargetRoots(handle, unsafe.Pointer(&targetRootsPointer), unsafe.Pointer(&targetRootsCount), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if err := a.model.library.check("get target rotations", a.model.library.motionTargetRotations(handle, unsafe.Pointer(&targetRotationsPointer), unsafe.Pointer(&targetRotationsCount), errorPointer(buffer), uint64(len(buffer))), buffer); err != nil {
		return nil, err
	}
	if rootsCount != motion.Frames*3 || rotationsCount != motion.Frames*motion.Joints*4 ||
		targetRootsCount != targets.Frames*3 || targetRotationsCount != targets.Frames*targets.Joints*4 {
		return nil, errors.New("native motion data has inconsistent dimensions")
	}
	motion.Roots = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(rootsPointer)), rootsCount)...)
	motion.Rotations = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(rotationsPointer)), rotationsCount)...)
	targets.Roots = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(targetRootsPointer)), targetRootsCount)...)
	targets.Rotations = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(targetRotationsPointer)), targetRotationsCount)...)
	motion.Targets = targets
	return motion, nil
}
