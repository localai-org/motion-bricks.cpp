package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

const (
	kimodoFPS         = 30
	kimodoJointCount  = 34
	kimodoEntryFrames = 8
)

var errUnsupportedKimodoSkeleton = errors.New("unsupported Kimodo skeleton")

type kimodoClipInfo struct {
	ID       string  `json:"id"`
	Name     string  `json:"name"`
	Prompt   string  `json:"prompt"`
	Frames   uint64  `json:"frames"`
	FPS      uint32  `json:"fps"`
	Duration float64 `json:"duration"`
}

type kimodoClip struct {
	Info   kimodoClipInfo
	Motion mb.Motion
}

type gltfDocument struct {
	Asset struct {
		Version   string `json:"version"`
		Generator string `json:"generator"`
	} `json:"asset"`
	Extras struct {
		Skeleton      string `json:"skeleton"`
		FPS           uint32 `json:"fps"`
		RotationOrder string `json:"rotation_order"`
	} `json:"extras"`
	Nodes []struct {
		Name     string `json:"name"`
		Children []int  `json:"children"`
	} `json:"nodes"`
	BufferViews []struct {
		Buffer     int `json:"buffer"`
		ByteOffset int `json:"byteOffset"`
		ByteLength int `json:"byteLength"`
		ByteStride int `json:"byteStride"`
	} `json:"bufferViews"`
	Accessors []struct {
		BufferView    int    `json:"bufferView"`
		ByteOffset    int    `json:"byteOffset"`
		ComponentType int    `json:"componentType"`
		Count         int    `json:"count"`
		Type          string `json:"type"`
		Sparse        any    `json:"sparse"`
	} `json:"accessors"`
	Animations []struct {
		Samplers []struct {
			Input         int    `json:"input"`
			Output        int    `json:"output"`
			Interpolation string `json:"interpolation"`
		} `json:"samplers"`
		Channels []struct {
			Sampler int `json:"sampler"`
			Target  struct {
				Node int    `json:"node"`
				Path string `json:"path"`
			} `json:"target"`
		} `json:"channels"`
	} `json:"animations"`
}

func readGLB(path string) (gltfDocument, []byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return gltfDocument{}, nil, err
	}
	return readGLBData(data)
}

func readGLBData(data []byte) (gltfDocument, []byte, error) {
	var document gltfDocument
	if len(data) < 28 || len(data) > 128<<20 || binary.LittleEndian.Uint32(data[0:4]) != 0x46546c67 ||
		binary.LittleEndian.Uint32(data[4:8]) != 2 || int(binary.LittleEndian.Uint32(data[8:12])) != len(data) {
		return document, nil, errors.New("invalid GLB 2.0 header")
	}
	offset := 12
	var jsonChunk, binChunk []byte
	for offset < len(data) {
		if offset+8 > len(data) {
			return document, nil, errors.New("truncated GLB chunk header")
		}
		length := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		kind := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
		offset += 8
		if length < 0 || offset+length > len(data) {
			return document, nil, errors.New("truncated GLB chunk")
		}
		switch kind {
		case 0x4e4f534a:
			if jsonChunk != nil {
				return document, nil, errors.New("GLB has multiple JSON chunks")
			}
			jsonChunk = data[offset : offset+length]
		case 0x004e4942:
			if binChunk != nil {
				return document, nil, errors.New("GLB has multiple BIN chunks")
			}
			binChunk = data[offset : offset+length]
		}
		offset += length
	}
	if jsonChunk == nil || binChunk == nil {
		return document, nil, errors.New("GLB must contain JSON and BIN chunks")
	}
	decoder := json.NewDecoder(bytes.NewReader(jsonChunk))
	if err := decoder.Decode(&document); err != nil {
		return document, nil, fmt.Errorf("decode glTF JSON: %w", err)
	}
	return document, binChunk, nil
}

func gltfFloats(document *gltfDocument, bin []byte, accessorIndex int, kind string, components int) ([]float32, error) {
	if accessorIndex < 0 || accessorIndex >= len(document.Accessors) {
		return nil, errors.New("animation accessor is out of bounds")
	}
	accessor := document.Accessors[accessorIndex]
	if accessor.ComponentType != 5126 || accessor.Type != kind || accessor.Count < 1 || accessor.Count > 1_000_000 || accessor.Sparse != nil {
		return nil, fmt.Errorf("unsupported %s animation accessor", kind)
	}
	if accessor.BufferView < 0 || accessor.BufferView >= len(document.BufferViews) {
		return nil, errors.New("animation buffer view is out of bounds")
	}
	view := document.BufferViews[accessor.BufferView]
	if view.Buffer != 0 || (view.ByteStride != 0 && view.ByteStride != components*4) || view.ByteOffset < 0 ||
		view.ByteLength < 0 || accessor.ByteOffset < 0 || view.ByteOffset > len(bin) ||
		view.ByteLength > len(bin)-view.ByteOffset || accessor.ByteOffset > view.ByteLength {
		return nil, errors.New("unsupported animation buffer view")
	}
	start := view.ByteOffset + accessor.ByteOffset
	count := accessor.Count * components
	end := start + count*4
	if start < view.ByteOffset || end > view.ByteOffset+view.ByteLength || end > len(bin) {
		return nil, errors.New("animation accessor exceeds its buffer view")
	}
	values := make([]float32, count)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(bin[start+index*4 : start+index*4+4]))
		if !finite(values[index]) {
			return nil, errors.New("animation contains a non-finite value")
		}
	}
	return values, nil
}

func parseKimodoGLB(path string, joints []mb.Joint) (*kimodoClip, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return parseKimodoGLBData(data, joints)
}

func parseKimodoGLBData(data []byte, joints []mb.Joint) (*kimodoClip, error) {
	document, bin, err := readGLBData(data)
	if err != nil {
		return nil, err
	}
	if document.Extras.Skeleton != "g1skel34" {
		return nil, fmt.Errorf("%w %q", errUnsupportedKimodoSkeleton, document.Extras.Skeleton)
	}
	if document.Asset.Version != "2.0" || !strings.Contains(strings.ToLower(document.Asset.Generator), "kimodo") ||
		document.Extras.FPS != kimodoFPS || document.Extras.RotationOrder != "xyzw" {
		return nil, errors.New("unsupported Kimodo GLB metadata")
	}
	if len(joints) != kimodoJointCount || len(document.Nodes) != len(joints) || len(document.Animations) != 1 {
		return nil, errors.New("Kimodo GLB is not a 34-joint G1 animation")
	}
	parents := make([]int, len(joints))
	for index := range parents {
		parents[index] = -2
	}
	parents[0] = -1
	for parent, node := range document.Nodes {
		if node.Name != joints[parent].Name {
			return nil, fmt.Errorf("joint %d is %q, expected %q", parent, node.Name, joints[parent].Name)
		}
		for _, child := range node.Children {
			if child <= parent || child >= len(joints) || parents[child] != -2 {
				return nil, errors.New("Kimodo joint hierarchy is invalid")
			}
			parents[child] = parent
		}
	}
	for index, joint := range joints {
		if parents[index] != int(joint.Parent) {
			return nil, fmt.Errorf("joint %q parent differs from MotionBricks", joint.Name)
		}
	}
	animation := document.Animations[0]
	if len(animation.Samplers) != kimodoJointCount+1 || len(animation.Channels) != kimodoJointCount+1 {
		return nil, errors.New("Kimodo animation must have one root track and 34 rotation tracks")
	}
	var roots []float32
	rotationTracks := make([][]float32, kimodoJointCount)
	frames := 0
	var times []float32
	for _, channel := range animation.Channels {
		if channel.Sampler < 0 || channel.Sampler >= len(animation.Samplers) || channel.Target.Node < 0 || channel.Target.Node >= kimodoJointCount {
			return nil, errors.New("Kimodo animation channel is out of bounds")
		}
		sampler := animation.Samplers[channel.Sampler]
		if sampler.Interpolation != "LINEAR" {
			return nil, errors.New("Kimodo animation interpolation must be LINEAR")
		}
		channelTimes, readErr := gltfFloats(&document, bin, sampler.Input, "SCALAR", 1)
		if readErr != nil {
			return nil, readErr
		}
		if times == nil {
			times, frames = channelTimes, len(channelTimes)
			if frames > 18000 {
				return nil, errors.New("animation exceeds the 10-minute limit")
			}
		} else if len(channelTimes) != frames {
			return nil, errors.New("Kimodo animation tracks have different lengths")
		} else {
			for index := range times {
				if math.Abs(float64(channelTimes[index]-times[index])) > 1e-6 {
					return nil, errors.New("Kimodo animation tracks use different time grids")
				}
			}
		}
		switch channel.Target.Path {
		case "translation":
			if channel.Target.Node != 0 || roots != nil {
				return nil, errors.New("Kimodo animation has an unexpected translation track")
			}
			roots, readErr = gltfFloats(&document, bin, sampler.Output, "VEC3", 3)
		case "rotation":
			if rotationTracks[channel.Target.Node] != nil {
				return nil, errors.New("Kimodo animation has a duplicate rotation track")
			}
			rotationTracks[channel.Target.Node], readErr = gltfFloats(&document, bin, sampler.Output, "VEC4", 4)
		default:
			return nil, errors.New("Kimodo animation has an unsupported channel")
		}
		if readErr != nil {
			return nil, readErr
		}
		if channel.Target.Path == "translation" && len(roots) != frames*3 ||
			channel.Target.Path == "rotation" && len(rotationTracks[channel.Target.Node]) != frames*4 {
			return nil, errors.New("animation values do not match the time grid")
		}
	}
	if frames < 4 || len(roots) != frames*3 {
		return nil, errors.New("Kimodo animation is missing its root track")
	}
	for frame, value := range times {
		if math.Abs(float64(value)-float64(frame)/kimodoFPS) > 1e-4 {
			return nil, errors.New("Kimodo animation does not use a uniform 30 FPS time grid")
		}
	}
	rotations := make([]float32, frames*kimodoJointCount*4)
	for joint, track := range rotationTracks {
		if len(track) != frames*4 {
			return nil, fmt.Errorf("Kimodo animation is missing rotation track %d", joint)
		}
		for frame := 0; frame < frames; frame++ {
			source := track[frame*4 : frame*4+4]
			length := math.Sqrt(float64(source[0]*source[0] + source[1]*source[1] + source[2]*source[2] + source[3]*source[3]))
			if length < 0.9 || length > 1.1 {
				return nil, fmt.Errorf("Kimodo rotation %d:%d is not normalized", frame, joint)
			}
			for axis := 0; axis < 4; axis++ {
				rotations[(frame*kimodoJointCount+joint)*4+axis] = float32(float64(source[axis]) / length)
			}
		}
	}
	return &kimodoClip{Motion: mb.Motion{Frames: uint64(frames), Joints: kimodoJointCount, Roots: roots, Rotations: rotations}}, nil
}

func kimodoPrompt(directory string) string {
	if data, err := os.ReadFile(filepath.Join(directory, "prompt.txt")); err == nil {
		if value := strings.TrimSpace(string(data)); value != "" {
			return value
		}
	}
	var metadata struct {
		Prompt string `json:"prompt"`
	}
	data, err := os.ReadFile(directory + ".json")
	if err == nil && json.Unmarshal(data, &metadata) == nil {
		return strings.TrimSpace(metadata.Prompt)
	}
	return "Kimodo animation"
}

func loadKimodoDirectory(directory string, joints []mb.Joint) (map[string]*kimodoClip, []kimodoClipInfo, error) {
	clips := make(map[string]*kimodoClip)
	if directory == "" {
		return clips, nil, nil
	}
	err := filepath.WalkDir(directory, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || entry.Name() != "animation.glb" {
			return nil
		}
		clip, parseErr := parseKimodoGLB(path, joints)
		if errors.Is(parseErr, errUnsupportedKimodoSkeleton) {
			return nil
		}
		if parseErr != nil {
			return fmt.Errorf("load %s: %w", path, parseErr)
		}
		relative, relErr := filepath.Rel(directory, filepath.Dir(path))
		if relErr != nil {
			return relErr
		}
		id := filepath.ToSlash(relative)
		if id == "." {
			id = strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
		}
		if _, exists := clips[id]; exists {
			return fmt.Errorf("duplicate Kimodo clip ID %q", id)
		}
		prompt := kimodoPrompt(filepath.Dir(path))
		clip.Info = kimodoClipInfo{ID: id, Name: prompt, Prompt: prompt, Frames: clip.Motion.Frames, FPS: kimodoFPS,
			Duration: float64(clip.Motion.Frames) / kimodoFPS}
		clips[id] = clip
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	infos := make([]kimodoClipInfo, 0, len(clips))
	for _, clip := range clips {
		infos = append(infos, clip.Info)
	}
	sort.Slice(infos, func(i, j int) bool {
		if infos[i].Name == infos[j].Name {
			return infos[i].ID < infos[j].ID
		}
		return infos[i].Name < infos[j].Name
	})
	return clips, infos, nil
}

func multiplyQuaternion(a, b []float32) [4]float32 {
	return [4]float32{
		a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
		a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
		a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
		a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
	}
}

func slerpQuaternion(a, b []float32, amount float32) [4]float32 {
	dot := a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
	copyB := [4]float32{b[0], b[1], b[2], b[3]}
	if dot < 0 {
		dot = -dot
		for index := range copyB {
			copyB[index] = -copyB[index]
		}
	}
	var result [4]float32
	if dot > 0.9995 {
		for index := range result {
			result[index] = a[index] + amount*(copyB[index]-a[index])
		}
	} else {
		theta := float32(math.Acos(float64(math.Min(1, float64(dot)))))
		sine := float32(math.Sin(float64(theta)))
		left := float32(math.Sin(float64((1-amount)*theta))) / sine
		right := float32(math.Sin(float64(amount*theta))) / sine
		for index := range result {
			result[index] = left*a[index] + right*copyB[index]
		}
	}
	length := float32(math.Sqrt(float64(result[0]*result[0] + result[1]*result[1] + result[2]*result[2] + result[3]*result[3])))
	for index := range result {
		result[index] /= length
	}
	return result
}

func alignKimodoClip(clip *kimodoClip, currentRoot [3]float32, currentRotations []float32) (*mb.Motion, error) {
	if clip == nil || len(currentRotations) != kimodoJointCount*4 || !finite(currentRoot[:]...) || !finite(currentRotations...) {
		return nil, errors.New("invalid current pose")
	}
	normalizedCurrent := make([]float32, len(currentRotations))
	for joint := 0; joint < kimodoJointCount; joint++ {
		offset := joint * 4
		value := currentRotations[offset : offset+4]
		length := float32(math.Sqrt(float64(value[0]*value[0] + value[1]*value[1] + value[2]*value[2] + value[3]*value[3])))
		if length < 0.9 || length > 1.1 {
			return nil, fmt.Errorf("current rotation %d is not normalized", joint)
		}
		for axis := 0; axis < 4; axis++ {
			normalizedCurrent[offset+axis] = value[axis] / length
		}
	}
	rootRotation := normalizedCurrent[:4]
	clipRotation := clip.Motion.Rotations[:4]
	heading := func(q []float32) float64 {
		return math.Atan2(float64(2*(q[0]*q[2]+q[3]*q[1])), float64(1-2*(q[0]*q[0]+q[1]*q[1])))
	}
	yaw := heading(rootRotation) - heading(clipRotation)
	sine, cosine := float32(math.Sin(yaw)), float32(math.Cos(yaw))
	yawQ := []float32{0, float32(math.Sin(yaw / 2)), 0, float32(math.Cos(yaw / 2))}
	frames := int(clip.Motion.Frames)
	alignedRoots := make([]float32, frames*3)
	alignedRotations := make([]float32, len(clip.Motion.Rotations))
	base := clip.Motion.Roots[:3]
	for frame := 0; frame < frames; frame++ {
		dx := clip.Motion.Roots[frame*3] - base[0]
		dy := clip.Motion.Roots[frame*3+1] - base[1]
		dz := clip.Motion.Roots[frame*3+2] - base[2]
		alignedRoots[frame*3] = currentRoot[0] + cosine*dx + sine*dz
		alignedRoots[frame*3+1] = currentRoot[1] + dy
		alignedRoots[frame*3+2] = currentRoot[2] - sine*dx + cosine*dz
		root := multiplyQuaternion(yawQ, clip.Motion.Rotations[(frame*kimodoJointCount)*4:(frame*kimodoJointCount)*4+4])
		copy(alignedRotations[(frame*kimodoJointCount)*4:], root[:])
		copy(alignedRotations[(frame*kimodoJointCount+1)*4:(frame+1)*kimodoJointCount*4],
			clip.Motion.Rotations[(frame*kimodoJointCount+1)*4:(frame+1)*kimodoJointCount*4])
	}
	totalFrames := kimodoEntryFrames + frames
	result := &mb.Motion{Frames: uint64(totalFrames), Joints: kimodoJointCount,
		Roots: make([]float32, totalFrames*3), Rotations: make([]float32, totalFrames*kimodoJointCount*4)}
	for frame := 0; frame < kimodoEntryFrames; frame++ {
		amount := float32(frame+1) / float32(kimodoEntryFrames+1)
		amount = amount * amount * (3 - 2*amount)
		for axis := 0; axis < 3; axis++ {
			result.Roots[frame*3+axis] = currentRoot[axis] + amount*(alignedRoots[axis]-currentRoot[axis])
		}
		for joint := 0; joint < kimodoJointCount; joint++ {
			offset := joint * 4
			value := slerpQuaternion(normalizedCurrent[offset:offset+4], alignedRotations[offset:offset+4], amount)
			copy(result.Rotations[(frame*kimodoJointCount+joint)*4:], value[:])
		}
	}
	copy(result.Roots[kimodoEntryFrames*3:], alignedRoots)
	copy(result.Rotations[kimodoEntryFrames*kimodoJointCount*4:], alignedRotations)
	return result, nil
}
