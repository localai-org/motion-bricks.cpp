package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

// A small valid GLB built without weights or external fixture downloads.
func testAnimationGLB(t *testing.T, joints []mb.Joint, skeleton string) []byte {
	t.Helper()
	const frames = 60
	var bin bytes.Buffer
	write := func(values []float32) {
		if err := binary.Write(&bin, binary.LittleEndian, values); err != nil {
			t.Fatal(err)
		}
	}
	for frame := 0; frame < frames; frame++ {
		write([]float32{float32(frame) / 30})
	}
	for frame := 0; frame < frames; frame++ {
		write([]float32{0, 0.8, 0})
	}
	for frame := 0; frame < frames; frame++ {
		write([]float32{0, 0, 0, 1})
	}
	nodes := make([]map[string]any, len(joints))
	channels := make([]map[string]any, len(joints)+1)
	samplers := make([]map[string]any, len(channels))
	for i, joint := range joints {
		children := []int{}
		for child, candidate := range joints {
			if candidate.Parent == int32(i) {
				children = append(children, child)
			}
		}
		nodes[i] = map[string]any{"name": joint.Name, "children": children}
		channels[i+1] = map[string]any{"sampler": i + 1, "target": map[string]any{"node": i, "path": "rotation"}}
		samplers[i+1] = map[string]any{"input": 0, "output": 2, "interpolation": "LINEAR"}
	}
	channels[0] = map[string]any{"sampler": 0, "target": map[string]any{"node": 0, "path": "translation"}}
	samplers[0] = map[string]any{"input": 0, "output": 1, "interpolation": "LINEAR"}
	doc := map[string]any{
		"asset":  map[string]any{"version": "2.0", "generator": "Kimodo test"},
		"extras": map[string]any{"skeleton": skeleton, "fps": 30, "rotation_order": "xyzw"},
		"nodes":  nodes, "animations": []any{map[string]any{"channels": channels, "samplers": samplers}},
		"buffers": []any{map[string]any{"byteLength": bin.Len()}},
		"bufferViews": []any{
			map[string]any{"buffer": 0, "byteOffset": 0, "byteLength": frames * 4},
			map[string]any{"buffer": 0, "byteOffset": frames * 4, "byteLength": frames * 12},
			map[string]any{"buffer": 0, "byteOffset": frames * 16, "byteLength": frames * 16}},
		"accessors": []any{
			map[string]any{"bufferView": 0, "componentType": 5126, "count": frames, "type": "SCALAR"},
			map[string]any{"bufferView": 1, "componentType": 5126, "count": frames, "type": "VEC3"},
			map[string]any{"bufferView": 2, "componentType": 5126, "count": frames, "type": "VEC4"}},
	}
	encoded, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	for len(encoded)%4 != 0 {
		encoded = append(encoded, ' ')
	}
	var result bytes.Buffer
	_ = binary.Write(&result, binary.LittleEndian, []uint32{0x46546c67, 2, uint32(28 + len(encoded) + bin.Len()), uint32(len(encoded)), 0x4e4f534a})
	result.Write(encoded)
	_ = binary.Write(&result, binary.LittleEndian, []uint32{uint32(bin.Len()), 0x004e4942})
	result.Write(bin.Bytes())
	return result.Bytes()
}

func TestAnimationUpload(t *testing.T) {
	joints := make([]mb.Joint, 34)
	for i := range joints {
		joints[i].Name = fmt.Sprintf("joint%d", i)
		joints[i].Parent = 0
	}
	joints[0].Parent = -1
	demo := &demoServer{joints: joints, static: http.NotFoundHandler()}
	valid := testAnimationGLB(t, joints, "g1skel34")
	upload := func(name string, data []byte, want int) *httptest.ResponseRecorder {
		t.Helper()
		var body bytes.Buffer
		form := multipart.NewWriter(&body)
		part, err := form.CreateFormFile("animation", name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write(data)
		_ = form.Close()
		request := httptest.NewRequest(http.MethodPost, "/api/kimodo/upload", &body)
		request.Header.Set("Content-Type", form.FormDataContentType())
		response := httptest.NewRecorder()
		demo.routes().ServeHTTP(response, request)
		if response.Code != want {
			t.Fatalf("upload %q: status=%d want=%d: %s", name, response.Code, want, response.Body)
		}
		return response
	}
	upload("broken.glb", []byte("not a GLB"), http.StatusBadRequest)
	upload("mesh.glb", testAnimationGLB(t, joints, "smplx"), http.StatusBadRequest)
	upload("animation.txt", valid, http.StatusBadRequest)
	upload("big.glb", make([]byte, maxAnimationUpload+1), http.StatusRequestEntityTooLarge)
	if len(demo.clips) != 0 {
		t.Fatal("invalid uploads changed the catalogue")
	}
	response := upload("../../walk.glb", valid, http.StatusCreated)
	var info kimodoClipInfo
	if err := json.Unmarshal(response.Body.Bytes(), &info); err != nil {
		t.Fatal(err)
	}
	if info.Name != "walk.glb" || info.Frames != 60 || info.Duration != 2 || demo.kimodo[info.ID] == nil {
		t.Fatalf("unexpected upload: %+v", info)
	}
	upload("duplicate.glb", valid, http.StatusOK)
	if len(demo.clips) != 1 {
		t.Fatal("duplicate was added twice")
	}
	meta := httptest.NewRecorder()
	demo.routes().ServeHTTP(meta, httptest.NewRequest(http.MethodGet, "/api/meta", nil))
	if !bytes.Contains(meta.Body.Bytes(), []byte(info.ID)) {
		t.Fatal("upload missing from refreshed metadata")
	}
}
