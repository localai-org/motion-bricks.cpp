package main

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strings"
)

const maxAnimationUpload = 16 << 20
const maxUploadedMotionBytes = 128 << 20

// Uploads are validated before registration, kept in bounded memory until
// restart, and never written using user-supplied paths. Serialize parsing to
// bound concurrent allocation without holding the session/catalogue mutex.
func (s *demoServer) uploadKimodo(w http.ResponseWriter, r *http.Request) {
	if !s.uploadMu.TryLock() {
		apiError(w, http.StatusTooManyRequests, errors.New("another animation is uploading; try again shortly"))
		return
	}
	defer s.uploadMu.Unlock()
	r.Body = http.MaxBytesReader(w, r.Body, maxAnimationUpload+(64<<10))
	reader, err := r.MultipartReader()
	if err != nil {
		apiError(w, http.StatusBadRequest, errors.New("upload a GLB in the animation form field"))
		return
	}
	part, err := reader.NextPart()
	if err != nil {
		apiError(w, http.StatusBadRequest, errors.New("animation file is missing"))
		return
	}
	name := filepath.Base(strings.ReplaceAll(part.FileName(), "\\", "/"))
	if part.FormName() != "animation" || len(name) > 255 || !strings.EqualFold(filepath.Ext(name), ".glb") {
		apiError(w, http.StatusBadRequest, errors.New("select one .glb animation file"))
		return
	}
	data, err := io.ReadAll(io.LimitReader(part, maxAnimationUpload+1))
	if err != nil || len(data) > maxAnimationUpload {
		apiError(w, http.StatusRequestEntityTooLarge, errors.New("animation upload must be at most 16 MiB"))
		return
	}
	if _, err = reader.NextPart(); err != io.EOF {
		apiError(w, http.StatusBadRequest, errors.New("upload exactly one animation file"))
		return
	}
	clip, err := parseKimodoGLBData(data, s.joints)
	if err != nil {
		apiError(w, http.StatusBadRequest, fmt.Errorf("cannot import animation: %w; export a Kimodo G1 (g1skel34) animation GLB", err))
		return
	}
	id := fmt.Sprintf("upload-%x", sha256.Sum256(data))
	clip.Info = kimodoClipInfo{ID: id, Name: name, Frames: clip.Motion.Frames, FPS: kimodoFPS,
		Duration: float64(clip.Motion.Frames) / kimodoFPS}
	s.mu.Lock()
	defer s.mu.Unlock()
	if existing := s.kimodo[id]; existing != nil {
		jsonResponse(w, http.StatusOK, existing.Info)
		return
	}
	size := 4 * (len(clip.Motion.Roots) + len(clip.Motion.Rotations))
	if s.uploadBytes+size > maxUploadedMotionBytes || len(s.clips) >= 128 {
		apiError(w, http.StatusInsufficientStorage, errors.New("animation catalogue is full; restart the demo to clear uploads"))
		return
	}
	if s.kimodo == nil {
		s.kimodo = make(map[string]*kimodoClip)
	}
	s.kimodo[id] = clip
	s.clips = append(s.clips, clip.Info)
	s.uploadBytes += size
	jsonResponse(w, http.StatusCreated, clip.Info)
}
