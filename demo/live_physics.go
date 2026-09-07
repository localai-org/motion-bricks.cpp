package main

import (
	"errors"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"net/http"
	"sync"
	"time"
)

// One bounded physical worker. Requests never select server filesystem paths.
// A browser owns its state until release or 30 seconds of inactivity. No queue
// grows behind simulation: overlapping work receives an explicit busy error.
func livePhysicsRoutes(next http.Handler, physics *mb.Physics) http.Handler {
	mux := http.NewServeMux()
	var mu sync.Mutex
	owner := ""
	var touched time.Time
	var parents []int32
	mux.HandleFunc("GET /api/live-physics", func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, 200, map[string]any{"available": physics != nil, "controller_hz": 50, "physics_hz": 200, "inference": "GGML", "simulation": "MuJoCo"})
	})
	mux.HandleFunc("POST /api/live-physics", func(w http.ResponseWriter, r *http.Request) {
		if physics == nil {
			apiError(w, 503, errors.New("native physics is not configured"))
			return
		}
		if !mu.TryLock() {
			apiError(w, 409, errors.New("physical worker busy; retry without advancing playback"))
			return
		}
		defer mu.Unlock()
		var req struct {
			Session string     `json:"session"`
			Motion  *mb.Motion `json:"motion"`
			Frame   float64    `json:"frame"`
			Steps   uint32     `json:"steps"`
			Release bool       `json:"release"`
		}
		if err := decodeJSON(r, &req); err != nil {
			apiError(w, 400, err)
			return
		}
		if len(req.Session) < 8 || len(req.Session) > 128 {
			apiError(w, 400, errors.New("valid session required"))
			return
		}
		if owner != "" && time.Since(touched) > 30*time.Second {
			physics.Reset()
			owner = ""
		}
		if owner != "" && owner != req.Session {
			apiError(w, 409, errors.New("another browser owns the physical worker"))
			return
		}
		if req.Release {
			physics.Reset()
			owner = ""
			jsonResponse(w, 200, map[string]bool{"released": true})
			return
		}
		if req.Steps < 1 || req.Steps > 5 {
			apiError(w, 400, errors.New("request 1–5 control ticks"))
			return
		}
		if owner == "" {
			var err error
			parents, err = physics.Start(req.Motion, req.Frame)
			if err != nil {
				apiError(w, 400, err)
				return
			}
			owner = req.Session
		}
		touched = time.Now()
		frames := make([]mb.PhysicalFrame, 0, req.Steps)
		for i := uint32(0); i < req.Steps; i++ {
			frame, err := physics.Step(req.Motion, req.Frame+float64(i)*.6)
			if err != nil {
				apiError(w, 400, err)
				return
			}
			frames = append(frames, frame)
		}
		jsonResponse(w, 200, map[string]any{"frames": frames, "parents": parents, "inference": "GGML", "simulation": "MuJoCo"})
	})
	mux.Handle("/", next)
	return mux
}
