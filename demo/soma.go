package main

import (
	"encoding/json"
	"errors"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
	"math"
	"net/http"
	"os"
	"sync"
)

type somaStyle struct {
	TargetHorizon float32   `json:"target_horizon"`
	VelocityScale float32   `json:"velocity_scale"`
	Name          string    `json:"name"`
	Speed         float32   `json:"speed"`
	Heading       float64   `json:"heading"`
	Local         []float32 `json:"local"`
	World         []float32 `json:"world"`
}
type somaAssets struct {
	Format   string              `json:"format"`
	Skeleton mb.RetargetSkeleton `json:"skeleton"`
	Styles   []somaStyle         `json:"styles"`
}
type somaServer struct {
	mu               sync.Mutex
	model            *poseRequest
	assets           somaAssets
	controllerAssets *locomotionAssets
}

func loadSOMAServer(library, bundle, assetsPath string, device mb.Device) (*somaServer, error) {
	data, err := os.ReadFile(assetsPath)
	if err != nil {
		return nil, err
	}
	s := &somaServer{}
	if err = json.Unmarshal(data, &s.assets); err != nil {
		return nil, err
	}
	if s.assets.Format == "motionbricks-controller-assets-v1" {
		a, e := loadLocomotionAssets(assetsPath)
		if e != nil {
			return nil, e
		}
		s.controllerAssets = a
		skeleton := mb.RetargetSkeleton{Names: a.Names, Parents: a.Parents, Reference: a.Reference, Mapping: a.ModelToPose}
		for i, name := range a.Names {
			if name == "RightLeg" {
				skeleton.RightLeg = uint32(i)
			}
			if name == "LeftLeg" {
				skeleton.LeftLeg = uint32(i)
			}
		}
		s.assets.Skeleton = skeleton
		// The skeleton preview uses a model-space reference pose, assembled by FK.
		world := make([]float32, a.Joints*3)
		transforms := make([][7]float32, a.Joints)
		for i, parent := range a.Parents {
			copy(transforms[i][:], a.Reference[i*7:i*7+7])
			if parent >= 0 {
				transforms[i] = controllerTransform(transforms[parent][:], transforms[i][:])
			}
			copy(world[i*3:i*3+3], transforms[i][:3])
		}
		s.assets.Styles = []somaStyle{{Name: "idle", World: world}}
		s.model, e = openPoseRequest(library, bundle, device, skeleton)
		if e != nil {
			return nil, e
		}
		// Compile cold GPU pipelines before accepting interactive commands. This
		// throwaway policy does not advance the stream's initial idle/history state.
		warm := newLocomotionController(a, s.controllerGenerator)
		if e = warm.Step(locomotionInput{Mode: "linear_1d", Move: groundVector{1, 0}, Facing: groundVector{1, 0}}, .02); e != nil {
			s.model.Close()
			return nil, e
		}
		return s, nil
	}
	a := s.assets
	n := len(a.Skeleton.Parents)
	if a.Format != "motionbricks-soma-interactive-v1" || n < 30 || n > 256 || len(a.Skeleton.Names) != n || len(a.Styles) == 0 {
		return nil, errors.New("invalid SOMA interactive assets")
	}
	seen := map[string]bool{}
	for i := range s.assets.Styles {
		v := &s.assets.Styles[i]
		// Older assets retain their original speed and distance settings.
		if v.TargetHorizon == 0 {
			v.TargetHorizon = .8
		}
		if v.VelocityScale == 0 {
			v.VelocityScale = 1
		}
		if !finite(v.TargetHorizon, v.VelocityScale) || v.TargetHorizon < 0 || v.VelocityScale < 0 {
			return nil, errors.New("invalid SOMA target tuning")
		}
		if v.Name == "" || seen[v.Name] || len(v.Local) != 4*n*7 || len(v.World) != n*3 || !finite(v.Local...) || !finite(v.World...) || !finite(v.Speed) || v.Speed < 0 || math.IsNaN(v.Heading) || math.IsInf(v.Heading, 0) {
			return nil, errors.New("invalid SOMA style")
		}
		seen[v.Name] = true
	}
	if !seen["idle"] {
		return nil, errors.New("SOMA requires an idle style")
	}
	s.model, err = openPoseRequest(library, bundle, device, a.Skeleton)
	if err != nil {
		return nil, err
	}
	return s, nil
}
func (s *somaServer) style(name string) *somaStyle {
	for i := range s.assets.Styles {
		if s.assets.Styles[i].Name == name {
			return &s.assets.Styles[i]
		}
	}
	return nil
}

// placeSOMATarget rotates the authored root trajectory into the requested facing,
// and places four constraints ahead of the current root at the authored speed.
// All child local transforms remain untouched.
func placeSOMATarget(style *somaStyle, context []float32, n int, move, facing [2]float32) []float32 {
	target := append([]float32(nil), style.Local...)
	width := n * 7
	root := context[3*width : 3*width+7]
	heading := math.Atan2(float64(facing[0]), float64(facing[1]))
	angle := heading - style.Heading
	c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
	qy, qw := float32(math.Sin(angle/2)), float32(math.Cos(angle/2))
	length := float32(math.Hypot(float64(move[0]), float64(move[1])))
	speed := style.Speed * style.VelocityScale
	if length > 1 {
		move[0] /= length
		move[1] /= length
	}
	x, z := style.Local[0], style.Local[2]
	sourceSpeed := float32(math.Hypot(float64(style.Local[3*width]-x), float64(style.Local[3*width+2]-z))) * 10
	velocityScale := float32(1)
	if sourceSpeed > 1e-6 {
		velocityScale = speed * min(length, 1) / sourceSpeed
	}
	for f := 0; f < 4; f++ {
		p := target[f*width : f*width+7]
		dx, dz := (p[0]-x)*velocityScale, (p[2]-z)*velocityScale
		p[0] = root[0] + move[0]*speed*style.TargetHorizon + c*dx + s*dz
		p[2] = root[2] + move[1]*speed*style.TargetHorizon - s*dx + c*dz
		qx, oy, qz, ow := p[3], p[4], p[5], p[6]
		p[3] = qw*qx + qy*qz
		p[4] = qw*oy + qy*ow
		p[5] = qw*qz - qy*qx
		p[6] = qw*ow - qy*oy
	}
	return target
}
func (s *somaServer) routes(next http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, r *http.Request) { jsonResponse(w, 200, map[string]any{"ok": true}) })
	mux.HandleFunc("GET /api/meta", func(w http.ResponseWriter, r *http.Request) {
		idle := s.style("idle")
		joints := make([]mb.Joint, len(s.assets.Skeleton.Parents))
		for i := range joints {
			joints[i] = mb.Joint{Name: s.assets.Skeleton.Names[i], Parent: s.assets.Skeleton.Parents[i]}
			copy(joints[i].Position[:], idle.World[i*3:i*3+3])
		}
		jsonResponse(w, 200, map[string]any{"model": "soma", "fps": 30, "joints": joints, "styles": (somaStreamPlanner{s}).styles(), "precision": "fp32"})
	})
	mux.Handle("/", next)
	return securityHeaders(mux)
}

// The demo owns the generic library/model/request handles together.
type poseRequest struct {
	*mb.InferenceRequest
	model   *mb.Model
	library *mb.Library
}

func (p *poseRequest) Close() {
	if p != nil {
		p.InferenceRequest.Close()
		p.model.Close()
		p.library.Close()
	}
}
func openPoseRequest(path, bundle string, device mb.Device, skeleton mb.RetargetSkeleton) (*poseRequest, error) {
	l, e := mb.Open(path)
	if e != nil {
		return nil, e
	}
	m, e := l.LoadModelKind(bundle, device, mb.ModelSOMA)
	if e != nil {
		l.Close()
		return nil, e
	}
	r, e := m.NewInferenceRequest()
	if e != nil {
		m.Close()
		l.Close()
		return nil, e
	}
	p := &poseRequest{r, m, l}
	if e = r.SetSkeleton(skeleton); e != nil {
		p.Close()
		return nil, e
	}
	return p, nil
}
