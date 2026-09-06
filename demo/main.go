package main

import (
	"bytes"
	"crypto/rand"
	"embed"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

var errKimodoActive = errors.New("a non-interruptible Kimodo animation is playing")

//go:embed web/* web/vendor/*
var webFiles embed.FS

type styleInfo struct {
	Name  string  `json:"name"`
	Speed float32 `json:"speed"`
}

type session struct {
	mu      sync.Mutex
	agent   *mb.Agent
	planned bool
	kimodo  *mb.Motion
}

type demoServer struct {
	library     *mb.Library
	model       *mb.Model
	styles      map[string]*mb.Style
	ordered     []styleInfo
	joints      []mb.Joint
	kimodo      map[string]*kimodoClip
	clips       []kimodoClipInfo
	planMu      sync.Mutex
	uploadMu    sync.Mutex
	uploadBytes int
	mu          sync.Mutex
	sessions    map[string]*session
	static      http.Handler
}

type replayMetadata struct {
	Runtime      string `json:"runtime"`
	Format       string `json:"format"`
	FPS          uint32 `json:"fps"`
	Frames       uint32 `json:"frames"`
	Joints       uint32 `json:"joints"`
	QPos         uint32 `json:"qpos"`
	Plans        uint32 `json:"plans"`
	TargetFrames uint32 `json:"targetFrames"`
}

type replayServer struct {
	data   []byte
	meta   replayMetadata
	static http.Handler
}

type comparisonPlan struct {
	ExpectedFrames               uint32     `json:"expected_frames"`
	ActualFrames                 uint32     `json:"actual_frames"`
	Movement                     [3]float32 `json:"movement"`
	Facing                       [3]float32 `json:"facing"`
	ExpectedJointPositions       []float32  `json:"expected_joint_positions"`
	NativeJointPositions         []float32  `json:"native_joint_positions"`
	ExpectedTargetJointPositions []float32  `json:"expected_target_joint_positions"`
	NativeTargetJointPositions   []float32  `json:"native_target_joint_positions"`
}
type comparisonDocument struct {
	Format  string           `json:"format"`
	Device  string           `json:"device"`
	FPS     uint32           `json:"fps"`
	Joints  uint32           `json:"joints"`
	Passed  bool             `json:"passed"`
	Parents []int32          `json:"parents"`
	Neutral []float32        `json:"neutral_joints"`
	Plans   []comparisonPlan `json:"plans"`
}
type comparisonServer struct {
	data   []byte
	meta   map[string]any
	static http.Handler
}

type sessionRequest struct {
	Style string `json:"style"`
}
type planRequest struct {
	Session string     `json:"session"`
	Style   string     `json:"style"`
	Move    [2]float32 `json:"move"`
	Facing  [2]float32 `json:"facing"`
	Speed   *float32   `json:"speed,omitempty"`
	Seed    uint64     `json:"seed"`
	Advance uint32     `json:"advance"`
}
type planResponse struct {
	Session string        `json:"session"`
	Style   string        `json:"style"`
	Motion  *mb.Motion    `json:"motion"`
	Targets *mb.Keyframes `json:"targets"`
}
type kimodoStartRequest struct {
	Session          string     `json:"session"`
	Clip             string     `json:"clip"`
	Advance          uint32     `json:"advance"`
	CurrentRoot      [3]float32 `json:"current_root"`
	CurrentRotations []float32  `json:"current_rotations"`
}
type kimodoStartResponse struct {
	Session     string         `json:"session"`
	Clip        kimodoClipInfo `json:"clip"`
	EntryFrames uint32         `json:"entry_frames"`
	Motion      *mb.Motion     `json:"motion"`
}
type kimodoFinishRequest struct {
	Session string     `json:"session"`
	Style   string     `json:"style"`
	Facing  [2]float32 `json:"facing"`
	Move    [2]float32 `json:"move"`
	Seed    uint64     `json:"seed"`
}

func parseDevice(value string) (mb.Device, error) {
	switch strings.ToLower(value) {
	case "auto":
		return mb.DeviceAuto, nil
	case "cpu":
		return mb.DeviceCPU, nil
	case "vulkan":
		return mb.DeviceVulkan, nil
	default:
		return 0, fmt.Errorf("unknown device %q", value)
	}
}

func webHandler() (http.Handler, error) {
	root, err := fs.Sub(webFiles, "web")
	if err != nil {
		return nil, err
	}
	return http.FileServer(http.FS(root)), nil
}

func loadDemoServer(libraryPath, modelPath, styleDirectory string, device mb.Device) (*demoServer, error) {
	library, err := mb.Open(libraryPath)
	if err != nil {
		return nil, fmt.Errorf("open native library: %w", err)
	}
	model, err := library.LoadModel(modelPath, device)
	if err != nil {
		library.Close()
		return nil, err
	}
	paths, err := filepath.Glob(filepath.Join(styleDirectory, "*.mbstyle"))
	if err != nil || len(paths) == 0 {
		model.Close()
		library.Close()
		return nil, fmt.Errorf("no .mbstyle files in %s", styleDirectory)
	}
	sort.Strings(paths)
	server := &demoServer{library: library, model: model, styles: make(map[string]*mb.Style), sessions: make(map[string]*session), kimodo: make(map[string]*kimodoClip)}
	for _, path := range paths {
		style, loadErr := model.LoadStyle(path)
		if loadErr != nil {
			server.Close()
			return nil, fmt.Errorf("load %s: %w", path, loadErr)
		}
		if _, exists := server.styles[style.Name]; exists {
			style.Close()
			server.Close()
			return nil, fmt.Errorf("duplicate style %q", style.Name)
		}
		server.styles[style.Name] = style
		server.ordered = append(server.ordered, styleInfo{Name: style.Name, Speed: style.Speed})
	}
	sort.Slice(server.ordered, func(i, j int) bool { return server.ordered[i].Name < server.ordered[j].Name })
	server.joints, err = model.Skeleton()
	if err != nil {
		server.Close()
		return nil, err
	}
	static, err := webHandler()
	if err != nil {
		server.Close()
		return nil, err
	}
	server.static = static
	return server, nil
}

func (s *demoServer) loadKimodo(directory string) error {
	clips, ordered, err := loadKimodoDirectory(directory, s.joints)
	if err != nil {
		return err
	}
	s.kimodo, s.clips = clips, ordered
	return nil
}

func loadReplayServer(path string) (*replayServer, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect replay: %w", err)
	}
	if info.Size() < 40 || info.Size() > 512<<20 {
		return nil, fmt.Errorf("replay size %d is outside supported bounds", info.Size())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read replay: %w", err)
	}
	if !bytes.Equal(data[:8], []byte{'M', 'B', 'R', 'P', 'L', 'Y', '1', 0}) {
		return nil, errors.New("unsupported replay magic")
	}
	value := func(index int) uint32 { return binary.LittleEndian.Uint32(data[index : index+4]) }
	if value(8) != 1 || value(36) != 0 {
		return nil, errors.New("unsupported replay header")
	}
	meta := replayMetadata{
		Runtime: "replay", Format: "motionbricks-portable-replay-v1", FPS: value(12),
		Frames: value(16), Joints: value(20), QPos: value(24), Plans: value(28), TargetFrames: value(32),
	}
	if meta.FPS == 0 || meta.FPS > 1000 || meta.Frames == 0 || meta.Frames > 1_000_000 ||
		meta.Joints == 0 || meta.Joints > 64 || meta.QPos == 0 || meta.QPos > 256 ||
		meta.Plans == 0 || meta.Plans > meta.Frames || meta.TargetFrames != 4 {
		return nil, errors.New("replay header dimensions are outside supported bounds")
	}
	words := uint64(meta.Joints) + uint64(meta.Frames)*2 + uint64(meta.Frames)*uint64(meta.QPos) +
		uint64(meta.Frames)*uint64(meta.Joints)*3 + uint64(meta.Plans)*3 +
		uint64(meta.Plans)*uint64(meta.TargetFrames)*uint64(meta.Joints)*3
	if words > (512<<20-40)/4 || uint64(len(data)) != 40+words*4 {
		return nil, errors.New("replay payload size does not match its header")
	}
	static, err := webHandler()
	if err != nil {
		return nil, err
	}
	return &replayServer{data: data, meta: meta, static: static}, nil
}

func loadComparisonServer(path string) (*comparisonServer, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("inspect comparison: %w", err)
	}
	if info.Size() < 64 || info.Size() > 512<<20 {
		return nil, fmt.Errorf("comparison size %d is outside supported bounds", info.Size())
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read comparison: %w", err)
	}
	var document comparisonDocument
	decoder := json.NewDecoder(bytes.NewReader(data))
	if err = decoder.Decode(&document); err != nil {
		return nil, fmt.Errorf("decode comparison: %w", err)
	}
	if document.Format != "motionbricks-open-loop-report-v1" || document.FPS == 0 ||
		document.FPS > 1000 || document.Joints == 0 || document.Joints > 64 ||
		len(document.Plans) == 0 || len(document.Plans) > 10000 ||
		len(document.Parents) != int(document.Joints) || len(document.Neutral) != int(document.Joints*3) {
		return nil, errors.New("comparison metadata is unsupported")
	}
	if document.Parents[0] != -1 {
		return nil, errors.New("comparison root parent must be -1")
	}
	for joint := 1; joint < int(document.Joints); joint++ {
		if document.Parents[joint] < 0 || document.Parents[joint] >= int32(joint) {
			return nil, errors.New("comparison joint topology is invalid")
		}
	}
	for index, plan := range document.Plans {
		if plan.ExpectedFrames < 24 || plan.ExpectedFrames > 64 || plan.ActualFrames < 24 || plan.ActualFrames > 64 ||
			len(plan.ExpectedJointPositions) != int(plan.ExpectedFrames*document.Joints*3) ||
			len(plan.NativeJointPositions) != int(plan.ActualFrames*document.Joints*3) ||
			len(plan.ExpectedTargetJointPositions) != int(4*document.Joints*3) ||
			len(plan.NativeTargetJointPositions) != int(4*document.Joints*3) {
			return nil, fmt.Errorf("comparison plan %d dimensions are invalid", index)
		}
		for _, direction := range append(plan.Movement[:], plan.Facing[:]...) {
			if math.IsNaN(float64(direction)) || math.IsInf(float64(direction), 0) {
				return nil, fmt.Errorf("comparison plan %d command is not finite", index)
			}
		}
	}
	static, err := webHandler()
	if err != nil {
		return nil, err
	}
	meta := map[string]any{"runtime": "comparison", "format": document.Format, "fps": document.FPS,
		"joints": document.Joints, "plans": len(document.Plans), "device": document.Device, "passed": document.Passed}
	return &comparisonServer{data: data, meta: meta, static: static}, nil
}

func (s *demoServer) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	for _, item := range s.sessions {
		item.agent.Close()
	}
	s.sessions = nil
	s.mu.Unlock()
	for _, style := range s.styles {
		style.Close()
	}
	if s.model != nil {
		s.model.Close()
	}
	if s.library != nil {
		_ = s.library.Close()
	}
}

func randomID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(value[:]), nil
}

func jsonResponse(writer http.ResponseWriter, status int, value any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.Header().Set("Cache-Control", "no-store")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
func apiError(writer http.ResponseWriter, status int, err error) {
	jsonResponse(writer, status, map[string]string{"error": err.Error()})
}
func decodeJSON(request *http.Request, output any) error {
	decoder := json.NewDecoder(io.LimitReader(request.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(output); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return nil
}
func finite(values ...float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func (s *demoServer) style(name string) (*mb.Style, error) {
	style := s.styles[name]
	if style == nil {
		return nil, fmt.Errorf("unknown style %q", name)
	}
	return style, nil
}

func (s *demoServer) commandPlan(item *session, style *mb.Style, request planRequest) (*mb.Motion, error) {
	if !finite(request.Move[0], request.Move[1], request.Facing[0], request.Facing[1]) {
		return nil, errors.New("control vector is not finite")
	}
	if math.Hypot(float64(request.Facing[0]), float64(request.Facing[1])) < 1e-6 {
		return nil, errors.New("facing vector is zero")
	}
	item.mu.Lock()
	defer item.mu.Unlock()
	if item.kimodo != nil {
		return nil, errKimodoActive
	}
	if item.planned && request.Advance > 0 {
		if err := item.agent.Advance(request.Advance); err != nil {
			return nil, err
		}
	}
	command, err := s.library.NewCommand()
	if err != nil {
		return nil, err
	}
	defer command.Close()
	if err = command.SetStyle(style); err != nil {
		return nil, err
	}
	if err = command.SetMovement(request.Move[0], 0, request.Move[1]); err != nil {
		return nil, err
	}
	if err = command.SetFacing(request.Facing[0], 0, request.Facing[1]); err != nil {
		return nil, err
	}
	if request.Speed != nil {
		if !finite(*request.Speed) || *request.Speed < 0 {
			return nil, errors.New("speed is invalid")
		}
		if err = command.SetSpeed(*request.Speed); err != nil {
			return nil, err
		}
	}
	if err = command.SetSeed(request.Seed); err != nil {
		return nil, err
	}
	s.planMu.Lock()
	motion, err := item.agent.Plan(command)
	s.planMu.Unlock()
	if err == nil {
		item.planned = true
	}
	return motion, err
}

func (s *demoServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, map[string]any{"ok": true})
	})
	mux.HandleFunc("GET /api/meta", func(w http.ResponseWriter, _ *http.Request) {
		s.mu.Lock()
		clips := append([]kimodoClipInfo(nil), s.clips...)
		s.mu.Unlock()
		jsonResponse(w, http.StatusOK, map[string]any{"fps": 30, "joints": s.joints, "styles": s.ordered, "kimodo_clips": clips})
	})
	mux.HandleFunc("POST /api/kimodo/upload", s.uploadKimodo)
	mux.HandleFunc("POST /api/session", s.createSession)
	mux.HandleFunc("POST /api/plan", s.plan)
	mux.HandleFunc("POST /api/kimodo/start", s.startKimodo)
	mux.HandleFunc("POST /api/kimodo/finish", s.finishKimodo)
	mux.Handle("/", s.static)
	return securityHeaders(mux)
}

func (s *replayServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, map[string]any{"ok": true, "runtime": "replay"})
	})
	mux.HandleFunc("GET /api/meta", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, s.meta)
	})
	mux.HandleFunc("GET /api/replay", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		http.ServeContent(w, r, "session.mbreplay", time.Time{}, bytes.NewReader(s.data))
	})
	mux.Handle("/", s.static)
	return securityHeaders(mux)
}

func (s *comparisonServer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, map[string]any{"ok": true, "runtime": "comparison"})
	})
	mux.HandleFunc("GET /api/meta", func(w http.ResponseWriter, _ *http.Request) {
		jsonResponse(w, http.StatusOK, s.meta)
	})
	mux.HandleFunc("GET /api/comparison", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		http.ServeContent(w, r, "open-loop-report.json", time.Time{}, bytes.NewReader(s.data))
	})
	mux.Handle("/", s.static)
	return securityHeaders(mux)
}

func securityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'")
		next.ServeHTTP(w, r)
	})
}

func (s *demoServer) createSession(w http.ResponseWriter, r *http.Request) {
	var request sessionRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	if request.Style == "" {
		if _, ok := s.styles["idle"]; ok {
			request.Style = "idle"
		} else {
			request.Style = s.ordered[0].Name
		}
	}
	style, err := s.style(request.Style)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	agent, err := s.model.NewAgent()
	if err != nil {
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	if err = agent.Reset(style); err != nil {
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	id, err := randomID()
	if err != nil {
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	item := &session{agent: agent}
	s.mu.Lock()
	s.sessions[id] = item
	s.mu.Unlock()
	motion, err := s.commandPlan(item, style, planRequest{Move: [2]float32{0, 0}, Facing: [2]float32{0, 1}, Seed: 1})
	if err != nil {
		s.mu.Lock()
		delete(s.sessions, id)
		s.mu.Unlock()
		agent.Close()
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	jsonResponse(w, http.StatusOK, planResponse{Session: id, Style: style.Name, Motion: motion, Targets: motion.Targets})
}

func (s *demoServer) plan(w http.ResponseWriter, r *http.Request) {
	var request planRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	s.mu.Lock()
	item := s.sessions[request.Session]
	s.mu.Unlock()
	if item == nil {
		apiError(w, http.StatusNotFound, errors.New("unknown session"))
		return
	}
	style, err := s.style(request.Style)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	motion, err := s.commandPlan(item, style, request)
	if err != nil {
		if errors.Is(err, errKimodoActive) {
			apiError(w, http.StatusConflict, err)
			return
		}
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	jsonResponse(w, http.StatusOK, planResponse{Session: request.Session, Style: style.Name, Motion: motion, Targets: motion.Targets})
}

func (s *demoServer) startKimodo(w http.ResponseWriter, r *http.Request) {
	var request kimodoStartRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	s.mu.Lock()
	item := s.sessions[request.Session]
	clip := s.kimodo[request.Clip]
	s.mu.Unlock()
	if item == nil {
		apiError(w, http.StatusNotFound, errors.New("unknown session"))
		return
	}
	if clip == nil {
		apiError(w, http.StatusBadRequest, fmt.Errorf("unknown Kimodo clip %q", request.Clip))
		return
	}
	item.mu.Lock()
	defer item.mu.Unlock()
	if item.kimodo != nil {
		apiError(w, http.StatusConflict, errors.New("a Kimodo animation is already playing"))
		return
	}
	if len(request.CurrentRotations) != kimodoJointCount*4 || !finite(request.CurrentRoot[:]...) || !finite(request.CurrentRotations...) {
		apiError(w, http.StatusBadRequest, errors.New("current G1 pose is invalid"))
		return
	}
	if item.planned && request.Advance > 0 {
		if err := item.agent.Advance(request.Advance); err != nil {
			apiError(w, http.StatusInternalServerError, err)
			return
		}
	}
	motion, err := alignKimodoClip(clip, request.CurrentRoot, request.CurrentRotations)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	item.kimodo = motion
	jsonResponse(w, http.StatusOK, kimodoStartResponse{Session: request.Session, Clip: clip.Info,
		EntryFrames: kimodoEntryFrames, Motion: motion})
}

func (s *demoServer) finishKimodo(w http.ResponseWriter, r *http.Request) {
	var request kimodoFinishRequest
	if err := decodeJSON(r, &request); err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	s.mu.Lock()
	item := s.sessions[request.Session]
	s.mu.Unlock()
	if item == nil {
		apiError(w, http.StatusNotFound, errors.New("unknown session"))
		return
	}
	style, err := s.style(request.Style)
	if err != nil {
		apiError(w, http.StatusBadRequest, err)
		return
	}
	if !finite(request.Move[0], request.Move[1], request.Facing[0], request.Facing[1]) || math.Hypot(float64(request.Facing[0]), float64(request.Facing[1])) < 1e-6 {
		apiError(w, http.StatusBadRequest, errors.New("facing vector is invalid"))
		return
	}
	item.mu.Lock()
	if item.kimodo == nil {
		item.mu.Unlock()
		apiError(w, http.StatusConflict, errors.New("no Kimodo animation is playing"))
		return
	}
	motion := item.kimodo
	first := int(motion.Frames-4) * 3
	firstRotation := int(motion.Frames-4) * kimodoJointCount * 4
	err = item.agent.SetContext(motion.Roots[first:], motion.Rotations[firstRotation:], 4)
	if err == nil {
		item.kimodo = nil
		item.planned = false
	}
	item.mu.Unlock()
	if err != nil {
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	planned, err := s.commandPlan(item, style, planRequest{Move: request.Move, Facing: request.Facing, Seed: request.Seed})
	if err != nil {
		apiError(w, http.StatusInternalServerError, err)
		return
	}
	jsonResponse(w, http.StatusOK, planResponse{Session: request.Session, Style: style.Name, Motion: planned, Targets: planned.Targets})
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "HTTP listen address")
	libraryPath := flag.String("library", os.Getenv("MOTIONBRICKS_LIB"), "path to libmotionbricks")
	modelPath := flag.String("model", os.Getenv("MOTIONBRICKS_MODEL"), "model bundle directory")
	stylesPath := flag.String("styles", os.Getenv("MOTIONBRICKS_STYLES"), "style directory")
	kimodoPath := flag.String("kimodo-dir", os.Getenv("MOTIONBRICKS_KIMODO_DIR"), "directory containing Kimodo G1 animation.glb files")
	replayPath := flag.String("replay", os.Getenv("MOTIONBRICKS_REPLAY"), "portable .mbreplay session (disables native planning)")
	comparisonPath := flag.String("comparison", os.Getenv("MOTIONBRICKS_COMPARISON"), "open-loop parity report JSON (disables native planning)")
	deviceName := flag.String("device", "cpu", "auto, cpu, or vulkan")
	flag.Parse()
	var handler http.Handler
	closeDemo := func() {}
	if *replayPath != "" && *comparisonPath != "" {
		log.Fatal("-replay and -comparison are mutually exclusive")
	} else if *comparisonPath != "" {
		comparison, err := loadComparisonServer(*comparisonPath)
		if err != nil {
			log.Fatal(err)
		}
		handler = comparison.routes()
	} else if *replayPath != "" {
		replay, err := loadReplayServer(*replayPath)
		if err != nil {
			log.Fatal(err)
		}
		handler = replay.routes()
	} else {
		if *libraryPath == "" || *modelPath == "" || *stylesPath == "" {
			log.Fatal("-library, -model, and -styles are required unless -replay or -comparison is provided")
		}
		device, err := parseDevice(*deviceName)
		if err != nil {
			log.Fatal(err)
		}
		demo, err := loadDemoServer(*libraryPath, *modelPath, *stylesPath, device)
		if err != nil {
			log.Fatal(err)
		}
		if err = demo.loadKimodo(*kimodoPath); err != nil {
			demo.Close()
			log.Fatal(err)
		}
		if len(demo.clips) > 0 {
			log.Printf("loaded %d Kimodo G1 animations", len(demo.clips))
		}
		handler = demo.routes()
		closeDemo = demo.Close
	}
	defer closeDemo()
	server := &http.Server{Addr: *listen, Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	stopped := make(chan os.Signal, 1)
	signal.Notify(stopped, os.Interrupt, syscall.SIGTERM)
	go func() { <-stopped; _ = server.Close() }()
	log.Printf("MotionBricks demo: http://%s", *listen)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
