package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

type streamCommand struct {
	Type    string     `json:"type"`
	Seq     uint64     `json:"seq"`
	Style   string     `json:"style,omitempty"`
	Move    [2]float32 `json:"move,omitempty"`
	Facing  [2]float32 `json:"facing,omitempty"`
	Clip    string     `json:"clip,omitempty"`
	Enabled bool       `json:"enabled,omitempty"`
}
type streamFrame struct {
	Type                string        `json:"type"`
	Epoch               uint64        `json:"epoch"`
	Tick                uint64        `json:"tick"`
	Time                float64       `json:"time"`
	Root                []float32     `json:"root"`
	Rotations           []float32     `json:"rotations"`
	Physical            []float32     `json:"physical,omitempty"`
	CollisionTransforms []float32     `json:"collision_transforms,omitempty"`
	Parents             []int32       `json:"parents,omitempty"`
	Contacts            uint32        `json:"contacts"`
	Physics             bool          `json:"physics"`
	PhysicsTime         float64       `json:"physics_time"`
	Fallen              bool          `json:"fallen"`
	Paused              bool          `json:"paused"`
	Holding             bool          `json:"holding"`
	Planning            bool          `json:"planning"`
	Error               string        `json:"error,omitempty"`
	Kind                string        `json:"kind"`
	Progress            float64       `json:"progress"`
	Revision            uint64        `json:"revision"`
	Targets             *mb.Keyframes `json:"-"`
	Style               string        `json:"style"`
	SlowTicks           uint64        `json:"slow_ticks"`
}
type streamWire struct {
	op   ws.OpCode
	data []byte
}
type streamPeer struct {
	conn   net.Conn
	events chan streamWire
	poses  chan []byte
	done   chan struct{}
	once   sync.Once
	token  string
}

func (p *streamPeer) close() { p.once.Do(func() { close(p.done); _ = p.conn.Close() }) }
func (p *streamPeer) event(v any) {
	data, _ := json.Marshal(v)
	p.control(ws.OpText, data)
}
func (p *streamPeer) control(op ws.OpCode, data []byte) {
	select {
	case p.events <- streamWire{op, data}:
	default:
		p.close()
	}
}
func (p *streamPeer) pose(data []byte) {
	select {
	case p.poses <- data:
	default:
		select {
		case <-p.poses:
		default:
		}
		select {
		case p.poses <- data:
		default:
		}
	}
}

type streamInput struct {
	peer    *streamPeer
	command streamCommand
}
type streamJoin struct {
	peer   *streamPeer
	resume string
}
type streamPlan struct {
	epoch, revision, seq uint64
	at                   int
	command              planRequest
	context              *mb.Motion
	kind                 string
	clip                 *kimodoClip
}
type streamResult struct {
	job    streamPlan
	motion *mb.Motion
	err    error
	ms     float64
}
type streamHub struct {
	s       *demoServer
	physics *mb.Physics
	ctx     context.Context
	cancel  context.CancelFunc
	done    chan struct{}
	join    chan streamJoin
	leave   chan *streamPeer
	input   chan streamInput
	jobs    chan streamPlan
	results chan streamResult
	workers sync.WaitGroup
}

func newStreamHub(s *demoServer, p *mb.Physics) *streamHub {
	ctx, cancel := context.WithCancel(context.Background())
	h := &streamHub{s: s, physics: p, ctx: ctx, cancel: cancel, done: make(chan struct{}), join: make(chan streamJoin, 8), leave: make(chan *streamPeer, 16), input: make(chan streamInput, 64), jobs: make(chan streamPlan, 1), results: make(chan streamResult, 1)}
	h.workers.Add(1)
	go h.planner()
	go h.run()
	return h
}
func (h *streamHub) Close() { h.cancel(); <-h.done; h.workers.Wait() }
func (h *streamHub) routes(next http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/stream", func(w http.ResponseWriter, r *http.Request) {
		jsonResponse(w, 200, map[string]any{"available": true, "version": 1, "physics": h.physics != nil})
	})
	mux.HandleFunc("GET /api/stream/socket", h.socket)
	mux.HandleFunc("POST /api/live-physics", func(w http.ResponseWriter, r *http.Request) {
		apiError(w, 410, errors.New("use the server-owned WebSocket stream"))
	})
	for _, path := range []string{"/api/session", "/api/plan", "/api/kimodo/start", "/api/kimodo/finish"} {
		mux.HandleFunc("POST "+path, func(w http.ResponseWriter, r *http.Request) {
			apiError(w, 410, errors.New("motion commands are owned by the WebSocket session"))
		})
	}
	mux.Handle("/", next)
	return mux
}
func (h *streamHub) socket(w http.ResponseWriter, r *http.Request) {
	if origin := r.Header.Get("Origin"); origin != "" {
		u, err := url.Parse(origin)
		if err != nil || u.Host != r.Host || (u.Scheme != "http" && u.Scheme != "https") {
			http.Error(w, "cross-origin socket rejected", 403)
			return
		}
	}
	conn, rw, _, err := ws.UpgradeHTTP(r, w)
	if err != nil {
		return
	}
	p := &streamPeer{conn: conn, events: make(chan streamWire, 32), poses: make(chan []byte, 1), done: make(chan struct{})}
	defer p.close()
	select {
	case h.join <- streamJoin{p, r.URL.Query().Get("resume")}:
	case <-h.ctx.Done():
		return
	}
	go func() {
		defer p.close()
		for {
			var packet streamWire
			select {
			case <-p.done:
				return
			case <-h.ctx.Done():
				return
			case packet = <-p.events:
			default:
				select {
				case <-p.done:
					return
				case <-h.ctx.Done():
					return
				case packet = <-p.events:
				case data := <-p.poses:
					packet = streamWire{ws.OpText, data}
				}
			}
			_ = conn.SetWriteDeadline(time.Now().Add(2 * time.Second))
			if wsutil.WriteServerMessage(conn, packet.op, packet.data) != nil || packet.op == ws.OpClose {
				return
			}
		}
	}()
	defer func() {
		select {
		case h.leave <- p:
		case <-h.ctx.Done():
		}
	}()
	reader := wsutil.NewReader(rw, ws.StateServerSide)
	reader.MaxFrameSize = 4096
	reader.CheckUTF8 = true
	handleControl := func(header ws.Header, src io.Reader) error {
		data, err := io.ReadAll(io.LimitReader(src, 126))
		if err != nil || len(data) > 125 {
			return errors.New("invalid control frame")
		}
		switch header.OpCode {
		case ws.OpPing:
			p.control(ws.OpPong, data)
		case ws.OpClose:
			p.control(ws.OpClose, data)
			<-p.done
			return io.EOF
		}
		return nil
	}
	reader.OnIntermediate = handleControl
	window := time.Now()
	count := 0
	for {
		_ = conn.SetReadDeadline(time.Now().Add(40 * time.Second))
		header, err := reader.NextFrame()
		if err != nil {
			return
		}
		if header.OpCode.IsControl() {
			if handleControl(header, reader) != nil {
				return
			}
			continue
		}
		data, err := io.ReadAll(io.LimitReader(reader, 4097))
		if err != nil || len(data) > 4096 {
			return
		}
		if header.OpCode == ws.OpClose {
			return
		}
		if header.OpCode != ws.OpText {
			return
		}
		if time.Since(window) > time.Second {
			window = time.Now()
			count = 0
		}
		count++
		if count > 120 {
			return
		}
		var command streamCommand
		decoder := json.NewDecoder(bytes.NewReader(data))
		decoder.DisallowUnknownFields()
		if err = decoder.Decode(&command); err != nil {
			p.event(map[string]any{"type": "error", "error": "invalid command JSON"})
			continue
		}
		if decoder.Decode(new(any)) != io.EOF {
			p.event(map[string]any{"type": "error", "error": "trailing command JSON"})
			continue
		}
		select {
		case h.input <- streamInput{p, command}:
		case <-p.done:
			return
		case <-h.ctx.Done():
			return
		default:
			p.event(map[string]any{"type": "error", "error": "command queue full"})
		}
	}
}

func (h *streamHub) planner() {
	defer h.workers.Done()
	for {
		select {
		case <-h.ctx.Done():
			return
		case job := <-h.jobs:
			start := time.Now()
			result := streamResult{job: job}
			if job.clip != nil {
				root, rot := sampleStreamMotion(job.context, 0)
				result.motion, result.err = alignKimodoClip(job.clip, [3]float32{root[0], root[1], root[2]}, rot)
			} else {
				style, err := h.s.style(job.command.Style)
				result.err = err
				if err == nil {
					agent, e := h.s.model.NewAgent()
					result.err = e
					if e == nil {
						if job.context == nil {
							result.err = agent.Reset(style)
						} else {
							result.err = agent.SetContext(job.context.Roots, job.context.Rotations, job.context.Frames)
						}
						if result.err == nil {
							result.motion, result.err = h.s.commandPlan(&session{agent: agent}, style, job.command)
						}
						agent.Close()
					}
				}
			}
			if result.err == nil {
				result.err = validateStreamMotion(result.motion)
			}
			result.ms = float64(time.Since(start)) / 1e6
			select {
			case h.results <- result:
			case <-h.ctx.Done():
				return
			}
		}
	}
}

func (h *streamHub) run() {
	defer close(h.done)
	peers := map[*streamPeer]bool{}
	var owner *streamPeer
	token := ""
	lastSeq := uint64(0)
	epoch, revision := uint64(1), uint64(1)
	tick := uint64(0)
	slow := uint64(0)
	var timeline referenceTimeline
	var last *streamFrame
	var targets *mb.Keyframes
	control := planRequest{Style: "walk", Facing: [2]float32{0, 1}, Seed: 1}
	if h.s.styles[control.Style] == nil {
		control.Style = h.s.ordered[0].Name
	}
	paused := false
	physicsOn := false
	physicalReady := false
	var parents []int32
	var previous []float32
	busy := false
	reservedAt := 0
	holdAtEnd := false
	dirty := true
	nextPlan := 0
	lockedUntil := -1
	kind := "motion"
	clipStart, clipEnd := 0, 0
	wantKind := "motion"
	var pendingClip *kimodoClip
	pendingSeq := uint64(0)
	var pendingActionSeq uint64
	type appliedCommand struct {
		at  int
		seq uint64
	}
	var scheduled []appliedCommand
	failure := ""
	deadline := time.Now()
	var disconnected, lastPausedSend time.Time
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	sendAll := func(v any) {
		for p := range peers {
			p.event(v)
		}
	}
	planEvent := func() map[string]any {
		return map[string]any{"type": "targets", "revision": revision, "targets": targets, "style": control.Style}
	}
	snapshot := func(p *streamPeer) {
		if h.physics != nil {
			p.event(map[string]any{"type": "collision_shapes", "id": h.physics.CollisionShapesID(), "shapes": h.physics.CollisionShapes()})
		}
		p.event(planEvent())
		if last != nil {
			data, _ := json.Marshal(last)
			p.pose(data)
		}
	}
	defer func() {
		for p := range peers {
			p.close()
		}
		if h.physics != nil {
			h.physics.Reset()
		}
	}()
	fail := func(err error) {
		failure = err.Error()
		paused = true
		log.Printf("stream paused: epoch=%d tick=%d kind=%s physics=%t: %s", epoch, tick, kind, physicsOn, failure)
		sendAll(map[string]any{"type": "error", "error": failure})
	}
	for {
		select {
		case <-h.ctx.Done():
			return
		case joined := <-h.join:
			p := joined.peer
			if len(peers) >= 8 {
				p.event(map[string]any{"type": "error", "error": "viewer limit reached"})
				p.close()
				continue
			}
			peers[p] = true
			if token == "" || (owner == nil && !disconnected.IsZero() && time.Since(disconnected) > 30*time.Second) {
				token, _ = randomID()
				owner = p
				lastSeq = 0
			} else if joined.resume == token {
				if owner != nil {
					owner.close()
				}
				owner = p
			}
			p.token = ""
			if owner == p {
				p.token = token
			}
			p.event(map[string]any{"type": "hello", "version": 1, "owner": owner == p, "token": p.token, "last_seq": lastSeq, "epoch": epoch, "physics_available": h.physics != nil, "buffer_seconds": .15})
			snapshot(p)
			if owner == p {
				deadline = time.Now()
			}
		case p := <-h.leave:
			delete(peers, p)
			if owner == p {
				owner = nil
				paused = true
				disconnected = time.Now()
				if last != nil {
					last.Paused = true
					data, _ := json.Marshal(last)
					for viewer := range peers {
						viewer.pose(data)
					}
				}
			}
		case input := <-h.input:
			p, c := input.peer, input.command
			if c.Type == "ping" {
				p.event(map[string]any{"type": "pong"})
				continue
			}
			ack := func(state string, extra string) {
				p.event(map[string]any{"type": "ack", "seq": c.Seq, "state": state, "error": extra, "epoch": epoch})
			}
			if p != owner {
				ack("rejected", "spectator cannot control simulation")
				continue
			}
			if c.Seq > 9007199254740991 {
				ack("rejected", "sequence exceeds safe integer range")
				continue
			}
			if c.Seq == 0 || c.Seq <= lastSeq {
				ack("duplicate", "")
				continue
			}
			lastSeq = c.Seq
			switch c.Type {
			case "control":
				if !finite(c.Move[0], c.Move[1], c.Facing[0], c.Facing[1]) || math.Hypot(float64(c.Move[0]), float64(c.Move[1])) > 1.01 || math.Hypot(float64(c.Facing[0]), float64(c.Facing[1])) < .1 || math.Hypot(float64(c.Facing[0]), float64(c.Facing[1])) > 1.01 || h.s.styles[c.Style] == nil {
					ack("rejected", "invalid control vector/style")
					continue
				}
				control.Move = c.Move
				control.Facing = c.Facing
				control.Style = c.Style
				dirty = true
				revision++
				pendingSeq = c.Seq
				ack("received", "")
			case "jump", "play_clip":
				if lockedUntil >= int(tick)*3/5 || pendingClip != nil || wantKind == "jump" {
					ack("rejected", "action already active")
					continue
				}
				if c.Type == "play_clip" {
					h.s.mu.Lock()
					clip := h.s.kimodo[c.Clip]
					h.s.mu.Unlock()
					if clip == nil {
						ack("rejected", "unknown clip")
						continue
					}
					pendingClip = clip
					wantKind = "kimodo"
				} else {
					wantKind = "jump"
				}
				dirty = true
				revision++
				pendingSeq = c.Seq
				pendingActionSeq = c.Seq
				ack("received", "")
			case "pause":
				paused = true
				ack("applied", "")
			case "resume":
				if failure != "" {
					ack("rejected", "reset or disable physics after a failure")
					continue
				}
				paused = false
				deadline = time.Now()
				ack("applied", "")
			case "physics":
				if c.Enabled && h.physics == nil {
					ack("rejected", "physics unavailable")
					continue
				}
				if physicsOn != c.Enabled {
					physicsOn = c.Enabled
					physicalReady = false
					previous = nil
					if h.physics != nil {
						h.physics.Reset()
					}
					failure = ""
					paused = false
					deadline = time.Now()
				}
				ack("applied", "")
			case "reset":
				epoch++
				revision++
				tick = 0
				timeline = referenceTimeline{}
				holdAtEnd = false
				last = nil
				targets = nil
				dirty = true
				nextPlan = 0
				lockedUntil = -1
				kind = "motion"
				wantKind = "motion"
				pendingClip = nil
				physicalReady = false
				previous = nil
				scheduled = nil
				pendingActionSeq = 0
				pendingSeq = 0
				paused = false
				failure = ""
				if h.physics != nil {
					h.physics.Reset()
				}
				deadline = time.Now()
				ack("applied", "")
			default:
				ack("rejected", "unknown command")
			}
		case result := <-h.results:
			busy = false
			job := result.job
			if job.seq != 0 {
				log.Printf("stream plan: seq=%d revision=%d/%d at=%d now=%.1f style=%s move=%v ms=%.1f err=%v", job.seq, job.revision, revision, job.at, float64(tick)*.6, job.command.Style, job.command.Move, result.ms, result.err)
			}
			if job.epoch != epoch || job.revision != revision {
				dirty = true
				continue
			}
			if result.err != nil {
				fail(fmt.Errorf("planner: %w", result.err))
				continue
			}
			if err := timeline.replace(job.at, result.motion, float64(tick)*.6, true); err != nil {
				if float64(job.at) < float64(tick)*.6+1 && timeline.motion != nil {
					dirty = true
					continue
				}
				fail(err)
				continue
			}
			dirty = false
			holdAtEnd = job.kind == "motion" && job.command.Move == [2]float32{}
			targets = result.motion.Targets
			nextPlan = job.at + min(16, int(result.motion.Frames)-4)
			if job.kind != "motion" {
				lockedUntil = job.at + int(result.motion.Frames) - 1
				clipStart = job.at
				clipEnd = lockedUntil
				kind = job.kind
				pendingClip = nil
				wantKind = "motion"
				nextPlan = lockedUntil
			}
			kept := scheduled[:0]
			for _, item := range scheduled {
				if item.at >= job.at {
					sendAll(map[string]any{"type": "ack", "seq": item.seq, "state": "superseded", "epoch": epoch})
				} else {
					kept = append(kept, item)
				}
			}
			scheduled = kept
			if job.seq != 0 {
				scheduled = append(scheduled, appliedCommand{job.at, job.seq})
				if pendingSeq == job.seq {
					pendingSeq = 0
				}
				if pendingActionSeq == job.seq {
					pendingActionSeq = 0
				}
			}
			sendAll(planEvent())
			if job.seq != 0 {
				sendAll(map[string]any{"type": "ack", "seq": job.seq, "state": "scheduled", "at": float64(job.at) / 30, "planner_ms": result.ms, "epoch": epoch})
			}
		case <-ticker.C:
			if owner == nil {
				continue
			}
			frame := float64(tick) * .6
			if lockedUntil >= 0 && frame >= float64(lockedUntil) {
				kind = "motion"
				lockedUntil = -1
			}
			if !busy && !paused && (lockedUntil < 0 || frame >= float64(lockedUntil)-30) && (timeline.motion == nil || dirty || (!holdAtEnd && frame >= float64(nextPlan)-12)) {
				at := 0
				if timeline.motion != nil {
					at = max(int(math.Ceil(frame))+12, nextPlan)
					if dirty {
						at = int(math.Ceil(frame)) + 12
					}
					if lockedUntil >= 0 {
						at = max(at, lockedUntil)
					}
				}
				// A finite authored clip is played in full. Prepare its exit in
				// advance from its final pose, never send the last four frames
				// backwards as the start of the next animation.
				job := streamPlan{epoch: epoch, revision: revision, seq: pendingSeq, at: at, command: control, kind: wantKind, clip: pendingClip}
				// Demo input policy: no direction means stop, not upstream's
				// small forward-direction fallback. Preserve the selected gait
				// in control for when movement resumes; do not change native parity.
				if job.kind == "motion" && job.command.Move == [2]float32{} {
					zero := float32(0)
					job.command.Speed = &zero
					if h.s.styles["idle"] != nil {
						job.command.Style = "idle"
					}
				}
				if job.kind != "motion" {
					job.seq = pendingActionSeq
				}
				control.Seed++
				job.command.Seed = control.Seed
				if timeline.motion != nil {
					job.context = timeline.context(at)
				}
				if job.kind == "jump" {
					speed := float32(5)
					job.command.Speed = &speed
					if job.command.Move == [2]float32{} {
						job.command.Move = control.Facing
					}
				}
				select {
				case h.jobs <- job:
					busy = true
					reservedAt = job.at
				default:
				}
			}
			if timeline.motion == nil || paused {
				deadline = time.Now()
				if last != nil && time.Since(lastPausedSend) > 100*time.Millisecond {
					lastPausedSend = time.Now()
					last.Paused = paused
					last.Error = failure
					data, _ := json.Marshal(last)
					for p := range peers {
						p.pose(data)
					}
				}
				continue
			}
			for steps := 0; steps < 4 && !time.Now().Before(deadline); steps++ {
				frame = float64(tick) * .6
				// A slow/cold plan must not lose its reserved boundary while it
				// computes. Hold both reference and physics clocks before the seam;
				// never discard the plan repeatedly until the reference runs out.
				limit := reservedAt - 1
				if !holdAtEnd {
					limit = min(limit, timeline.end())
				}
				if busy && reservedAt > 0 && frame+.6 > float64(limit) {
					deadline = time.Now()
					if last != nil && time.Since(lastPausedSend) > 100*time.Millisecond {
						lastPausedSend = time.Now()
						last.Planning = true
						data, _ := json.Marshal(last)
						for p := range peers {
							p.pose(data)
						}
					}
					break
				}
				if !holdAtEnd && frame > float64(timeline.end())+1 {
					fail(fmt.Errorf("reference buffer exhausted before a replacement plan was ready (frame=%.1f end=%d busy=%t dirty=%t)", frame, timeline.end(), busy, dirty))
					break
				}
				if physicsOn && !physicalReady {
					root, rot := timeline.sample(frame)
					initial := &mb.Motion{Frames: 2, Joints: 34, Roots: append(append([]float32{}, root...), root...), Rotations: append(append([]float32{}, rot...), rot...)}
					var err error
					parents, err = h.physics.Start(initial, 0)
					if err != nil {
						fail(fmt.Errorf("physics start: %w", err))
						break
					}
					physicalReady = true
					deadline = time.Now()
				}
				var physical mb.PhysicalFrame
				if physicsOn {
					window, at := timeline.window(frame)
					var err error
					physical, err = h.physics.Step(window, at)
					if err != nil {
						fail(fmt.Errorf("physics: %w", err))
						break
					}
					if err = validatePhysicalFrame(physical, previous); err != nil {
						fail(err)
						break
					}
					previous = physical.Actual
				}
				tick++
				for len(scheduled) > 0 && float64(scheduled[0].at) <= float64(tick)*.6 {
					sendAll(map[string]any{"type": "ack", "seq": scheduled[0].seq, "state": "applied", "time": float64(tick) / 50, "epoch": epoch})
					scheduled = scheduled[1:]
				}
				root, rot := timeline.sample(float64(tick) * .6)
				last = &streamFrame{Type: "frame", Epoch: epoch, Tick: tick, Time: float64(tick) / 50, Root: root, Rotations: rot, Physics: physicsOn, Revision: revision, Style: control.Style, Kind: kind, SlowTicks: slow}
				last.Holding = holdAtEnd && float64(tick)*.6 >= float64(timeline.end())
				if kind != "motion" {
					last.Progress = math.Max(0, math.Min(1, (float64(tick)*.6-float64(clipStart))/float64(max(1, clipEnd-clipStart))))
				}
				if physicsOn {
					last.Physical = physical.Actual
					last.CollisionTransforms = physical.CollisionTransforms
					last.Parents = parents
					last.Contacts = physical.Contacts
					last.PhysicsTime = physical.Time
					last.Fallen = physical.Fallen
				}
				data, _ := json.Marshal(last)
				for p := range peers {
					p.pose(data)
				}
				deadline = deadline.Add(20 * time.Millisecond)
				if paused {
					break
				}
			}
			if time.Since(deadline) > 200*time.Millisecond {
				slow++
				deadline = time.Now()
			}
		}
	}
}
