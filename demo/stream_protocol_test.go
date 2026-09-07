package main

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
	mb "github.com/localai/motion-bricks.cpp/bindings/go"
)

// No models or GPU: exercise the actual HTTP upgrade and framed socket reader.
func TestStreamSocketProtocol(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	h := &streamHub{ctx: ctx, join: make(chan streamJoin, 8), leave: make(chan *streamPeer, 8), input: make(chan streamInput, 8)}
	s := httptest.NewServer(http.HandlerFunc(h.socket))
	defer s.Close()
	req, _ := http.NewRequest("GET", s.URL, nil)
	req.Header.Set("Origin", "https://untrusted.example")
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if response.StatusCode != 403 {
		t.Fatal("accepted cross-origin connection")
	}
	conn, buffered, _, err := ws.Dial(ctx, "ws"+strings.TrimPrefix(s.URL, "http"))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(3 * time.Second))
	var source io.Reader = conn
	if buffered != nil {
		source = buffered
	}
	reader := wsutil.NewReader(source, ws.StateClientSide)
	read := func(op ws.OpCode) []byte {
		t.Helper()
		header, e := reader.NextFrame()
		if e != nil {
			t.Fatal(e)
		}
		data, e := io.ReadAll(reader)
		if e != nil {
			t.Fatal(e)
		}
		if header.OpCode != op {
			t.Fatalf("opcode %v != %v", header.OpCode, op)
		}
		return data
	}
	if err = wsutil.WriteClientMessage(conn, ws.OpPing, []byte("probe")); err != nil {
		t.Fatal(err)
	}
	if string(read(ws.OpPong)) != "probe" {
		t.Fatal("wrong pong")
	}
	wsutil.WriteClientText(conn, []byte(`{"type":"pause","seq":1,"unexpected":true}`))
	if !strings.Contains(string(read(ws.OpText)), "invalid command JSON") {
		t.Fatal("accepted unknown field")
	}
	wsutil.WriteClientText(conn, []byte(`{"type":"pause","seq":2} {}`))
	if !strings.Contains(string(read(ws.OpText)), "trailing command JSON") {
		t.Fatal("accepted trailing JSON")
	}
	wsutil.WriteClientText(conn, []byte(`{"type":"pause","seq":3}`))
	select {
	case input := <-h.input:
		if input.command.Seq != 3 || input.command.Type != "pause" {
			t.Fatal(input.command)
		}
	case <-time.After(time.Second):
		t.Fatal("command not delivered")
	}
	wsutil.WriteClientText(conn, []byte(strings.Repeat("x", 4097)))
	if _, err = reader.NextFrame(); err == nil {
		t.Fatal("oversize message did not close socket")
	}
}

func TestStreamOwnershipAndDeduplication(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h := &streamHub{s: &demoServer{styles: map[string]*mb.Style{"walk": {Name: "walk"}}}, ctx: ctx, cancel: cancel, done: make(chan struct{}), join: make(chan streamJoin, 8), leave: make(chan *streamPeer, 8), input: make(chan streamInput, 8), jobs: make(chan streamPlan, 1), results: make(chan streamResult, 1)}
	go h.run()
	defer h.Close()
	peer := func() *streamPeer {
		a, b := net.Pipe()
		t.Cleanup(func() { a.Close(); b.Close() })
		return &streamPeer{conn: a, events: make(chan streamWire, 32), poses: make(chan []byte, 1), done: make(chan struct{})}
	}
	next := func(p *streamPeer, kind string) map[string]any {
		t.Helper()
		for {
			select {
			case event := <-p.events:
				var v map[string]any
				if err := json.Unmarshal(event.data, &v); err != nil {
					t.Fatal(err)
				}
				if v["type"] == kind {
					return v
				}
			case <-time.After(time.Second):
				t.Fatal("missing event", kind)
				return nil
			}
		}
	}
	owner := peer()
	h.join <- streamJoin{peer: owner}
	hello := next(owner, "hello")
	if hello["owner"] != true {
		t.Fatal(hello)
	}
	watcher := peer()
	h.join <- streamJoin{peer: watcher}
	if next(watcher, "hello")["owner"] != false {
		t.Fatal("second viewer controls session")
	}
	h.input <- streamInput{watcher, streamCommand{Type: "reset", Seq: 1}}
	if next(watcher, "ack")["state"] != "rejected" {
		t.Fatal("spectator reset")
	}
	h.input <- streamInput{owner, streamCommand{Type: "reset", Seq: 1}}
	first := next(owner, "ack")
	if first["state"] != "applied" {
		t.Fatal(first)
	}
	h.input <- streamInput{owner, streamCommand{Type: "reset", Seq: 1}}
	again := next(owner, "ack")
	if again["state"] != "duplicate" || again["epoch"] != first["epoch"] {
		t.Fatal("duplicate reset changed epoch", again)
	}
	h.leave <- owner
	resumed := peer()
	h.join <- streamJoin{peer: resumed, resume: hello["token"].(string)}
	resume := next(resumed, "hello")
	if resume["owner"] != true || resume["epoch"] != first["epoch"] || resume["last_seq"] != float64(1) {
		t.Fatal("resume lost state", resume)
	}
}

func TestStreamIdleScheduling(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	h := &streamHub{s: &demoServer{styles: map[string]*mb.Style{"walk": {Name: "walk"}, "idle": {Name: "idle"}}}, ctx: ctx, cancel: cancel, done: make(chan struct{}), join: make(chan streamJoin, 8), leave: make(chan *streamPeer, 8), input: make(chan streamInput, 8), jobs: make(chan streamPlan, 1), results: make(chan streamResult, 1)}
	go h.run()
	defer h.Close()
	a, b := net.Pipe()
	defer b.Close()
	p := &streamPeer{conn: a, events: make(chan streamWire, 32), poses: make(chan []byte, 1), done: make(chan struct{})}
	h.join <- streamJoin{peer: p}
	var job streamPlan
	select {
	case job = <-h.jobs:
	case <-time.After(time.Second):
		t.Fatal("no initial plan")
	}
	if job.command.Style != "idle" || job.command.Speed == nil || *job.command.Speed != 0 {
		t.Fatal("zero direction did not request a stop", job.command)
	}
	h.results <- streamResult{job: job, motion: testMotion(8)}
	time.Sleep(400 * time.Millisecond)
	select {
	case <-h.jobs:
		t.Fatal("settled idle replanned without input")
	default:
	}
	var frame streamFrame
	select {
	case data := <-p.poses:
		if err := json.Unmarshal(data, &frame); err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("no held frame")
	}
	if !frame.Holding || frame.Paused || frame.Error != "" || frame.Time < .3 {
		t.Fatal("idle failed to hold with clock running", frame)
	}
	h.input <- streamInput{p, streamCommand{Type: "control", Seq: 1, Style: "walk", Move: [2]float32{0, 1}, Facing: [2]float32{0, 1}}}
	select {
	case job = <-h.jobs:
	case <-time.After(time.Second):
		t.Fatal("held idle cannot resume")
	}
	if job.command.Style != "walk" || job.command.Speed != nil {
		t.Fatal("idle speed leaked into walking", job.command)
	}
	// Deliberately exceed the lookahead. The simulation must wait before the
	// reserved seam, then accept the same plan instead of exhausting/retrying.
	time.Sleep(600 * time.Millisecond)
	select {
	case data := <-p.poses:
		if err := json.Unmarshal(data, &frame); err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("no planning snapshot")
	}
	if !frame.Planning || frame.Paused || frame.Error != "" || frame.Time*30 > float64(job.at-1) {
		t.Fatal("late planner crossed its reserved seam", frame)
	}
	h.results <- streamResult{job: job, motion: testMotion(24)}
	select {
	case <-h.jobs:
	case <-time.After(time.Second):
		t.Fatal("late planner did not recover to ordinary walking")
	}
}
