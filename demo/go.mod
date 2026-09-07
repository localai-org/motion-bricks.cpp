module github.com/localai/motion-bricks.cpp/demo

go 1.26

require (
	github.com/chromedp/cdproto v0.0.0-20260714215040-dc233986426f
	github.com/chromedp/chromedp v0.16.0
	github.com/gobwas/ws v1.4.0
	github.com/localai/motion-bricks.cpp/bindings/go v0.0.0
)

require (
	github.com/chromedp/sysutil v1.1.0 // indirect
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/go-json-experiment/json v0.0.0-20260623181947-01eb4420fa68 // indirect
	github.com/gobwas/httphead v0.1.0 // indirect
	github.com/gobwas/pool v0.2.1 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/localai/motion-bricks.cpp/bindings/go => ../bindings/go
