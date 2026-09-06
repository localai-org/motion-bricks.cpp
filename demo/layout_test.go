package main

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	cdpinput "github.com/chromedp/cdproto/input"
	"github.com/chromedp/chromedp"
)

// Exercise the actual embedded markup and CSS without loading any weights.
func TestControlPaneScrolling(t *testing.T) {
	chrome := os.Getenv("MOTIONBRICKS_CHROME")
	if chrome == "" {
		var err error
		chrome, err = exec.LookPath("chromium")
		if err != nil {
			t.Skip("Chromium is not installed")
		}
	}
	page, err := webFiles.ReadFile("web/index.html")
	if err != nil {
		t.Fatal(err)
	}
	css, err := webFiles.ReadFile("web/style.css")
	if err != nil {
		t.Fatal(err)
	}
	markup := strings.ReplaceAll(string(page), `<script type="module" src="app.js"></script>`, "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/style.css" {
			w.Header().Set("Content-Type", "text/css")
			_, _ = w.Write(css)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = fmt.Fprint(w, markup)
	}))
	defer server.Close()
	options := append([]chromedp.ExecAllocatorOption{}, chromedp.DefaultExecAllocatorOptions[:]...)
	options = append(options, chromedp.ExecPath(chrome), chromedp.Flag("no-sandbox", true),
		chromedp.Flag("disable-dev-shm-usage", true))
	allocator, cancelAllocator := chromedp.NewExecAllocator(context.Background(), options...)
	defer cancelAllocator()
	browser, cancelBrowser := chromedp.NewContext(allocator)
	defer cancelBrowser()
	ctx, cancel := context.WithTimeout(browser, 30*time.Second)
	defer cancel()
	for _, size := range [][2]int64{{1280, 600}, {390, 650}, {740, 400}} {
		t.Run(fmt.Sprintf("%dx%d", size[0], size[1]), func(t *testing.T) {
			var point struct{ X, Y float64 }
			var layoutOK, reachable bool
			if err := chromedp.Run(ctx,
				chromedp.EmulateViewport(size[0], size[1]),
				chromedp.Navigate(server.URL),
				chromedp.Evaluate(`document.querySelector('#kimodo-controls').hidden = false`, nil),
				chromedp.Evaluate(`(() => {
				  const pane = document.querySelector('aside'), box = pane.getBoundingClientRect();
				  return {X: box.left + box.width / 2, Y: box.top + box.height / 2};
				})()`, &point),
				chromedp.Evaluate(`(() => {
				  const pane = document.querySelector('aside'), box = pane.getBoundingClientRect();
				  const view = document.querySelector('#viewport').getBoundingClientRect();
				  return getComputedStyle(pane).overflowY === 'auto' &&
				    pane.scrollHeight > pane.clientHeight && pane.scrollWidth <= pane.clientWidth &&
				    box.bottom <= innerHeight + 1 && view.height >= 100 &&
				    document.documentElement.scrollHeight <= innerHeight;
				})()`, &layoutOK),
				chromedp.ActionFunc(func(ctx context.Context) error {
					return cdpinput.DispatchMouseEvent(cdpinput.MouseWheel, point.X, point.Y).WithDeltaY(500).Do(ctx)
				}),
				chromedp.Poll(`document.querySelector('aside').scrollTop > 0`, nil),
				chromedp.Evaluate(`(() => {
				  const jump = document.querySelector('#jump');
				  jump.scrollIntoView({block: 'center'});
				  const box = jump.getBoundingClientRect(), pane = document.querySelector('aside').getBoundingClientRect();
				  return box.top >= pane.top && box.bottom <= pane.bottom &&
				    jump.contains(document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2));
				})()`, &reachable),
			); err != nil {
				t.Fatal(err)
			}
			if !layoutOK || !reachable {
				t.Fatalf("bounded scroll layout=%v, jump visible and unobscured=%v", layoutOK, reachable)
			}
		})
	}
}
