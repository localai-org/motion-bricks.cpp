package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
)

type physicsInfo struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

// Load bounded, validated recordings once. No request-controlled filesystem
// paths, directory listing, uploads, or subprocess execution are exposed.
func physicsRoutes(next http.Handler, directory string) (http.Handler, error) {
	index := []physicsInfo{}
	files := map[string][]byte{}
	if directory != "" {
		entries, err := os.ReadDir(directory)
		if err != nil {
			return nil, err
		}
		var total int64
		validID := regexp.MustCompile(`^[A-Za-z0-9_-]+\.json$`)
		for _, entry := range entries {
			if !validID.MatchString(entry.Name()) || !entry.Type().IsRegular() {
				continue
			}
			info, err := entry.Info()
			if err != nil {
				return nil, err
			}
			total += info.Size()
			if len(files) >= 32 || info.Size() > 16<<20 || total > 64<<20 {
				return nil, fmt.Errorf("physics recordings exceed size/count limit")
			}
			data, err := os.ReadFile(filepath.Join(directory, entry.Name()))
			if err != nil {
				return nil, err
			}
			var doc struct {
				Format   string    `json:"format"`
				Title    string    `json:"title"`
				Recorded bool      `json:"recorded"`
				Frames   int       `json:"frames"`
				FPS      int       `json:"fps"`
				Times    []float64 `json:"times"`
				Joints   []struct {
					Parent int `json:"parent"`
				} `json:"joints"`
				Actual      []float64 `json:"actual_positions"`
				Reference   []float64 `json:"reference_positions"`
				RootError   []float64 `json:"root_error_m"`
				BodyError   []float64 `json:"body_error_m"`
				Contacts    []int     `json:"contacts"`
				Diagnostics *struct {
					JointRMSE *float64 `json:"joint_rmse_rad"`
					RootDrift *float64 `json:"root_xy_final_error_m"`
					Failure   *string  `json:"failure"`
				} `json:"diagnostics"`
			}
			if err := json.Unmarshal(data, &doc); err != nil {
				return nil, err
			}
			n := len(doc.Joints)
			if doc.Diagnostics == nil || doc.Diagnostics.JointRMSE == nil || doc.Diagnostics.RootDrift == nil ||
				*doc.Diagnostics.JointRMSE < 0 || *doc.Diagnostics.RootDrift < 0 {
				return nil, fmt.Errorf("missing/invalid physics diagnostics: %s", entry.Name())
			}
			if doc.Format != "motionbricks-sonic-playback-v1" || !doc.Recorded || doc.FPS != 50 || doc.Frames < 2 || doc.Frames > 15000 ||
				n != 30 || len(doc.Times) != doc.Frames || len(doc.Actual) != doc.Frames*n*3 || len(doc.Reference) != len(doc.Actual) ||
				len(doc.RootError) != doc.Frames || len(doc.BodyError) != doc.Frames || len(doc.Contacts) != doc.Frames {
				return nil, fmt.Errorf("invalid physics recording dimensions: %s", entry.Name())
			}
			for i, j := range doc.Joints {
				if (i == 0 && j.Parent != -1) || (i > 0 && (j.Parent < 0 || j.Parent >= i)) {
					return nil, fmt.Errorf("invalid physics topology")
				}
			}
			for _, array := range [][]float64{doc.Times, doc.Actual, doc.Reference, doc.RootError, doc.BodyError} {
				for _, v := range array {
					if math.IsNaN(v) || math.IsInf(v, 0) {
						return nil, fmt.Errorf("non-finite physics recording")
					}
				}
			}
			for i, t := range doc.Times {
				if (i == 0 && t != 0) || (i > 0 && (t <= doc.Times[i-1] || t-doc.Times[i-1] > .1)) {
					return nil, fmt.Errorf("invalid physics timeline")
				}
			}
			id := entry.Name()[:len(entry.Name())-5]
			files[id] = data
			index = append(index, physicsInfo{ID: id, Title: doc.Title})
		}
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/physics", func(w http.ResponseWriter, r *http.Request) { jsonResponse(w, http.StatusOK, index) })
	mux.HandleFunc("GET /api/physics/{id}", func(w http.ResponseWriter, r *http.Request) {
		data, ok := files[r.PathValue("id")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(data)
	})
	mux.Handle("/", next)
	return mux, nil
}
