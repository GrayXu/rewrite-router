package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pkoukk/tiktoken-go"
)

func initTestTokenizer(t *testing.T) {
	t.Helper()

	if tokenizer != nil {
		return
	}

	encoding, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		t.Fatalf("init tokenizer: %v", err)
	}
	tokenizer = encoding
}

func TestHandleResponsesRoutingRewriteOrder(t *testing.T) {
	initTestTokenizer(t)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}

		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode forwarded request: %v", err)
		}

		if req["model"] != "final-model" {
			t.Fatalf("expected rewritten model, got %v", req["model"])
		}
		if stream, ok := req["stream"].(bool); !ok || stream {
			t.Fatalf("expected stream=false after rewrite, got %v", req["stream"])
		}
		if req["temperature"] != 0.1 {
			t.Fatalf("expected extra rewrite field, got %v", req["temperature"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_123","object":"response"}`))
	}))
	defer backend.Close()

	configVal.Store(Config{
		BackendURL: backend.URL,
		RoutingRules: map[string]RoutingRule{
			"auto-model": {
				Models: map[string]string{
					"512": "routed-model",
				},
				Threshold: 1.0,
			},
		},
		RewriteRules: map[string]RewriteRules{
			"routed-model": {
				"model":       "final-model",
				"stream":      false,
				"temperature": 0.1,
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model": "auto-model",
		"input": "hello world",
		"stream": true
	}`))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	handleResponses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
}

func TestHandleResponsesRewriteMessagesAndTools(t *testing.T) {
	initTestTokenizer(t)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode forwarded request: %v", err)
		}

		input, ok := req["input"].([]interface{})
		if !ok || len(input) != 2 {
			t.Fatalf("expected normalized input with 2 items, got %#v", req["input"])
		}

		first, ok := input[0].(map[string]interface{})
		if !ok {
			t.Fatalf("expected first input item to be object, got %#v", input[0])
		}
		if first["type"] != "message" || first["role"] != "system" {
			t.Fatalf("expected prepended system message, got %#v", first)
		}

		firstContent, ok := first["content"].([]interface{})
		if !ok || len(firstContent) != 1 {
			t.Fatalf("expected one content item, got %#v", first["content"])
		}

		firstText, ok := firstContent[0].(map[string]interface{})
		if !ok || firstText["type"] != "input_text" || firstText["text"] != "prefix rule" {
			t.Fatalf("expected input_text conversion, got %#v", firstContent[0])
		}

		second, ok := input[1].(map[string]interface{})
		if !ok || second["role"] != "user" {
			t.Fatalf("expected normalized user message, got %#v", input[1])
		}

		tools, ok := req["tools"].([]interface{})
		if !ok || len(tools) != 2 {
			t.Fatalf("expected two tools, got %#v", req["tools"])
		}

		firstTool := tools[0].(map[string]interface{})
		secondTool := tools[1].(map[string]interface{})
		if firstTool["name"] != "original" || secondTool["name"] != "rewrite" {
			t.Fatalf("unexpected tool order: %#v", tools)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	configVal.Store(Config{
		BackendURL: backend.URL,
		RewriteRules: map[string]RewriteRules{
			"rewrite-model": {
				"message": []interface{}{
					map[string]interface{}{
						"role":    "system",
						"content": "prefix rule",
					},
				},
				"tools": []interface{}{
					map[string]interface{}{
						"name": "rewrite",
					},
				},
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model": "rewrite-model",
		"input": "user input",
		"tools": [{"name":"original"}]
	}`))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	handleResponses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
}

func TestHandleResponsesRoutingCountsInstructionsAndIgnoresImages(t *testing.T) {
	initTestTokenizer(t)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode forwarded request: %v", err)
		}

		if req["model"] != "large-model" {
			t.Fatalf("expected large-model after routing, got %v", req["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer backend.Close()

	configVal.Store(Config{
		BackendURL: backend.URL,
		RoutingRules: map[string]RoutingRule{
			"auto-model": {
				Models: map[string]string{
					"10":   "small-model",
					"4096": "large-model",
				},
				Threshold: 1.0,
			},
		},
	})

	reqBody := `{
		"model": "auto-model",
		"instructions": "` + strings.Repeat("long instructions ", 100) + `",
		"input": [
			{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://example.com/image.png"}]},
			{"type":"function_call_output","output":"tool output text"}
		]
	}`

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	handleResponses(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}
}

func TestHandleResponsesStreamingPassthrough(t *testing.T) {
	initTestTokenizer(t)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)

		_, _ = w.Write([]byte("data: first\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		_, _ = w.Write([]byte("data: second\n\n"))
	}))
	defer backend.Close()

	configVal.Store(Config{
		BackendURL: backend.URL,
	})

	front := httptest.NewServer(http.HandlerFunc(handleResponses))
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/responses", "application/json", strings.NewReader(`{
		"model": "stream-model",
		"input": "hello",
		"stream": true
	}`))
	if err != nil {
		t.Fatalf("post to front server: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read stream body: %v", err)
	}

	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected event stream content type, got %q", resp.Header.Get("Content-Type"))
	}
	if string(body) != "data: first\n\ndata: second\n\n" {
		t.Fatalf("unexpected stream body: %q", string(body))
	}
}

func TestHandleModelsIncludesVirtualModelNames(t *testing.T) {
	initTestTokenizer(t)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [{"id":"base-model","object":"model","created":123,"owned_by":"backend"}]
		}`))
	}))
	defer backend.Close()

	configVal.Store(Config{
		BackendURL: backend.URL,
		RoutingRules: map[string]RoutingRule{
			"route-alias": {
				Models:    map[string]string{"10": "base-model"},
				Threshold: 1.0,
			},
		},
		RewriteRules: map[string]RewriteRules{
			"rewrite-alias": {
				"model": "base-model",
			},
		},
	})

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handleModels(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rec.Code)
	}

	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	ids := map[string]bool{}
	for _, model := range payload.Data {
		ids[model.ID] = true
	}

	if !ids["base-model"] || !ids["route-alias"] || !ids["rewrite-alias"] {
		t.Fatalf("missing expected models: %#v", ids)
	}
}
