package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

// --- Configuration Structures ---

type RoutingRule struct {
	Models    map[string]string `json:"models"`
	Threshold float64           `json:"threshold"`
}

type RewriteRules map[string]interface{}

type Config struct {
	BackendURL   string                  `json:"BACKEND_URL"`
	RoutingRules map[string]RoutingRule  `json:"ROUTING_RULES"`
	RewriteRules map[string]RewriteRules `json:"REWRITE_RULES"`
}

// --- Global Variables ---

var (
	configVal atomic.Value
	tokenizer *tiktoken.Tiktoken
)

const (
	colorGreen = "\033[32m"
	colorReset = "\033[0m"
)

func colorModel(name string) string {
	return colorGreen + name + colorReset
}

// --- Token Counting Helper ---

func getTokenCount(text string) int {
	if tokenizer == nil {
		return 0
	}
	tokens := tokenizer.Encode(text, nil, nil)
	return len(tokens)
}

// Message represents a chat message
type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or array of objects
}

// ChatRequest represents the incoming request body for /v1/chat/completions
// We use map[string]interface{} for flexibility in rewriting, but specific structs for logic
type ChatRequest struct {
	Model    string        `json:"model"`
	Messages []Message     `json:"messages"`
	Stream   bool          `json:"stream"`
	Tools    []interface{} `json:"tools,omitempty"`
	// Capture other fields to preserve them
	Other map[string]interface{} `json:"-"`
}

// ResponsesRequest represents the incoming request body for /v1/responses.
type ResponsesRequest struct {
	Model        string                 `json:"model"`
	Input        interface{}            `json:"input"`
	Instructions interface{}            `json:"instructions,omitempty"`
	Stream       bool                   `json:"stream"`
	Tools        []interface{}          `json:"tools,omitempty"`
	Other        map[string]interface{} `json:"-"`
}

// Custom Unmarshal to handle "Other" fields
func (c *ChatRequest) UnmarshalJSON(data []byte) error {
	type Alias ChatRequest
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(c),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	// Remove known fields from map
	delete(m, "model")
	delete(m, "messages")
	delete(m, "stream")
	delete(m, "tools")
	c.Other = m
	return nil
}

func (c ChatRequest) MarshalJSON() ([]byte, error) {
	// Create a map combining specific fields and Other
	m := make(map[string]interface{})
	for k, v := range c.Other {
		m[k] = v
	}
	m["model"] = c.Model
	m["messages"] = c.Messages
	m["stream"] = c.Stream
	if c.Tools != nil {
		m["tools"] = c.Tools
	}
	return json.Marshal(m)
}

// Custom Unmarshal to handle "Other" fields.
func (r *ResponsesRequest) UnmarshalJSON(data []byte) error {
	type Alias ResponsesRequest
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(r),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	var m map[string]interface{}
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}

	delete(m, "model")
	delete(m, "input")
	delete(m, "instructions")
	delete(m, "stream")
	delete(m, "tools")
	r.Other = m
	return nil
}

func (r ResponsesRequest) MarshalJSON() ([]byte, error) {
	m := make(map[string]interface{})
	for k, v := range r.Other {
		m[k] = v
	}
	m["model"] = r.Model
	m["input"] = r.Input
	m["stream"] = r.Stream
	if r.Instructions != nil {
		m["instructions"] = r.Instructions
	}
	if r.Tools != nil {
		m["tools"] = r.Tools
	}
	return json.Marshal(m)
}

func getMessagesTokenCount(messages []Message) int {
	totalTokens := 2 // Base format overhead
	perMessageTokens := 4

	for _, msg := range messages {
		totalTokens += perMessageTokens

		// Handle Content which can be string or array
		var textContent string
		switch v := msg.Content.(type) {
		case string:
			textContent = v
		case []interface{}:
			for _, part := range v {
				if partMap, ok := part.(map[string]interface{}); ok {
					if text, ok := partMap["text"].(string); ok {
						textContent += text
					}
				}
			}
		}
		totalTokens += getTokenCount(textContent)
	}
	return totalTokens
}

func getInterfaceTokenCount(value interface{}) int {
	switch v := value.(type) {
	case nil:
		return 0
	case string:
		return getTokenCount(v)
	case []interface{}:
		total := 0
		for _, item := range v {
			total += getInterfaceTokenCount(item)
		}
		return total
	case map[string]interface{}:
		total := 0

		if text, ok := v["text"].(string); ok {
			total += getTokenCount(text)
		}
		if content, ok := v["content"]; ok {
			total += getInterfaceTokenCount(content)
		}
		if output, ok := v["output"]; ok {
			total += getInterfaceTokenCount(output)
		}

		return total
	default:
		return 0
	}
}

func getResponsesTokenCount(req ResponsesRequest) int {
	return getInterfaceTokenCount(req.Instructions) + getInterfaceTokenCount(req.Input)
}

// --- Logic Functions ---

func selectModelForTokenCount(tokenCount int, rule RoutingRule) string {
	if len(rule.Models) == 0 || rule.Threshold == 0 {
		return ""
	}

	// Extract lengths and sort
	var lengths []int
	lengthToModel := make(map[int]string)
	for k, v := range rule.Models {
		l, err := strconv.Atoi(k)
		if err == nil {
			lengths = append(lengths, l)
			lengthToModel[l] = v
		}
	}
	sort.Ints(lengths)

	for _, length := range lengths {
		if float64(tokenCount) <= float64(length)*rule.Threshold {
			return lengthToModel[length]
		}
	}

	// Return largest model if none match
	if len(lengths) > 0 {
		return lengthToModel[lengths[len(lengths)-1]]
	}
	return ""
}

func selectModel(messages []Message, rule RoutingRule) string {
	return selectModelForTokenCount(getMessagesTokenCount(messages), rule)
}

func selectResponsesModel(req ResponsesRequest, rule RoutingRule) string {
	return selectModelForTokenCount(getResponsesTokenCount(req), rule)
}

func appendRewriteMessages(messages []Message, raw interface{}) []Message {
	msgs, ok := raw.([]interface{})
	if !ok {
		return messages
	}

	var newMsgs []Message
	tempBytes, _ := json.Marshal(msgs)
	if err := json.Unmarshal(tempBytes, &newMsgs); err != nil {
		return messages
	}

	return append(newMsgs, messages...)
}

func appendRewriteTools(tools []interface{}, raw interface{}) []interface{} {
	if rewriteTools, ok := raw.([]interface{}); ok {
		return append(tools, rewriteTools...)
	}
	return tools
}

func rewriteDisablesStream(raw interface{}) bool {
	if s, ok := raw.(string); ok && s == "false" {
		return true
	}
	if b, ok := raw.(bool); ok && !b {
		return true
	}
	return false
}

func applyChatRewriteRules(chatReq *ChatRequest, rules RewriteRules) {
	for key, value := range rules {
		switch key {
		case "message":
			chatReq.Messages = appendRewriteMessages(chatReq.Messages, value)
		case "tools":
			chatReq.Tools = appendRewriteTools(chatReq.Tools, value)
		case "stream":
			if rewriteDisablesStream(value) {
				chatReq.Stream = false
			}
		case "model":
			if s, ok := value.(string); ok {
				chatReq.Model = s
			}
		default:
			log.Printf("\t\trewrite rule: %s = %v", key, value)
			chatReq.Other[key] = value
		}
	}
}

func normalizeResponseMessageContent(content interface{}) []interface{} {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		return []interface{}{
			map[string]interface{}{
				"type": "input_text",
				"text": v,
			},
		}
	case []interface{}:
		items := make([]interface{}, 0, len(v))
		for _, rawPart := range v {
			partMap, ok := rawPart.(map[string]interface{})
			if !ok {
				if text, ok := rawPart.(string); ok {
					items = append(items, map[string]interface{}{
						"type": "input_text",
						"text": text,
					})
				}
				continue
			}

			partType, _ := partMap["type"].(string)
			text, hasText := partMap["text"].(string)
			if strings.HasPrefix(partType, "input_") {
				items = append(items, partMap)
				continue
			}
			if partType == "text" && hasText {
				items = append(items, map[string]interface{}{
					"type": "input_text",
					"text": text,
				})
				continue
			}
			if hasText {
				items = append(items, map[string]interface{}{
					"type": "input_text",
					"text": text,
				})
				continue
			}
			items = append(items, partMap)
		}
		return items
	default:
		return nil
	}
}

func convertRewriteMessageToResponseItem(raw interface{}) interface{} {
	msgMap, ok := raw.(map[string]interface{})
	if !ok {
		return raw
	}

	role, _ := msgMap["role"].(string)
	item := map[string]interface{}{
		"type": "message",
		"role": role,
	}
	item["content"] = normalizeResponseMessageContent(msgMap["content"])
	return item
}

func normalizeResponsesInput(input interface{}) []interface{} {
	switch v := input.(type) {
	case nil:
		return nil
	case string:
		return []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": v,
					},
				},
			},
		}
	case []interface{}:
		return v
	default:
		return []interface{}{v}
	}
}

func prependResponsesRewriteMessages(input interface{}, raw interface{}) interface{} {
	msgs, ok := raw.([]interface{})
	if !ok {
		return input
	}

	rewriteItems := make([]interface{}, 0, len(msgs))
	for _, msg := range msgs {
		rewriteItems = append(rewriteItems, convertRewriteMessageToResponseItem(msg))
	}

	return append(rewriteItems, normalizeResponsesInput(input)...)
}

func applyResponsesRewriteRules(respReq *ResponsesRequest, rules RewriteRules) {
	for key, value := range rules {
		switch key {
		case "message":
			respReq.Input = prependResponsesRewriteMessages(respReq.Input, value)
		case "tools":
			respReq.Tools = appendRewriteTools(respReq.Tools, value)
		case "stream":
			if rewriteDisablesStream(value) {
				respReq.Stream = false
			}
		case "model":
			if s, ok := value.(string); ok {
				respReq.Model = s
			}
		default:
			log.Printf("\t\trewrite rule: %s = %v", key, value)
			respReq.Other[key] = value
		}
	}
}

// --- Comment Stripping ---

func getConfig() Config {
	v := configVal.Load()
	if v == nil {
		return Config{}
	}
	return v.(Config)
}

func loadConfig(filename string) (Config, error) {
	var cfg Config

	data, err := os.ReadFile(filename)
	if err != nil {
		return cfg, err
	}

	// Simple state machine to strip // comments while respecting strings
	var cleanLines []string
	lines := strings.Split(string(data), "\n")

	for _, line := range lines {
		var cleanLine []rune
		inString := false
		escape := false

		runes := []rune(line)
		skipRest := false

		for i := 0; i < len(runes); i++ {
			c := runes[i]

			if inString {
				if escape {
					escape = false
				} else if c == '\\' {
					escape = true
				} else if c == '"' {
					inString = false
				}
			} else {
				if c == '"' {
					inString = true
				} else if c == '/' && i+1 < len(runes) && runes[i+1] == '/' {
					skipRest = true
					break
				}
			}
			cleanLine = append(cleanLine, c)
		}

		if !skipRest {
			cleanLines = append(cleanLines, string(cleanLine))
		} else {
			cleanLines = append(cleanLines, string(cleanLine))
		}
	}
	cleanJson := strings.Join(cleanLines, "\n")

	err = json.Unmarshal([]byte(cleanJson), &cfg)
	if err != nil {
		return cfg, fmt.Errorf("json parse error: %v", err)
	}
	return cfg, nil
}

func updateConfig(filename string) error {
	cfg, err := loadConfig(filename)
	if err != nil {
		return err
	}

	configVal.Store(cfg)
	return nil
}

func getFileFingerprint(filename string) (string, error) {
	info, err := os.Stat(filename)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%d-%d", info.ModTime().UnixNano(), info.Size()), nil
}

func startConfigReloader(filename string, interval time.Duration) {
	go func() {
		lastFingerprint, err := getFileFingerprint(filename)
		if err != nil {
			log.Printf("Failed to read config metadata: %v", err)
		}

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			fingerprint, err := getFileFingerprint(filename)
			if err != nil {
				log.Printf("Failed to stat config file: %v", err)
				continue
			}

			if fingerprint == lastFingerprint {
				continue
			}

			if err := updateConfig(filename); err != nil {
				log.Printf("Config reload skipped due to invalid file: %v", err)
				lastFingerprint = fingerprint
				continue
			}

			lastFingerprint = fingerprint
			log.Printf("Config reloaded from %s", filename)
		}
	}()
}

// --- Handlers ---

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	log.Printf("[%s] Received chat completions request: %s", time.Now().Format(time.RFC3339), r.Header.Get("Content-Type"))

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}

	var chatReq ChatRequest
	if err := json.Unmarshal(bodyBytes, &chatReq); err != nil {
		log.Printf("Failed to parse JSON: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	log.Printf("Request model: %s", colorModel(chatReq.Model))

	cfg := getConfig()

	// 1. Auto-routing
	if rule, ok := cfg.RoutingRules[chatReq.Model]; ok {
		newModel := selectModel(chatReq.Messages, rule)
		if newModel != "" {
			log.Printf("Auto-routing: %s -> %s", colorModel(chatReq.Model), colorModel(newModel))
			chatReq.Model = newModel
		}
	}

	// 2. Rewrite rules
	if rules, ok := cfg.RewriteRules[chatReq.Model]; ok {
		log.Printf("Applying rewrite rules for %s", colorModel(chatReq.Model))
		applyChatRewriteRules(&chatReq, rules)
	}

	// Prepare forward request
	newBody, err := chatReq.MarshalJSON()
	if err != nil {
		http.Error(w, "Failed to remarshal body", http.StatusInternalServerError)
		return
	}

	log.Printf("Forwarding data to backend")

	forwardJSONRequest(w, r, cfg.BackendURL+"/v1/chat/completions", newBody)
}

func handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		handleProxy(w, r)
		return
	}

	log.Printf("[%s] Received responses request: %s", time.Now().Format(time.RFC3339), r.Header.Get("Content-Type"))

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read body", http.StatusBadRequest)
		return
	}

	var respReq ResponsesRequest
	if err := json.Unmarshal(bodyBytes, &respReq); err != nil {
		log.Printf("Failed to parse JSON: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	log.Printf("Request model: %s", colorModel(respReq.Model))

	cfg := getConfig()

	if rule, ok := cfg.RoutingRules[respReq.Model]; ok {
		newModel := selectResponsesModel(respReq, rule)
		if newModel != "" {
			log.Printf("Auto-routing: %s -> %s", colorModel(respReq.Model), colorModel(newModel))
			respReq.Model = newModel
		}
	}

	if rules, ok := cfg.RewriteRules[respReq.Model]; ok {
		log.Printf("Applying rewrite rules for %s", colorModel(respReq.Model))
		applyResponsesRewriteRules(&respReq, rules)
	}

	newBody, err := respReq.MarshalJSON()
	if err != nil {
		http.Error(w, "Failed to remarshal body", http.StatusInternalServerError)
		return
	}

	log.Printf("Forwarding data to backend")

	forwardJSONRequest(w, r, cfg.BackendURL+"/v1/responses", newBody)
}

func forwardJSONRequest(w http.ResponseWriter, r *http.Request, targetURL string, body []byte) {
	proxyReq, err := http.NewRequest("POST", targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, "Failed to create request", http.StatusInternalServerError)
		return
	}

	for k, v := range r.Header {
		if k != "Host" && k != "Content-Length" {
			proxyReq.Header[k] = v
		}
	}
	proxyReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{
		Timeout: 5 * time.Minute,
	}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("Proxy error: %v", err)
		http.Error(w, "Error forwarding request", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	for k, v := range resp.Header {
		if k != "Content-Encoding" && k != "Transfer-Encoding" {
			w.Header()[k] = v
		}
	}
	w.WriteHeader(resp.StatusCode)

	writer := io.Writer(w)
	if flusher, ok := w.(http.Flusher); ok {
		writer = flushWriter{ResponseWriter: w, Flusher: flusher}
	}

	if _, err := io.Copy(writer, resp.Body); err != nil {
		log.Printf("Error copying response: %v", err)
	}
}

// flushWriter forces a flush after each write to support streaming responses.
type flushWriter struct {
	http.ResponseWriter
	http.Flusher
}

func (fw flushWriter) Write(p []byte) (int, error) {
	n, err := fw.ResponseWriter.Write(p)
	if err == nil {
		fw.Flush()
	}
	return n, err
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	cfg := getConfig()
	targetURL := cfg.BackendURL + "/v1/models"
	proxyReq, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		http.Error(w, "Failed to create request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for k, v := range r.Header {
		if k != "Host" {
			proxyReq.Header[k] = v
		}
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(proxyReq)
	if err != nil {
		log.Printf("Error fetching models: %v", err)
		http.Error(w, "Error fetching models", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	var respData struct {
		Object string                   `json:"object"`
		Data   []map[string]interface{} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
		log.Printf("Invalid response from backend: %v", err)
		http.Error(w, "Invalid response from backend", http.StatusInternalServerError)
		return
	}

	// Logic to add virtual models
	existingIds := make(map[string]bool)
	for _, m := range respData.Data {
		if id, ok := m["id"].(string); ok {
			existingIds[id] = true
		}
	}

	created := int64(1677649963)
	if len(respData.Data) > 0 {
		if c, ok := respData.Data[0]["created"].(float64); ok {
			created = int64(c)
		}
	}

	addModel := func(name string, owner string) {
		if !existingIds[name] {
			respData.Data = append(respData.Data, map[string]interface{}{
				"id":       name,
				"object":   "model",
				"created":  created,
				"owned_by": owner,
			})
			existingIds[name] = true
		}
	}

	for name := range cfg.RoutingRules {
		addModel(name, "routing")
	}
	for name := range cfg.RewriteRules {
		addModel(name, "rewrite")
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(respData)
}

func handleProxy(w http.ResponseWriter, r *http.Request) {
	cfg := getConfig()
	targetURL, err := url.Parse(cfg.BackendURL)
	if err != nil {
		http.Error(w, "Invalid backend URL", http.StatusInternalServerError)
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Update the headers to allow for SSL redirection
	r.URL.Host = targetURL.Host
	r.URL.Scheme = targetURL.Scheme
	r.Header.Set("X-Forwarded-Host", r.Header.Get("Host"))
	r.Host = targetURL.Host

	proxy.ServeHTTP(w, r)
}

func main() {
	host := flag.String("host", "127.0.0.1", "Host to listen on")
	port := flag.Int("port", 3034, "Port to listen on")
	flag.Parse()

	if err := updateConfig("config.json"); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	startConfigReloader("config.json", 2*time.Second)

	// Initialize tokenizer
	var err error
	tokenizer, err = tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/v1/responses", handleResponses)
	http.HandleFunc("/v1/models", handleModels)

	// Catch-all for other routes
	http.HandleFunc("/", handleProxy)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("Server running at http://%s", addr)

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
