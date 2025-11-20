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
	BackendURL   string                 `json:"BACKEND_URL"`
	RoutingRules map[string]RoutingRule `json:"ROUTING_RULES"`
	RewriteRules map[string]RewriteRules `json:"REWRITE_RULES"`
}

// --- Global Variables ---

var (
	config    Config
	tokenizer *tiktoken.Tiktoken
)

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
	Model    string      `json:"model"`
	Messages []Message   `json:"messages"`
	Stream   bool        `json:"stream"`
	Tools    []interface{} `json:"tools,omitempty"`
	// Capture other fields to preserve them
	Other map[string]interface{} `json:"-"`
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
	type Alias ChatRequest
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

// --- Logic Functions ---

func selectModel(messages []Message, rule RoutingRule) string {
	if len(rule.Models) == 0 || rule.Threshold == 0 {
		return ""
	}

	tokenCount := getMessagesTokenCount(messages)

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

// --- Comment Stripping ---

func loadConfig(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
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

	err = json.Unmarshal([]byte(cleanJson), &config)
	if err != nil {
		return fmt.Errorf("json parse error: %v", err)
	}
	return nil
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

	log.Printf("Request model: %s", chatReq.Model)

	// 1. Auto-routing
	if rule, ok := config.RoutingRules[chatReq.Model]; ok {
		newModel := selectModel(chatReq.Messages, rule)
		if newModel != "" {
			log.Printf("Auto-routing: %s -> %s", chatReq.Model, newModel)
			chatReq.Model = newModel
		}
	}

	// 2. Rewrite rules
	if rules, ok := config.RewriteRules[chatReq.Model]; ok {
		log.Printf("Applying rewrite rules for %s", chatReq.Model)
		for key, value := range rules {
			switch key {
			case "message":
				// Prepend messages
				// value should be []interface{} (array of messages)
				if msgs, ok := value.([]interface{}); ok {
					// Convert interface{} to Message structs
					var newMsgs []Message
					tempBytes, _ := json.Marshal(msgs)
					json.Unmarshal(tempBytes, &newMsgs)
					
					chatReq.Messages = append(newMsgs, chatReq.Messages...)
				}
			case "tools":
				// Append tools
				if tools, ok := value.([]interface{}); ok {
					chatReq.Tools = append(chatReq.Tools, tools...)
				}
			case "stream":
				if s, ok := value.(string); ok && s == "false" {
					chatReq.Stream = false
				} else if b, ok := value.(bool); ok && !b {
					chatReq.Stream = false
				}
			case "model":
				if s, ok := value.(string); ok {
					chatReq.Model = s
				}
			default:
				// Generic rewrite for other fields
				log.Printf("\t\trewrite rule: %s = %v", key, value)
				chatReq.Other[key] = value
			}
		}
	}

	// Prepare forward request
	newBody, err := chatReq.MarshalJSON()
	if err != nil {
		http.Error(w, "Failed to remarshal body", http.StatusInternalServerError)
		return
	}

	log.Printf("Forwarding data to backend")

	targetURL := config.BackendURL + "/v1/chat/completions"
	proxyReq, err := http.NewRequest("POST", targetURL, bytes.NewReader(newBody))
	if err != nil {
		http.Error(w, "Failed to create request", http.StatusInternalServerError)
		return
	}

	// Copy headers
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

	// Copy response headers
	for k, v := range resp.Header {
		if k != "Content-Encoding" && k != "Transfer-Encoding" {
			w.Header()[k] = v
		}
	}
	w.WriteHeader(resp.StatusCode)

	// Stream or copy body
	// For simplicity and efficiency in Go, we can just copy the stream directly.
	// This handles both streaming and non-streaming seamlessly.
	_, err = io.Copy(w, resp.Body)
	if err != nil {
		log.Printf("Error copying response: %v", err)
	}
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	targetURL := config.BackendURL + "/v1/models"
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

	for name := range config.RoutingRules {
		addModel(name, "routing")
	}
	for name := range config.RewriteRules {
		addModel(name, "rewrite")
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(respData)
}

func handleProxy(w http.ResponseWriter, r *http.Request) {
	targetURL, _ := url.Parse(config.BackendURL)
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

	if err := loadConfig("config.json"); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize tokenizer
	var err error
	tokenizer, err = tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Failed to initialize tokenizer: %v", err)
	}

	http.HandleFunc("/v1/chat/completions", handleChatCompletions)
	http.HandleFunc("/v1/models", handleModels)
	
	// Catch-all for other routes
	http.HandleFunc("/", handleProxy)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	log.Printf("Server running at http://%s", addr)
	
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
