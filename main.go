package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/fatih/color"
	"github.com/pkoukk/tiktoken-go"
)

// Configuration structures
type Config struct {
	BackendURL   string                 `json:"BACKEND_URL"`
	RewriteRules map[string]RewriteRule `json:"REWRITE_RULES"`
	RoutingRules map[string]RoutingRule `json:"ROUTING_RULES"`
}

type RewriteRule struct {
	Message []ChatMessage          `json:"message,omitempty"`
	Tools   []interface{}          `json:"tools,omitempty"`
	Stream  *string                `json:"stream,omitempty"`
	Model   string                 `json:"model,omitempty"`
	Extra   map[string]interface{} // For any additional fields
}

type RoutingRule struct {
	Models    map[string]string `json:"models"`
	Threshold float64           `json:"threshold"`
}

type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
	Name    string      `json:"name,omitempty"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      *bool         `json:"stream,omitempty"`
	Tools       []interface{} `json:"tools,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	// Add other fields as needed
	Extra map[string]interface{} `json:"-"`
}

type ModelResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// Global variables
var (
	config    Config
	redColor  = color.New(color.FgRed)
	tokenizer *tiktoken.Tiktoken
)

func init() {
	// Initialize tokenizer
	var err error
	tokenizer, err = tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Printf("Failed to initialize tokenizer: %v", err)
	}
}

// Token counting functions
func getTokenCount(text string) int {
	if text == "" {
		return 0
	}

	if tokenizer == nil {
		return 0
	}

	tokens := tokenizer.Encode(text, nil, nil)
	return len(tokens)
}

func getMessagesTokenCount(messages []ChatMessage) int {
	if len(messages) == 0 {
		return 0
	}

	totalTokens := 2      // Base format overhead
	perMessageTokens := 4 // Role marker tokens

	for _, msg := range messages {
		totalTokens += perMessageTokens

		switch content := msg.Content.(type) {
		case string:
			totalTokens += getTokenCount(content)
		case []interface{}:
			combinedText := ""
			for _, part := range content {
				if partMap, ok := part.(map[string]interface{}); ok {
					if text, ok := partMap["text"].(string); ok {
						combinedText += text
					}
				}
			}
			totalTokens += getTokenCount(combinedText)
		}
	}

	return totalTokens
}

func selectModel(messages []ChatMessage, routing RoutingRule) string {
	if len(routing.Models) == 0 || routing.Threshold == 0 {
		log.Printf("Invalid model routing configuration")
		return ""
	}

	tokenCount := getMessagesTokenCount(messages)

	// Sort context lengths
	var contextLengths []int
	for lengthStr := range routing.Models {
		if length, err := strconv.Atoi(lengthStr); err == nil {
			contextLengths = append(contextLengths, length)
		}
	}

	// Simple bubble sort
	for i := 0; i < len(contextLengths)-1; i++ {
		for j := 0; j < len(contextLengths)-i-1; j++ {
			if contextLengths[j] > contextLengths[j+1] {
				contextLengths[j], contextLengths[j+1] = contextLengths[j+1], contextLengths[j]
			}
		}
	}

	for _, length := range contextLengths {
		if float64(tokenCount) <= float64(length)*routing.Threshold {
			return routing.Models[strconv.Itoa(length)]
		}
	}

	// Return model for maximum context length
	maxLength := contextLengths[len(contextLengths)-1]
	return routing.Models[strconv.Itoa(maxLength)]
}

// HTTP handlers
func chatCompletionsHandler(w http.ResponseWriter, r *http.Request) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	log.Printf("[%s] Received chat completions request: %s", timestamp, r.Header.Get("Content-Type"))

	// Parse request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		log.Printf("[%s] Failed to read request body: %v", timestamp, err)
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var data ChatCompletionRequest
	if err := json.Unmarshal(body, &data); err != nil {
		log.Printf("[%s] Failed to parse request body as JSON: %v", timestamp, err)
		http.Error(w, "Invalid JSON in request body", http.StatusBadRequest)
		return
	}

	log.Printf("[%s] Request model: %s", timestamp, redColor.Sprint(data.Model))

	// Handle automatic model routing
	if routingRule, exists := config.RoutingRules[data.Model]; exists {
		selectedModel := selectModel(data.Messages, routingRule)
		if selectedModel != "" {
			log.Printf("[%s] Auto-routing: %s -> %s", timestamp, redColor.Sprint(data.Model), redColor.Sprint(selectedModel))
			data.Model = selectedModel
		}
	}

	// Apply rewrite rules
	if rewriteRule, exists := config.RewriteRules[data.Model]; exists {
		log.Printf("[%s] Applying rewrite rules for %s", timestamp, redColor.Sprint(data.Model))

		if len(rewriteRule.Message) > 0 {
			data.Messages = append(rewriteRule.Message, data.Messages...)
		}

		if len(rewriteRule.Tools) > 0 {
			data.Tools = append(data.Tools, rewriteRule.Tools...)
		}

		if rewriteRule.Stream != nil && *rewriteRule.Stream == "false" {
			streamFalse := false
			data.Stream = &streamFalse
		}

		if rewriteRule.Model != "" {
			log.Printf("[%s] \t\trewrite rule: model = %s", timestamp, rewriteRule.Model)
			data.Model = rewriteRule.Model
		}

		// Apply extra fields
		for key, value := range rewriteRule.Extra {
			log.Printf("[%s] \t\trewrite rule: %s = %v", timestamp, key, value)
			// This would need custom marshaling to handle properly
		}
	}

	// Forward request to backend
	jsonData, err := json.Marshal(data)
	if err != nil {
		log.Printf("[%s] Failed to marshal request data: %v", timestamp, err)
		http.Error(w, "Failed to process request", http.StatusInternalServerError)
		return
	}

	log.Printf("[%s] Forwarding data: %s", timestamp, string(jsonData))

	// Create request to backend
	req, err := http.NewRequest("POST", config.BackendURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("[%s] Failed to create backend request: %v", timestamp, err)
		http.Error(w, "Failed to create backend request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		if key != "Host" && key != "Content-Length" {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
	}
	req.Header.Set("Content-Type", "application/json")

	// Make request to backend
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[%s] Error forwarding request to backend: %v", timestamp, err)
		http.Error(w, "Error forwarding request to backend", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for key, values := range resp.Header {
		if key != "Content-Encoding" && key != "Transfer-Encoding" && key != "Connection" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}

	w.WriteHeader(resp.StatusCode)

	// Handle streaming response
	if data.Stream != nil && *data.Stream {
		// Stream response
		io.Copy(w, resp.Body)
	} else {
		// Non-streaming response
		io.Copy(w, resp.Body)
	}
}

func modelsHandler(w http.ResponseWriter, r *http.Request) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")

	// Create request to backend
	req, err := http.NewRequest("GET", config.BackendURL+"/v1/models", nil)
	if err != nil {
		log.Printf("[%s] Failed to create backend request: %v", timestamp, err)
		http.Error(w, "Failed to create backend request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		if key != "Host" {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
	}

	// Make request to backend
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[%s] Error fetching models from backend: %v", timestamp, err)
		http.Error(w, "Error connecting to backend", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		log.Printf("[%s] Backend returned error: %d", timestamp, resp.StatusCode)
		w.WriteHeader(resp.StatusCode)
		io.Copy(w, resp.Body)
		return
	}

	var modelResponse ModelResponse
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[%s] Failed to read backend response: %v", timestamp, err)
		http.Error(w, "Failed to read backend response", http.StatusInternalServerError)
		return
	}

	if err := json.Unmarshal(responseBody, &modelResponse); err != nil {
		log.Printf("[%s] Invalid response format from backend: %v", timestamp, err)
		http.Error(w, "Invalid response from backend", http.StatusInternalServerError)
		return
	}

	// Merge model list
	modelList := modelResponse.Data
	created := int64(1677649963)
	if len(modelList) > 0 {
		created = modelList[0].Created
	}

	// Add models from routing rules
	for modelName := range config.RoutingRules {
		found := false
		for _, model := range modelList {
			if model.ID == modelName {
				found = true
				break
			}
		}
		if !found {
			modelList = append(modelList, Model{
				ID:      modelName,
				Object:  "model",
				Created: created,
				OwnedBy: "routing",
			})
		}
	}

	// Add models from rewrite rules
	for modelName := range config.RewriteRules {
		found := false
		for _, model := range modelList {
			if model.ID == modelName {
				found = true
				break
			}
		}
		if !found {
			modelList = append(modelList, Model{
				ID:      modelName,
				Object:  "model",
				Created: created,
				OwnedBy: "rewrite",
			})
		}
	}

	modelResponse.Data = modelList

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(modelResponse)
}

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	log.Printf("[%s] Proxying %s request to: %s", timestamp, r.Method, r.URL.Path)

	// Read request body
	var bodyBytes []byte
	if r.Body != nil {
		var err error
		bodyBytes, err = io.ReadAll(r.Body)
		if err != nil {
			log.Printf("[%s] Failed to read request body: %v", timestamp, err)
			http.Error(w, "Failed to read request body", http.StatusInternalServerError)
			return
		}
		r.Body.Close()
	}

	// Create request to backend
	targetURL := config.BackendURL + r.URL.String()
	log.Printf("[%s] Forwarding to: %s", timestamp, targetURL)

	req, err := http.NewRequest(r.Method, targetURL, bytes.NewBuffer(bodyBytes))
	if err != nil {
		log.Printf("[%s] Failed to create backend request: %v", timestamp, err)
		http.Error(w, "Failed to create backend request", http.StatusInternalServerError)
		return
	}

	// Copy headers
	for key, values := range r.Header {
		if key != "Host" && key != "Content-Length" {
			for _, value := range values {
				req.Header.Add(key, value)
			}
		}
	}

	// Make request to backend
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("[%s] Proxy error: %v", timestamp, err)
		http.Error(w, "Error proxying request to backend", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	log.Printf("[%s] Backend response status: %d", timestamp, resp.StatusCode)

	// Copy response headers
	for key, values := range resp.Header {
		if key != "Content-Encoding" && key != "Transfer-Encoding" && key != "Connection" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}

	w.WriteHeader(resp.StatusCode)

	// Handle response body
	contentType := resp.Header.Get("Content-Type")
	if strings.Contains(contentType, "text/event-stream") {
		// Stream response
		log.Printf("[%s] Piping streaming response for Content-Type: %s", timestamp, contentType)
		io.Copy(w, resp.Body)
	} else {
		// Buffer response
		io.Copy(w, resp.Body)
	}
}

func loadConfig() error {
	file, err := os.Open("config.json")
	if err != nil {
		return fmt.Errorf("failed to open config file: %v", err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("failed to read config file: %v", err)
	}

	// Remove comments (basic implementation)
	lines := strings.Split(string(data), "\n")
	var cleanLines []string
	for _, line := range lines {
		if trimmed := strings.TrimSpace(line); !strings.HasPrefix(trimmed, "//") {
			cleanLines = append(cleanLines, line)
		}
	}
	cleanData := strings.Join(cleanLines, "\n")

	if err := json.Unmarshal([]byte(cleanData), &config); err != nil {
		return fmt.Errorf("failed to parse config JSON: %v", err)
	}

	return nil
}

func main() {
	var host = flag.String("host", "127.0.0.1", "Host to bind to")
	var port = flag.Int("port", 3034, "Port to bind to")
	flag.Parse()

	// Load configuration
	if err := loadConfig(); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Setup routes
	http.HandleFunc("/v1/chat/completions", chatCompletionsHandler)
	http.HandleFunc("/v1/models", modelsHandler)
	http.HandleFunc("/", proxyHandler)

	// Create server
	addr := fmt.Sprintf("%s:%d", *host, *port)
	server := &http.Server{
		Addr:         addr,
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 5 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in goroutine
	go func() {
		timestamp := time.Now().Format("2006-01-02 15:04:05")
		log.Printf("[%s] Server running at http://%s", timestamp, addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	timestamp := time.Now().Format("2006-01-02 15:04:05")
	log.Printf("[%s] Received shutdown signal, shutting down", timestamp)

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Printf("Server forced to shutdown: %v", err)
	} else {
		log.Printf("[%s] Server shutting down, resources freed", timestamp)
	}
}
