package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// --- Construction and Options ---

func TestNewGeminiEmbedder_Defaults(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	e, err := NewGeminiEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	if e.model != "gemini-embedding-001" {
		t.Errorf("expected default model gemini-embedding-001, got %s", e.model)
	}
	if e.dimensions != 3072 {
		t.Errorf("expected default dimensions 3072, got %d", e.dimensions)
	}
}

func TestNewGeminiEmbedder_WithOptions(t *testing.T) {
	e, err := NewGeminiEmbedder(
		WithGeminiKey("my-key"),
		WithGeminiModel("text-embedding-004"),
		WithGeminiDimensions(768),
		WithGeminiParallelism(8),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	if e.model != "text-embedding-004" {
		t.Errorf("expected model text-embedding-004, got %s", e.model)
	}
	if e.dimensions != 768 {
		t.Errorf("expected dimensions 768, got %d", e.dimensions)
	}
	if e.apiKey != "my-key" {
		t.Errorf("expected apiKey my-key, got %s", e.apiKey)
	}
	if e.parallelism != 8 {
		t.Errorf("expected parallelism 8, got %d", e.parallelism)
	}
}

func TestNewGeminiEmbedder_APIKeyFromEnv(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "env-gemini-key")

	e, err := NewGeminiEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	if e.apiKey != "env-gemini-key" {
		t.Errorf("expected apiKey from GEMINI_API_KEY env, got %s", e.apiKey)
	}
}

func TestNewGeminiEmbedder_APIKeyFallbackToGoogleKey(t *testing.T) {
	// Ensure GEMINI_API_KEY is not set
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "env-google-key")

	e, err := NewGeminiEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	if e.apiKey != "env-google-key" {
		t.Errorf("expected apiKey from GOOGLE_API_KEY env, got %s", e.apiKey)
	}
}

func TestNewGeminiEmbedder_NoAPIKey_Error(t *testing.T) {
	// Clear all possible API key sources
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_API_KEY", "")

	_, err := NewGeminiEmbedder()
	if err == nil {
		t.Fatal("expected error when no API key is provided")
	}
}

func TestNewGeminiEmbedder_InvalidDimensions_Error(t *testing.T) {
	tests := []struct {
		name       string
		dimensions int
	}{
		{"too small", 256},
		{"not a valid option", 1024},
		{"too large", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewGeminiEmbedder(
				WithGeminiKey("test-key"),
				WithGeminiDimensions(tt.dimensions),
			)
			if err == nil {
				t.Errorf("expected error for invalid dimensions %d", tt.dimensions)
			}
		})
	}
}

// --- Embed Method ---

func TestGeminiEmbed_ReturnsVector(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	e, err := NewGeminiEmbedder(WithGeminiDimensions(768))
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	ctx := context.Background()
	vec, err := e.Embed(ctx, "func main() { fmt.Println(\"hello\") }")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(vec) != 768 {
		t.Errorf("expected vector length 768, got %d", len(vec))
	}
}

func TestGeminiEmbed_UsesCodeRetrievalQuery(t *testing.T) {
	// This test verifies the task type sent to the API.
	// We use an HTTP test server to intercept and inspect the request.
	var capturedBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&capturedBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		// Return a valid Gemini-style embedding response
		resp := geminiEmbedResponse{
			Embedding: geminiEmbeddingValues{
				Values: make([]float32, 768),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, err := NewGeminiEmbedder(
		WithGeminiKey("test-key"),
		WithGeminiEndpoint(server.URL),
		WithGeminiDimensions(768),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	ctx := context.Background()
	_, err = e.Embed(ctx, "search query text")
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	// Verify the task type is set to CODE_RETRIEVAL_QUERY for search operations
	taskType, ok := capturedBody["taskType"].(string)
	if !ok {
		t.Fatal("expected taskType field in request body")
	}
	if taskType != "CODE_RETRIEVAL_QUERY" {
		t.Errorf("expected taskType CODE_RETRIEVAL_QUERY, got %s", taskType)
	}
}

// --- EmbedBatch Method ---

func TestGeminiEmbedBatch_ReturnsMultipleVectors(t *testing.T) {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}

	e, err := NewGeminiEmbedder(WithGeminiDimensions(768))
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	ctx := context.Background()
	texts := []string{
		"func hello() { return \"world\" }",
		"type User struct { Name string }",
		"func (u *User) Greet() string { return u.Name }",
	}

	vecs, err := e.EmbedBatch(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	if len(vecs) != len(texts) {
		t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
	}

	for i, vec := range vecs {
		if len(vec) != 768 {
			t.Errorf("vector %d: expected length 768, got %d", i, len(vec))
		}
	}
}

func TestGeminiEmbedBatch_UsesRetrievalDocument(t *testing.T) {
	var capturedBodies []map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		capturedBodies = append(capturedBodies, body)

		resp := geminiEmbedResponse{
			Embedding: geminiEmbeddingValues{
				Values: make([]float32, 768),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	e, err := NewGeminiEmbedder(
		WithGeminiKey("test-key"),
		WithGeminiEndpoint(server.URL),
		WithGeminiDimensions(768),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}
	defer e.Close()

	ctx := context.Background()
	texts := []string{"document content one", "document content two"}
	_, err = e.EmbedBatch(ctx, texts)
	if err != nil {
		t.Fatalf("EmbedBatch failed: %v", err)
	}

	// Verify all requests used RETRIEVAL_DOCUMENT task type
	for i, body := range capturedBodies {
		taskType, ok := body["taskType"].(string)
		if !ok {
			t.Errorf("request %d: expected taskType field", i)
			continue
		}
		if taskType != "RETRIEVAL_DOCUMENT" {
			t.Errorf("request %d: expected taskType RETRIEVAL_DOCUMENT, got %s", i, taskType)
		}
	}
}

// --- Dimensions ---

func TestGeminiDimensions_ReturnsConfigured(t *testing.T) {
	e, err := NewGeminiEmbedder(
		WithGeminiKey("test-key"),
		WithGeminiDimensions(1536),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	if e.Dimensions() != 1536 {
		t.Errorf("expected Dimensions() to return 1536, got %d", e.Dimensions())
	}
}

func TestGeminiDimensions_Default3072(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	e, err := NewGeminiEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	if e.Dimensions() != 3072 {
		t.Errorf("expected default Dimensions() to return 3072, got %d", e.Dimensions())
	}
}

// --- Close ---

func TestGeminiClose_NoError(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "test-key")

	e, err := NewGeminiEmbedder()
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	if err := e.Close(); err != nil {
		t.Errorf("expected Close() to return nil, got %v", err)
	}
}

// --- Response types for test server mocking ---

// geminiEmbedResponse mirrors the expected Gemini API embedding response structure.
// This is defined here for test mock servers; the real implementation will define its own.
type geminiEmbedResponse struct {
	Embedding geminiEmbeddingValues `json:"embedding"`
}

type geminiEmbeddingValues struct {
	Values []float32 `json:"values"`
}
