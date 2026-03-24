package embedder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestGeminiEmbedBatches_ProcessesAllBatches(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

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
		WithGeminiParallelism(2),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	batches := []Batch{
		{
			Index: 0,
			Entries: []BatchEntry{
				{FileIndex: 0, ChunkIndex: 0, Content: "func main() {}"},
				{FileIndex: 0, ChunkIndex: 1, Content: "func hello() {}"},
			},
		},
		{
			Index: 1,
			Entries: []BatchEntry{
				{FileIndex: 1, ChunkIndex: 0, Content: "type Foo struct{}"},
			},
		},
		{
			Index: 2,
			Entries: []BatchEntry{
				{FileIndex: 2, ChunkIndex: 0, Content: "var x = 42"},
				{FileIndex: 2, ChunkIndex: 1, Content: "var y = 99"},
				{FileIndex: 2, ChunkIndex: 2, Content: "var z = 0"},
			},
		},
	}

	ctx := context.Background()
	results, err := e.EmbedBatches(ctx, batches, nil)
	if err != nil {
		t.Fatalf("EmbedBatches failed: %v", err)
	}

	if len(results) != len(batches) {
		t.Errorf("expected %d results, got %d", len(batches), len(results))
	}

	// Verify each result has the correct batch index and embedding count
	for _, result := range results {
		expectedCount := len(batches[result.BatchIndex].Entries)
		if len(result.Embeddings) != expectedCount {
			t.Errorf("batch %d: expected %d embeddings, got %d",
				result.BatchIndex, expectedCount, len(result.Embeddings))
		}
	}
}

func TestGeminiEmbedBatches_ProgressCallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		WithGeminiParallelism(1), // Sequential for predictable progress
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	batches := make([]Batch, 3)
	for i := range batches {
		batches[i] = Batch{
			Index:   i,
			Entries: []BatchEntry{{FileIndex: i, ChunkIndex: 0, Content: "test content"}},
		}
	}

	type progressInfo struct {
		batchIndex      int
		totalBatches    int
		completedChunks int
		totalChunks     int
		retrying        bool
		attempt         int
	}

	var progressCalls []progressInfo
	var mu sync.Mutex
	progress := func(batchIndex, totalBatches, completedChunks, totalChunks int, retrying bool, attempt int, statusCode int) {
		mu.Lock()
		progressCalls = append(progressCalls, progressInfo{
			batchIndex:      batchIndex,
			totalBatches:    totalBatches,
			completedChunks: completedChunks,
			totalChunks:     totalChunks,
			retrying:        retrying,
			attempt:         attempt,
		})
		mu.Unlock()
	}

	ctx := context.Background()
	_, err = e.EmbedBatches(ctx, batches, progress)
	if err != nil {
		t.Fatalf("EmbedBatches failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	// Should have 3 progress calls (one per batch completion)
	if len(progressCalls) != 3 {
		t.Errorf("expected 3 progress calls, got %d", len(progressCalls))
	}

	// All should report totalBatches = 3 and not retrying
	for _, call := range progressCalls {
		if call.totalBatches != 3 {
			t.Errorf("expected totalBatches=3, got %d", call.totalBatches)
		}
		if call.retrying {
			t.Error("unexpected retry flag")
		}
	}
}

func TestGeminiEmbedBatches_RespectsParallelism(t *testing.T) {
	var (
		maxConcurrent int32
		current       int32
		mu            sync.Mutex
		requestCount  int32
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c := atomic.AddInt32(&current, 1)
		defer atomic.AddInt32(&current, -1)

		mu.Lock()
		if c > maxConcurrent {
			maxConcurrent = c
		}
		mu.Unlock()

		atomic.AddInt32(&requestCount, 1)

		// Simulate processing time to overlap requests
		time.Sleep(50 * time.Millisecond)

		resp := geminiEmbedResponse{
			Embedding: geminiEmbeddingValues{
				Values: make([]float32, 768),
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	parallelism := 2
	e, err := NewGeminiEmbedder(
		WithGeminiKey("test-key"),
		WithGeminiEndpoint(server.URL),
		WithGeminiDimensions(768),
		WithGeminiParallelism(parallelism),
	)
	if err != nil {
		t.Fatalf("failed to create embedder: %v", err)
	}

	// Create 6 batches to test parallelism limit
	batches := make([]Batch, 6)
	for i := range batches {
		batches[i] = Batch{
			Index: i,
			Entries: []BatchEntry{
				{FileIndex: i, ChunkIndex: 0, Content: "test content"},
			},
		}
	}

	ctx := context.Background()
	results, err := e.EmbedBatches(ctx, batches, nil)
	if err != nil {
		t.Fatalf("EmbedBatches failed: %v", err)
	}

	// Verify all batches processed
	if len(results) != len(batches) {
		t.Errorf("expected %d results, got %d", len(batches), len(results))
	}

	// Verify parallelism was respected
	if maxConcurrent > int32(parallelism) {
		t.Errorf("max concurrent %d exceeded parallelism limit %d", maxConcurrent, parallelism)
	}

	// Verify all requests were made
	if atomic.LoadInt32(&requestCount) != int32(len(batches)) {
		t.Errorf("expected %d requests, got %d", len(batches), requestCount)
	}
}
