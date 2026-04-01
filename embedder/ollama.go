package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

const (
	defaultOllamaEndpoint    = "http://localhost:11434"
	defaultOllamaModel       = "nomic-embed-text"
	nomicEmbedDimensions     = 768
	defaultOllamaParallelism = 2
	defaultOllamaBatchSize   = 100
)

type OllamaEmbedder struct {
	endpoint    string
	model       string
	dimensions  int
	parallelism int
	client      *http.Client
}

// ollamaEmbedRequest uses the /api/embed endpoint which supports array input.
type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// ollamaEmbedResponse matches the /api/embed response format.
type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

type OllamaOption func(*OllamaEmbedder)

func WithOllamaEndpoint(endpoint string) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.endpoint = endpoint
	}
}

func WithOllamaModel(model string) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.model = model
	}
}
func WithOllamaDimensions(dimensions int) OllamaOption {
	return func(e *OllamaEmbedder) {
		e.dimensions = dimensions
	}
}

func WithOllamaParallelism(parallelism int) OllamaOption {
	return func(e *OllamaEmbedder) {
		if parallelism > 0 {
			e.parallelism = parallelism
		}
	}
}

func NewOllamaEmbedder(opts ...OllamaOption) *OllamaEmbedder {
	e := &OllamaEmbedder{
		endpoint:    defaultOllamaEndpoint,
		model:       defaultOllamaModel,
		dimensions:  nomicEmbedDimensions,
		parallelism: defaultOllamaParallelism,
		client: &http.Client{
			Timeout: 120 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("Ollama returned empty embedding")
	}
	return embeddings[0], nil
}

// EmbedBatch sends multiple texts in a single /api/embed call.
func (e *OllamaEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := ollamaEmbedRequest{
		Model: e.model,
		Input: texts,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/api/embed", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)

		// Check for context length error
		if resp.StatusCode == http.StatusInternalServerError &&
			strings.Contains(bodyStr, "exceeds the context length") {
			estimatedTokens := 0
			for _, t := range texts {
				estimatedTokens += len(t) / 4
			}
			return nil, NewContextLengthError(0, estimatedTokens, 0, bodyStr)
		}

		return nil, fmt.Errorf("Ollama returned status %d: %s", resp.StatusCode, bodyStr)
	}

	var result ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("Ollama returned %d embeddings for %d inputs", len(result.Embeddings), len(texts))
	}

	return result.Embeddings, nil
}

// EmbedBatches implements the BatchEmbedder interface.
// It processes multiple batches concurrently using a bounded worker pool.
func (e *OllamaEmbedder) EmbedBatches(ctx context.Context, batches []Batch, progress BatchProgress) ([]BatchResult, error) {
	if len(batches) == 0 {
		return nil, nil
	}

	totalChunks := 0
	for _, batch := range batches {
		totalChunks += batch.Size()
	}

	var completedChunks atomic.Int64
	results := make([]BatchResult, len(batches))
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(e.parallelism)

	for i := range batches {
		batch := batches[i]
		g.Go(func() error {
			contents := batch.Contents()
			embeddings, err := e.EmbedBatch(ctx, contents)
			if err != nil {
				return fmt.Errorf("batch %d failed: %w", batch.Index, err)
			}
			results[batch.Index] = BatchResult{
				BatchIndex: batch.Index,
				Embeddings: embeddings,
			}

			newCompleted := completedChunks.Add(int64(batch.Size()))
			if progress != nil {
				progress(batch.Index, len(batches), int(newCompleted), totalChunks, false, 0, 0)
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}

	return results, nil
}

func (e *OllamaEmbedder) Dimensions() int {
	return e.dimensions
}

func (e *OllamaEmbedder) Close() error {
	return nil
}

// Ping checks if Ollama is reachable
func (e *OllamaEmbedder) Ping(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/tags", e.endpoint)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to reach Ollama at %s: %w", e.endpoint, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Ollama returned status %d", resp.StatusCode)
	}

	return nil
}

// Compile-time interface check
var _ BatchEmbedder = (*OllamaEmbedder)(nil)
