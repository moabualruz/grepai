package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync/atomic"
	"time"

	"golang.org/x/sync/errgroup"
)

const (
	defaultGeminiEndpoint   = "https://generativelanguage.googleapis.com"
	defaultGeminiModel      = "gemini-embedding-001"
	defaultGeminiDimensions = 3072
)

// validGeminiDimensions contains the allowed dimension values for Gemini embedding models.
var validGeminiDimensions = map[int]bool{
	768:  true,
	1536: true,
	3072: true,
}

// GeminiEmbedder implements the Embedder and BatchEmbedder interfaces using the
// Google Gemini embedding API via direct HTTP calls.
type GeminiEmbedder struct {
	endpoint    string
	model       string
	apiKey      string
	dimensions  int
	parallelism int
	retryPolicy RetryPolicy
	rateLimiter *AdaptiveRateLimiter
	client      *http.Client
}

// geminiEmbedRequest is the request body for the Gemini embedContent API.
type geminiEmbedRequest struct {
	Model                string        `json:"model"`
	Content              geminiContent `json:"content"`
	TaskType             string        `json:"taskType"`
	OutputDimensionality *int          `json:"outputDimensionality,omitempty"`
}

type geminiContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

// GeminiOption is a functional option for configuring GeminiEmbedder.
type GeminiOption func(*GeminiEmbedder)

// WithGeminiModel sets the embedding model name.
func WithGeminiModel(model string) GeminiOption {
	return func(e *GeminiEmbedder) {
		if model != "" {
			e.model = model
		}
	}
}

// WithGeminiKey sets the API key for authentication.
func WithGeminiKey(key string) GeminiOption {
	return func(e *GeminiEmbedder) {
		e.apiKey = key
	}
}

// WithGeminiDimensions sets the output embedding dimensions.
func WithGeminiDimensions(dimensions int) GeminiOption {
	return func(e *GeminiEmbedder) {
		e.dimensions = dimensions
	}
}

// WithGeminiEndpoint sets a custom API endpoint (useful for testing).
func WithGeminiEndpoint(endpoint string) GeminiOption {
	return func(e *GeminiEmbedder) {
		if endpoint != "" {
			e.endpoint = endpoint
		}
	}
}

// WithGeminiParallelism sets the number of parallel workers for batch embedding.
func WithGeminiParallelism(parallelism int) GeminiOption {
	return func(e *GeminiEmbedder) {
		if parallelism > 0 {
			e.parallelism = parallelism
		}
	}
}

// NewGeminiEmbedder creates a new GeminiEmbedder with the given options.
func NewGeminiEmbedder(opts ...GeminiOption) (*GeminiEmbedder, error) {
	e := &GeminiEmbedder{
		endpoint:    defaultGeminiEndpoint,
		model:       defaultGeminiModel,
		dimensions:  defaultGeminiDimensions,
		parallelism: defaultParallelism,
		retryPolicy: DefaultRetryPolicy(),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}

	for _, opt := range opts {
		opt(e)
	}

	// API key resolution: option > GEMINI_API_KEY env > GOOGLE_API_KEY env
	if e.apiKey == "" {
		e.apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if e.apiKey == "" {
		e.apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if e.apiKey == "" {
		return nil, fmt.Errorf("Gemini API key not set (use GEMINI_API_KEY or GOOGLE_API_KEY environment variable)")
	}

	// Validate dimensions
	if !validGeminiDimensions[e.dimensions] {
		return nil, fmt.Errorf("invalid Gemini embedding dimensions %d: must be 768, 1536, or 3072", e.dimensions)
	}

	// Initialize adaptive rate limiter
	e.rateLimiter = NewAdaptiveRateLimiter(e.parallelism)

	return e, nil
}

// embedSingle sends a single embedding request to the Gemini API.
func (e *GeminiEmbedder) embedSingle(ctx context.Context, text string, taskType string) ([]float32, error) {
	reqBody := geminiEmbedRequest{
		Model: e.model,
		Content: geminiContent{
			Parts: []geminiPart{{Text: text}},
		},
		TaskType: taskType,
	}

	if e.dimensions != defaultGeminiDimensions {
		dim := e.dimensions
		reqBody.OutputDimensionality = &dim
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:embedContent", e.endpoint, e.model)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request to Gemini: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Embedding.Values, nil
}

// Embed converts a single text into a vector embedding using CODE_RETRIEVAL_QUERY task type.
func (e *GeminiEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return e.embedSingle(ctx, text, "CODE_RETRIEVAL_QUERY")
}

// EmbedBatch converts multiple texts into vector embeddings using RETRIEVAL_DOCUMENT task type.
// Each text is embedded individually since the Gemini API handles one content per request.
func (e *GeminiEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		vec, err := e.embedSingle(ctx, text, "RETRIEVAL_DOCUMENT")
		if err != nil {
			return nil, fmt.Errorf("failed to embed text %d: %w", i, err)
		}
		embeddings[i] = vec
	}

	return embeddings, nil
}

// EmbedBatches implements the BatchEmbedder interface.
// It processes multiple batches concurrently using a bounded worker pool.
func (e *GeminiEmbedder) EmbedBatches(ctx context.Context, batches []Batch, progress BatchProgress) ([]BatchResult, error) {
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
	g.SetLimit(e.rateLimiter.CurrentWorkers())

	for i := range batches {
		batch := batches[i]
		g.Go(func() error {
			contents := batch.Contents()
			embeddings, err := e.EmbedBatch(ctx, contents)
			if err != nil {
				return err
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

// Dimensions returns the configured embedding dimensions.
func (e *GeminiEmbedder) Dimensions() int {
	return e.dimensions
}

// Close releases any resources held by the embedder.
func (e *GeminiEmbedder) Close() error {
	return nil
}
