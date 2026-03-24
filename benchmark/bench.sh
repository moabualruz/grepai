#!/bin/bash
# Universal embedding model benchmark
# Usage: bash bench.sh <config.json> [project_path]
#
# Config JSON: array of {tag, model_id, lms_path, dims, ctx}
# Defaults: project=crispy-tivi, chunk=512, overlap=50

set -uo pipefail

CONFIG_REL="${1:?Usage: bash bench.sh <config.json> [project_path]}"
PROJECT="${2:-/d/work/crispy-tivi}"
# Resolve to absolute path, then convert to Windows format for Python
CONFIG_UNIX="$(cd "$(dirname "$CONFIG_REL")" && pwd)/$(basename "$CONFIG_REL")"
# Convert /d/path to D:/path for Python compatibility
CONFIG_WIN=$(echo "$CONFIG_UNIX" | sed 's|^/\([a-zA-Z]\)/|\U\1:/|')
CONFIG="$CONFIG_UNIX"
GREPAI="C:/Users/mkh/AppData/Local/Programs/grepai/grepai.exe"
ENDPOINT="http://127.0.0.1:12345"
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RESULTS_DIR="/d/work/gemgrep/benchmark-results"
MODELS_DIR="F:/work/models"

QUERIES=(
  "how does the video player handle buffering"
  "EPG channel list rendering"
  "authentication and login flow"
  "function that parses m3u playlist"
  "error handling for network requests"
  "state management for favorites"
  "how are TV channels sorted and filtered"
  "database schema for storing channels"
  "widget that displays program guide"
  "HTTP client configuration and interceptors"
)

mkdir -p "$RESULTS_DIR"

# ============================================================
# Helpers
# ============================================================

log() { echo "[$(date +%H:%M:%S)] $*"; }

nuclear_clean() {
  log "Killing all grepai processes..."
  taskkill //F //IM grepai.exe 2>/dev/null || true
  sleep 3
  # Verify no grepai running
  if tasklist 2>/dev/null | grep -qi grepai; then
    log "ERROR: grepai still running after kill"
    taskkill //F //IM grepai.exe 2>/dev/null || true
    sleep 5
  fi

  log "Removing index..."
  rm -rf "$PROJECT/.grepai" 2>/dev/null || true
  sleep 1
  if [[ -d "$PROJECT/.grepai" ]]; then
    log "ERROR: .grepai directory still exists, retrying..."
    rm -rf "$PROJECT/.grepai"
    sleep 2
  fi

  log "Unloading all LM Studio models..."
  lms unload --all 2>/dev/null || true
  sleep 3

  # Verify nothing loaded — retry until clean
  for attempt in 1 2 3; do
    LOADED=$(lms ps 2>&1 | grep -c "IDLE\|PROCESSING" || true)
    if [[ "$LOADED" -eq 0 ]]; then
      break
    fi
    log "WARNING: $LOADED models still loaded (attempt $attempt/3)"
    lms unload --all 2>/dev/null || true
    sleep 5
  done

  if [[ "$LOADED" -gt 0 ]]; then
    log "ERROR: Cannot unload all models after 3 attempts"
    lms ps 2>&1
    return 1
  fi

  log "Clean state: 0 models loaded, no grepai processes"
}

check_model_file() {
  local LMS_PATH="$1"
  local FULL_PATH="$MODELS_DIR/$LMS_PATH"
  if [[ -f "$FULL_PATH" ]]; then
    local SIZE=$(du -h "$FULL_PATH" | awk '{print $1}')
    log "Model file found: $FULL_PATH ($SIZE)"
    return 0
  fi
  # File missing — try to download from HuggingFace
  log "Model file missing: $FULL_PATH"
  # LMS_PATH format: org/repo-name/filename.gguf
  # HF URL: https://huggingface.co/org/repo-name/resolve/main/filename.gguf
  local REPO=$(dirname "$LMS_PATH")
  local FILENAME=$(basename "$LMS_PATH")
  local HF_URL="https://huggingface.co/${REPO}/resolve/main/${FILENAME}"
  log "Downloading from: $HF_URL"
  local DIR=$(dirname "$FULL_PATH")
  mkdir -p "$DIR"
  curl -L --progress-bar "$HF_URL" -o "$FULL_PATH" 2>&1
  if [[ ! -f "$FULL_PATH" ]] || [[ $(wc -c < "$FULL_PATH" 2>/dev/null || echo 0) -lt 10000 ]]; then
    log "FAILED to download (file too small or missing)"
    rm -f "$FULL_PATH" 2>/dev/null
    return 1
  fi
  log "Downloaded: $(du -h "$FULL_PATH" | awk '{print $1}')"
  return 0
}

verify_embeddings() {
  local MODEL_ID="$1"
  local EXPECTED_DIMS="$2"

  local RESPONSE=$(curl -s "$ENDPOINT/v1/embeddings" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_ID\", \"input\": [\"test embedding verification\"]}" 2>/dev/null)

  local ACTUAL_DIMS=$(echo "$RESPONSE" | python -c "
import sys,json
try:
    d=json.load(sys.stdin)
    v=d['data'][0]['embedding']
    dims=len(v)
    nonzero=sum(1 for x in v if abs(x)>0.0001)
    print(f'{dims}|{nonzero}')
except:
    print('0|0')
" 2>/dev/null)

  local DIMS=$(echo "$ACTUAL_DIMS" | cut -d'|' -f1)
  local NONZERO=$(echo "$ACTUAL_DIMS" | cut -d'|' -f2)

  echo "$DIMS|$NONZERO"
}

write_grepai_config() {
  local MODEL_ID="$1" DIMS="$2"

  cat > "$PROJECT/.grepai/config.yaml" << EOF
version: 1
embedder:
    provider: lmstudio
    model: $MODEL_ID
    endpoint: $ENDPOINT
    dimensions: $DIMS
    parallelism: 4
store:
    backend: gob
chunking:
    size: $CHUNK_SIZE
    overlap: $CHUNK_OVERLAP
watch:
    debounce_ms: 500
    rpg_persist_interval_ms: 1000
    rpg_derived_debounce_ms: 300
    rpg_full_reconcile_interval_sec: 300
    rpg_max_dirty_files_per_batch: 128
search:
    boost:
        enabled: true
        penalties:
            - pattern: /tests/
              factor: 0.5
            - pattern: /test/
              factor: 0.5
            - pattern: __tests__
              factor: 0.5
            - pattern: _test.
              factor: 0.5
            - pattern: .test.
              factor: 0.5
            - pattern: .spec.
              factor: 0.5
            - pattern: test_
              factor: 0.5
            - pattern: /mocks/
              factor: 0.4
            - pattern: /mock/
              factor: 0.4
            - pattern: .mock.
              factor: 0.4
            - pattern: /fixtures/
              factor: 0.4
            - pattern: /testdata/
              factor: 0.4
            - pattern: /generated/
              factor: 0.4
            - pattern: .generated.
              factor: 0.4
            - pattern: .gen.
              factor: 0.4
            - pattern: .md
              factor: 0.6
            - pattern: /docs/
              factor: 0.6
        bonuses:
            - pattern: /src/
              factor: 1.1
            - pattern: /lib/
              factor: 1.1
            - pattern: /app/
              factor: 1.1
    hybrid:
        enabled: false
        k: 60
trace:
    mode: fast
    enabled_languages:
        - .go
        - .js
        - .ts
        - .jsx
        - .tsx
        - .py
        - .php
        - .lua
        - .c
        - .h
        - .cpp
        - .hpp
        - .cc
        - .cxx
        - .rs
        - .zig
        - .cs
        - .java
        - .dart
        - .fs
        - .fsx
        - .fsi
        - .pas
        - .dpr
    exclude_patterns:
        - '*_test.go'
        - '*.spec.ts'
        - '*.spec.js'
        - '*.test.ts'
        - '*.test.js'
        - __tests__/*
rpg:
    enabled: false
update:
    check_on_startup: false
ignore:
    - .git
    - .grepai
    - node_modules
    - vendor
    - bin
    - dist
    - __pycache__
    - .venv
    - venv
    - .idea
    - .vscode
    - target
    - .zig-cache
    - zig-out
    - qdrant_storage
    - build
EOF
}

# ============================================================
# Main benchmark function
# ============================================================

run_benchmark() {
  local TAG="$1" MODEL_ID="$2" LMS_PATH="$3" EXPECTED_DIMS="$4" CTX="$5"
  local OUTFILE="$RESULTS_DIR/bench-${TAG}.txt"

  echo ""
  echo "############################################################"
  echo "# $TAG"
  echo "# Model: $MODEL_ID"
  echo "# File:  $LMS_PATH"
  echo "# Dims:  $EXPECTED_DIMS | Ctx: $CTX | Chunk: $CHUNK_SIZE"
  echo "############################################################"

  # --- Step 1: Nuclear clean ---
  nuclear_clean

  # --- Step 2: Check model file exists ---
  if ! check_model_file "$LMS_PATH"; then
    log "SKIP: Model file not available"
    echo "SKIPPED — model file not found" > "$OUTFILE"
    return 1
  fi

  # --- Step 3: Load model ---
  log "Loading model: $LMS_PATH (ctx=$CTX)..."
  LOAD_START=$(date +%s%N)
  lms load --exact "$LMS_PATH" -c "$CTX" 2>&1
  LOAD_END=$(date +%s%N)
  LOAD_MS=$(( (LOAD_END - LOAD_START) / 1000000 ))
  sleep 3

  # --- Step 4: Verify EXACTLY 1 model is loaded ---
  log "Verifying model loaded..."
  LMS_STATUS=$(lms ps 2>&1)
  LOADED_COUNT=$(echo "$LMS_STATUS" | grep -c "IDLE\|PROCESSING" || true)
  LOADED_MODEL=$(echo "$LMS_STATUS" | grep "IDLE\|PROCESSING" | awk '{print $1}' | head -1)
  log "LMS PS: $LOADED_COUNT model(s) loaded. Active: $LOADED_MODEL"
  if [[ "$LOADED_COUNT" -ne 1 ]]; then
    log "ERROR: Expected 1 model loaded, got $LOADED_COUNT"
    lms ps 2>&1
  fi

  # --- Step 5: Verify embeddings ---
  log "Testing embeddings..."
  EMBED_CHECK=$(verify_embeddings "$MODEL_ID" "$EXPECTED_DIMS")
  ACTUAL_DIMS=$(echo "$EMBED_CHECK" | cut -d'|' -f1)
  NONZERO=$(echo "$EMBED_CHECK" | cut -d'|' -f2)
  log "Dims: $ACTUAL_DIMS | NonZero: $NONZERO / $ACTUAL_DIMS"

  BROKEN="no"
  if [[ "$NONZERO" -lt 10 ]]; then
    log "WARNING: Near-zero embeddings detected — model likely broken on LM Studio"
    BROKEN="yes"
  fi

  # --- Step 6: Init grepai fresh ---
  cd "$PROJECT"
  "$GREPAI" init --provider lmstudio --backend gob --yes 2>&1 > /dev/null
  write_grepai_config "$MODEL_ID" "$ACTUAL_DIMS"
  log "Config written (dims=$ACTUAL_DIMS)"

  # --- Step 7: Index ---
  log "Starting indexing..."
  INDEX_START=$(date +%s)
  "$GREPAI" watch --no-ui 2>&1 &
  WPID=$!

  PREV=0; STABLE=0
  for i in $(seq 1 1200); do
    sleep 3
    C=$("$GREPAI" status 2>&1 | grep "Total chunks:" | awk '{print $3}' || echo "0")
    [[ -z "$C" ]] && C=0
    if [[ "$C" -gt 1000 && "$C" == "$PREV" ]]; then
      STABLE=$((STABLE+1))
      [[ "$STABLE" -ge 5 ]] && break
    else
      STABLE=0
    fi
    PREV="$C"
  done
  INDEX_END=$(date +%s)
  INDEX_SEC=$((INDEX_END - INDEX_START))

  # Kill watcher and wait
  kill $WPID 2>/dev/null || true
  wait $WPID 2>/dev/null || true
  sleep 2

  IDX_SIZE=$(du -sh "$PROJECT/.grepai/" 2>/dev/null | awk '{print $1}')
  FILES=$("$GREPAI" status 2>&1 | grep "Files indexed:" | awk '{print $3}' || echo "?")
  log "Indexed: $FILES files, $C chunks in ${INDEX_SEC}s ($IDX_SIZE)"

  # --- Step 8: Search ---
  log "Running ${#QUERIES[@]} search queries..."
  SEARCH_OUTPUT=""
  TOTAL_MS=0
  for Q in "${QUERIES[@]}"; do
    S=$(date +%s%N)
    RESULT=$("$GREPAI" search "$Q" 2>&1)
    E=$(date +%s%N)
    MS=$(( (E - S) / 1000000 ))
    TOTAL_MS=$((TOTAL_MS + MS))

    SCORE=$(echo "$RESULT" | grep -o 'score: [0-9.]*' | head -1)
    FILE=$(echo "$RESULT" | grep "File:" | head -1 | sed 's/.*File: //')
    LINE=$(printf "  %-14s %-65s %dms" "$SCORE" "$FILE" "$MS")
    echo "$LINE"
    SEARCH_OUTPUT+="$LINE
"
  done
  AVG=$((TOTAL_MS / ${#QUERIES[@]}))
  log "Avg search: ${AVG}ms"

  # --- Step 8b: Immediately free model from RAM ---
  log "Unloading model to free RAM..."
  lms unload --all 2>/dev/null || true
  sleep 2

  # --- Step 9: Write report ---
  cat > "$OUTFILE" << REPORT
============================================================
BENCHMARK: $TAG
============================================================
Date:       $(date -Iseconds)
Model:      $MODEL_ID
LMS Path:   $LMS_PATH
Project:    $(basename "$PROJECT") ($FILES files, $C chunks)
Settings:   ctx=$CTX chunk=$CHUNK_SIZE overlap=$CHUNK_OVERLAP

METRICS:
  Model load:     ${LOAD_MS}ms
  Dimensions:     $ACTUAL_DIMS (expected $EXPECTED_DIMS)
  NonZero values: $NONZERO / $ACTUAL_DIMS
  Broken:         $BROKEN
  Index time:     ${INDEX_SEC}s
  Index size:     $IDX_SIZE
  Avg search:     ${AVG}ms
  Total search:   ${TOTAL_MS}ms (${#QUERIES[@]} queries)

SEARCH RESULTS:
$SEARCH_OUTPUT
============================================================
REPORT

  log "Report saved: $OUTFILE"
  echo ""
  echo "--- $TAG SUMMARY ---"
  echo "  Load: ${LOAD_MS}ms | Index: ${INDEX_SEC}s | Search: ${AVG}ms | Dims: $ACTUAL_DIMS | NonZero: $NONZERO | Broken: $BROKEN"
  echo "---"
}

# ============================================================
# Parse config and run all models
# ============================================================

echo "=========================================================="
echo "EMBEDDING MODEL BENCHMARK"
echo "Config: $CONFIG"
echo "Project: $PROJECT"
echo "Chunk: $CHUNK_SIZE tokens | Overlap: $CHUNK_OVERLAP"
echo "Started: $(date -Iseconds)"
echo "=========================================================="

NUM_MODELS=$(python -c "import json; print(len(json.load(open('$CONFIG_WIN'))))" 2>/dev/null)
log "Loaded $NUM_MODELS models from config"

for i in $(seq 0 $((NUM_MODELS - 1))); do
  MODEL_JSON=$(python -c "
import json
m = json.load(open('$CONFIG_WIN'))[$i]
print(f\"{m['tag']}|{m['model_id']}|{m['lms_path']}|{m['dims']}|{m['ctx']}\")
" 2>/dev/null)

  IFS='|' read -r TAG MODEL_ID LMS_PATH DIMS CTX <<< "$MODEL_JSON"
  run_benchmark "$TAG" "$MODEL_ID" "$LMS_PATH" "$DIMS" "$CTX" || log "FAILED: $TAG"
done

# Final cleanup
nuclear_clean

# Generate comparison
echo ""
echo "=========================================================="
echo "COMPARISON SUMMARY"
echo "=========================================================="
echo ""
printf "%-20s %-8s %-8s %-10s %-10s %-8s %-8s\n" "MODEL" "DIMS" "NONZERO" "INDEX(s)" "SEARCH(ms)" "SIZE" "BROKEN"
printf "%-20s %-8s %-8s %-10s %-10s %-8s %-8s\n" "----" "----" "----" "----" "----" "----" "----"

for f in "$RESULTS_DIR"/bench-*.txt; do
  [[ ! -f "$f" ]] && continue
  TAG=$(basename "$f" .txt | sed 's/bench-//')
  DIMS=$(grep "Dimensions:" "$f" | awk '{print $2}')
  NZ=$(grep "NonZero" "$f" | awk '{print $3}')
  IDX=$(grep "Index time:" "$f" | awk '{print $3}' | sed 's/s//')
  SRCH=$(grep "Avg search:" "$f" | awk '{print $3}' | sed 's/ms//')
  SZ=$(grep "Index size:" "$f" | awk '{print $3}')
  BRK=$(grep "Broken:" "$f" | awk '{print $2}')
  printf "%-20s %-8s %-8s %-10s %-10s %-8s %-8s\n" "$TAG" "$DIMS" "$NZ" "$IDX" "$SRCH" "$SZ" "$BRK"
done

echo ""
echo "=========================================================="
echo "COMPLETE: $(date -Iseconds)"
echo "=========================================================="
