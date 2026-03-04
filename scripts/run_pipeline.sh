#!/bin/bash
set -e
cd /home/h1aihan/supermicro-rag
source venv/bin/activate

LOG=/home/h1aihan/supermicro-rag/pipeline.log
echo "=== Pipeline started at $(date) ===" > "$LOG"

echo "--- Step 2: process_pages ---" >> "$LOG"
python src/process_pages.py --input data/pages/ --output data/raw_pages/ >> "$LOG" 2>&1
echo "raw_pages count: $(ls data/raw_pages/ | wc -l)" >> "$LOG"
echo "eStore accessory files: $(ls data/raw_pages/ | grep -c 'eStore_MCP\|eStore_PWS\|eStore_CSE\|eStore_CBL\|eStore_AOC\|eStore_FAN\|eStore_SNK\|eStore_MBD' || echo 0)" >> "$LOG"

echo "--- Step 3: chunk ---" >> "$LOG"
python src/chunk.py --input data/raw_text/ data/raw_pages/ --output data/chunks.jsonl >> "$LOG" 2>&1
echo "chunks count: $(wc -l < data/chunks.jsonl)" >> "$LOG"

echo "--- Step 4: embed ---" >> "$LOG"
python src/embed.py --input data/chunks.jsonl --output embeddings/faiss_index/ >> "$LOG" 2>&1
echo "metadata count: $(wc -l < embeddings/faiss_index/metadata.jsonl)" >> "$LOG"

echo "=== Pipeline finished at $(date) ===" >> "$LOG"
echo "DONE" >> "$LOG"
