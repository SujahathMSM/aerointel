# AeroIntel

AeroIntel is an aviation-focused AI engineering and safety intelligence platform.

The project is being built from first principles to explore and implement:

- text embeddings
- semantic search
- vector retrieval
- keyword and hybrid retrieval
- retrieval-augmented generation (RAG)
- grounded answers with citations
- aviation safety data analysis

## Current Status

AeroIntel is currently in Stage 0: Local AI Foundation.

Implemented so far:

- Local LLM inference with Ollama
- Qwen3 local generation
- EmbeddingGemma embeddings
- 768-dimensional text embeddings
- Cosine similarity implemented in Python
- Basic semantic-search ranking experiment

## Current Semantic Search Flow

```text
Query text
    ↓
EmbeddingGemma
    ↓
Query embedding

Candidate text
    ↓
EmbeddingGemma
    ↓
Document embeddings

Query embedding
    ↓
Cosine similarity
    ↓
Ranked semantic search results