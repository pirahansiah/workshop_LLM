# Workshop LLM

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LLM](https://img.shields.io/badge/LLM-2025-FF6B6B?style=for-the-badge)](https://arxiv.org/abs/2501.12948)
[![RAG](https://img.shields.io/badge/RAG-2025-4ECDC4?style=for-the-badge)](https://arxiv.org/abs/2504.02675)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-5C3EE8?style=for-the-badge&logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

Hands-on workshop repository covering **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, and **Computer Vision** integration. Contains code examples, notebooks, and tools for building AI-powered applications.

## 2025-2026 LLM Landscape

### Latest Models & Architectures

| Model | Release | Key Innovation |
|-------|---------|----------------|
| GPT-5 | 2025 | Native multimodal reasoning |
| Claude 4 | 2025 | Extended thinking, tool use |
| Gemini 2.5 | 2025 | 1M+ context, native audio |
| Llama 4 | 2025 | MoE architecture, Scout/Maverick |
| DeepSeek-R1 | 2025 | Chain-of-thought reasoning |
| Qwen 3 | 2025 | Hybrid thinking mode |
| Mistral Large 2 | 2025 | 128K context, multilingual |
| Gemma 3 | 2025 | Multimodal, open-weight |

### RAG Evolution (2025-2026)

1. **GraphRAG 2.0** — Microsoft's graph-based RAG with community detection
2. **Agentic RAG** — Self-correcting retrieval with multi-step reasoning
3. **Multimodal RAG** — Vision + text retrieval (ColPali, Nomic Embed Vision)
4. **Cache-Augmented Generation (CAG)** — Preloading context to bypass retrieval
5. **Corrective RAG (CRAG)** — Self-evaluation and retrieval refinement
6. **Adaptive RAG** — Dynamic query routing between retrieval strategies
7. **Late Chunking** — Contextual chunk embeddings for better retrieval

### Vector Databases (2025)

| Database | Specialty | Scale |
|----------|-----------|-------|
| Qdrant | Rust-based, high perf | Billion+ vectors |
| Weaviate | Hybrid search, GraphQL | Enterprise |
| Chroma | Developer-friendly | Local-first |
| Pinecone | Serverless, managed | Auto-scaling |
| Milvus | GPU-accelerated | Distributed |
| pgvector | PostgreSQL native | SQL queries |
| LanceDB | Multimodal, disk-based | Edge-friendly |
| Turbopuffer | Serverless, fast | Vector search |

## Workshops

### Workshop 1: LLM Fundamentals
- Prompt engineering patterns (2025 best practices)
- Few-shot, chain-of-thought, tree-of-thought prompting
- Structured output with JSON mode
- Function calling and tool use

### Workshop 2: RAG Pipeline
- Document loading and chunking strategies
- Embedding models: `nomic-embed-text`, `mxbai-embed-large`
- Vector store setup with Qdrant/Chroma
- Query transformation and HyDE
- Reranking with cross-encoders

### Workshop 3: Multimodal LLMs
- Vision-Language Models (VLMs)
- Image understanding with GPT-4V, Gemini Vision
- Video analysis with LLMs
- Document parsing with LLMs

### Workshop 4: Fine-tuning & Deployment
- LoRA/QLoRA fine-tuning
- GGUF/GGML quantization for local inference
- Ollama, vLLM, TGI deployment
- Edge deployment with ONNX Runtime

## Code Examples

| Script | Description |
|--------|-------------|
| `gemma-test.py` | Kaggle Gemma model setup |
| `FP.py` | PyQt5 GUI for OpenCV testing |
| `chatWithLLMs.py` | Async chat with aiohttp |
| `LLMOPs.ipynb` | LLMOps & RAG workflow |
| `OpenRouter.py` | Multi-model chat via OpenRouter |
| `mlflow/test-mlflow.py` | MLflow experiment tracking |

## Quick Start

```bash
# Clone repository
git clone https://github.com/pirahansiah/workshop_LLM.git
cd workshop_LLM

# Install dependencies
pip install -r requirements.txt

# Start Ollama (local LLM)
ollama pull llama3.2
ollama serve

# Run Jupyter notebook
jupyter notebook
```

## Modern LLM Stack (2025-2026)

```
┌─────────────────────────────────────────────────────┐
│                    Application Layer                │
│   Streamlit / Gradio / FastAPI / LangServe          │
├─────────────────────────────────────────────────────┤
│                   Orchestration                     │
│   LangChain / LlamaIndex / CrewAI / AutoGen        │
├─────────────────────────────────────────────────────┤
│                   LLM Inference                     │
│   vLLM / TGI / Ollama / SGLang / LiteLLM           │
├─────────────────────────────────────────────────────┤
│                   Vector Store                       │
│   Qdrant / Chroma / Weaviate / pgvector / LanceDB   │
├─────────────────────────────────────────────────────┤
│                   Embeddings                         │
│   OpenAI / Cohere / Jina / Nomic / MxBai            │
├─────────────────────────────────────────────────────┤
│                   GPU / Hardware                     │
│   CUDA 12.x / ROCm / Apple Silicon / ONNX Runtime   │
└─────────────────────────────────────────────────────┘
```

## Resources

- [LangChain Docs](https://python.langchain.com)
- [LlamaIndex Docs](https://docs.llamaindex.ai)
- [Hugging Face](https://huggingface.co)
- [Ollama](https://ollama.ai)
- [vLLM](https://github.com/vllm-project/vllm)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## 12-Month Roadmap (2025-2026)

### Month 1-2: Foundation
- [ ] Add LLM chat modules with OpenAI API v2.x (chat completions, function calling)
- [ ] Implement async streaming responses for Ollama/OpenAI
- [ ] Add type hints and pytest to all existing Python files
- [ ] Set up CI/CD with GitHub Actions (lint, test, type-check)

### Month 3-4: RAG Pipeline
- [ ] Build document loader with smart chunking (semantic, recursive)
- [ ] Integrate Qdrant/Chroma vector store with hybrid search
- [ ] Implement HyDE (Hypothetical Document Embeddings) query transformation
- [ ] Add cross-encoder reranking with Cohere/Jina
- [ ] Create RAG evaluation framework (faithfulness, relevancy scores)

### Month 5-6: Multimodal Integration
- [ ] Connect CV calibration output to LLM vision pipelines
- [ ] Implement image-to-text with GPT-4V / Gemini Vision
- [ ] Add video frame extraction and LLM analysis pipeline
- [ ] Build Streamlit demo for real-time multimodal interaction

### Month 7-8: Fine-tuning & Edge Deployment
- [ ] LoRA/QLoRA fine-tuning scripts for Llama 4, Qwen 3
- [ ] GGUF quantization pipeline for Ollama deployment
- [ ] ONNX Runtime export for edge devices (Raspberry Pi 5, Jetson)
- [ ] Benchmark inference latency across hardware targets

### Month 9-10: Agentic Workflows
- [ ] Multi-agent system with CrewAI for automated CV analysis
- [ ] Tool-use framework connecting OpenCV functions as LLM tools
- [ ] Self-correcting RAG with CRAG pattern
- [ ] Add LangGraph state-machine for complex workflows

### Month 11-12: Production & Scale
- [ ] FastAPI backend with streaming SSE endpoints
- [ ] Docker Compose for full stack (Ollama + Qdrant + API)
- [ ] NVIDIA Spark optimization with CUDA 13 kernels
- [ ] Apple Silicon optimization (Metal, Neural Engine)
- [ ] Comprehensive documentation and video tutorials
- [ ] Performance benchmarks: M5 Max vs Intel Ultra 9 vs Jetson

---

**Maintainer:** [Farshid Pirahansiah](https://github.com/pirahansiah)