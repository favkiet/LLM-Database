# LLM-Database: RAG-based Text-to-SQL System

Dự án baseline RAG Text-to-SQL dựa trên paper **"Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQLs"** (BIRD-SQL benchmark).

## 🎯 Mục tiêu

Xây dựng hệ thống Text-to-SQL sử dụng RAG (Retrieval-Augmented Generation) để chuyển đổi câu hỏi ngôn ngữ tự nhiên thành SQL queries trên BIRD-SQL dataset.

## 🏗️ Kiến trúc

### LangGraph-Based Architecture
TODO

**🎯 Why LangGraph?**
- ✅ **Official**: Supported by LangChain team
- ✅ **Type-Safe**: TypedDict state definition
- ✅ **Production-Ready**: Battle-tested framework
- ✅ **Advanced Features**: Conditional routing, checkpointing, streaming
- ✅ **LangSmith**: Native tracing integration

## 🚀 Công nghệ sử dụng

- **Architecture**: LangGraph (official framework from LangChain)
- **Tracing**: LangSmith (professional monitoring)
- **LLM**: Ollama (Phase 1) → OpenAI API (Phase 2)
- **Embeddings**: 
- **Vector Store**: 
- **Database**: SQLite (cho BIRD-SQL databases)
- **Framework**: Python 3.9+

## 📁 Cấu trúc dự án

```

```

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone <repo-url>
cd LLM-Database
```

### 2. Cài đặt dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 3. Cài đặt Ollama

```bash
# MacOS
brew install ollama

# Start Ollama service
ollama serve

# Pull model (ví dụ: codellama)
ollama pull codellama:7b
```

## 💻 Sử dụng

### Inference

```bash
python main.py
```

### Visualize Graph

```bash
# Show graph structure
python visualize_graph.py
```

### Enable LangSmith Tracing
How to get API key Langsmith
Website: https://smith.langchain.com -> Setting -> + API Key

```bash
# Set environment variable
export LANGSMITH_API_KEY="your-api-key"

# Or in .env file
echo "LANGSMITH_API_KEY=your-api-key" >> .env
```


## 📈 Roadmap

- [x] Phase 1: Setup project structure
- [ ] Phase 1: Text-to-SQL
- [ ] Phase 1: Implement baseline with Ollama
- [ ] Phase 1: Evaluate on single domain
- [ ] Phase 2: Integrate OpenAI API
