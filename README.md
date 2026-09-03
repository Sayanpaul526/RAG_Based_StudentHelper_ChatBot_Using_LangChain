# 📚 RAG-Based Student Note Maker

A powerful **Retrieval-Augmented Generation (RAG)** system that lets students upload lecture notes and PDFs, then ask intelligent questions using AI. Built with LangChain, Pinecone, and Streamlit.

## ✨ Features

- 📤 **PDF Upload**: Upload multiple lecture notes/PDFs
- 🔍 **Semantic Search**: Find relevant content using vector embeddings
- 💬 **Chat Interface**: Multi-turn conversations with conversation memory
- 🔐 **User Isolation**: Each user's documents stored in separate namespaces
- 🧠 **RAG Pipeline**: LangChain-powered retrieval + generation
- 🚀 **Fast & Local**: Ollama embeddings run locally (no quota limits)
- ⚡ **Production-Ready**: Built with Streamlit for easy deployment

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | Google Gemini 2.0 Flash |
| **Embeddings** | Ollama (nomic-embed-text-v2-moe locally) |
| **Vector Database** | Pinecone (cloud) |
| **Framework** | LangChain |
| **UI** | Streamlit |
| **PDF Processing** | PyPDFLoader |
| **Text Splitting** | RecursiveCharacterTextSplitter (1000 chars, 200 overlap) |
| **Language** | Python 3.13+ |

## 📋 Architecture

```
User Input (PDF/Question)
    ↓
Document Loading (PyPDFLoader)
    ↓
Text Chunking (RecursiveCharacterTextSplitter)
    ↓
Embeddings (Ollama - Local)
    ↓
Vector Storage (Pinecone - Cloud)
    ↓
Semantic Retrieval (MMR Search)
    ↓
RAG Chain (LangChain)
    ↓
LLM Response (Gemini 2.0 Flash)
    ↓
User Chat Interface (Streamlit)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Ollama installed and running
- Google API key
- Pinecone API key

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Sayanpaul526/student-rag-notemaker.git
cd student-rag-notemaker
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Create `.env` file**
```bash
cp .env.example .env
```

**5. Add your own API keys to `.env`**
```
GOOGLE_API_KEY=your_actual_key_here
PINECONE_API_KEY=your_actual_key_here
```

### Running the Application

**1. Start Ollama server** (in a separate terminal)
```bash
ollama serve
```

Verify it's running:
```bash
ollama list
```

**2. Run the Streamlit app**
```bash
streamlit run app.py
```

**3. Open in browser**
```
http://localhost:8501
```

## 📖 Usage

1. **Enter your User ID**: In the sidebar, enter a unique identifier (email recommended)
2. **Upload PDFs**: Click "Upload your PDFs" and select files
3. **Process Documents**: Click "Process & Upload to Pinecone"
4. **Ask Questions**: Type questions in the chat and get AI-powered answers
5. **Maintain Conversation**: Chat history is preserved automatically

## 🎯 Key Components

### `app.py` (Main Streamlit Application)
- User authentication with namespaces
- PDF upload and processing
- Chat interface with memory
- RAG chain integration

### `RAG_based_chatbot.py` (Core RAG Logic)
- Document loading and chunking
- Vector store initialization
- Retrieval chain setup
- LLM prompting and generation

### `LocalOllamaEmbeddings` (Custom Class)
- HTTP wrapper for Ollama embeddings
- Bypasses langchain_ollama for reliability
- Supports both document and query embeddings

## 🔒 Security Features

- **User Isolation**: Each user's documents stored in separate Pinecone namespace
- **No API Key Leaks**: `.env` file excluded from git
- **Private Space**: User ID gates access to their documents

## 📊 Retrieval Strategy

- **Search Type**: MMR (Maximal Marginal Relevance)
- **Top K**: 4 documents
- **Fetch K**: 20 candidates
- **Lambda Mult**: 0.5 (diversity factor)

## 🧪 Example Workflow

```
# Upload notes on "Deep Learning"

# Question 1: "What is deep learning?"
# Answer: Retrieved from documents + LLM generation

# Question 2: "Explain neural networks"
# Answer: Uses previous context + new retrieval

# Question 3: "How are they connected?"
# Answer: Maintains conversation context
```

## ⚡ Performance

- **Embedding Generation**: Local Ollama (instant after model load)
- **Vector Search**: Pinecone (milliseconds)
- **LLM Response**: Gemini 2.0 Flash (1-3 seconds)
- **Total Latency**: ~2-4 seconds per query

## 🐛 Troubleshooting

### Ollama Connection Error
```
Error: No connection could be made
```
**Solution**: Make sure `ollama serve` is running in another terminal

### Dimension Mismatch
```
Vector dimension 4096 does not match index dimension 1024
```
**Solution**: Recreate Pinecone index with correct dimensions matching embedding model

### Model Not Found
```
gemini-1.5-flash is not found
```
**Solution**: Update to `gemini-2.0-flash` in `app.py`

## 📦 Project Structure

```
student-rag-notemaker/
├── app.py                      # Main Streamlit application
├── RAG_based_chatbot.py       # Core RAG pipeline
├── requirements.txt            # Python dependencies
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
└── data1/                     # Local PDF storage (not uploaded)
```

## 🔄 Workflow Overview

### Ingestion Phase
- User uploads PDF
- Documents loaded with PyPDFLoader
- Split into chunks (1000 chars, 200 overlap)
- Embedded with Ollama (local)
- Stored in Pinecone with user namespace

### Query Phase
- User asks question
- Question embedded with Ollama
- Semantic search in Pinecone (MMR)
- Top 4 documents retrieved
- RAG chain processes with LangChain
- Gemini generates response
- Conversation saved to memory

## 🎓 Learning Resources

This project demonstrates:
- ✅ RAG (Retrieval-Augmented Generation) patterns
- ✅ Vector embeddings and semantic search
- ✅ LangChain pipeline orchestration
- ✅ Multi-user isolation in LLM apps
- ✅ Streamlit for rapid UI development
- ✅ LLM integration and prompting

## 🚀 Deployment

Ready to deploy? See deployment guides:
- [Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [HuggingFace Spaces](https://huggingface.co/spaces)
- [AWS/GCP/Azure](https://docs.streamlit.io/deploy)

## 📝 Configuration

### Chunk Size (in `app.py`)
```python
chunk_size = 1000        # Characters per chunk
chunk_overlap = 200      # Character overlap between chunks
```

### Retrieval Parameters
```python
search_type = 'mmr'      # Maximal Marginal Relevance
k = 4                    # Number of documents to retrieve
fetch_k = 20             # Candidates to consider
lambda_mult = 0.5        # Diversity factor (0-1)
```

### LLM Model
```python
model = 'gemini-2.0-flash'  # Fast and efficient
```

## 🤝 Contributing

Contributions welcome! Areas to improve:
- [ ] Multi-language support
- [ ] Document summarization
- [ ] Flashcard generation
- [ ] Study mode (spaced repetition)
- [ ] Performance optimization
- [ ] Better error handling

## 📄 License

MIT License - Feel free to use for personal/educational projects

## 👤 Author

**Sayan Paul**
- GitHub: [@Sayanpaul526](https://github.com/Sayanpaul526)
- Email: sayanpaul526@gmail.com
- LinkedIn: [Sayan Paul](https://linkedin.com/in/sayanpaul)

## 🙏 Acknowledgments

- LangChain for RAG framework
- Pinecone for vector database
- Ollama for local embeddings
- Google Gemini for LLM
- Streamlit for UI framework

---

**Made with ❤️ for students. Happy learning! 📚**