# Autonomous arXiv Paper Digest & QA Agent

An autonomous research assistant that retrieves scientific papers from arXiv, processes their PDFs, creates a semantic vector index, generates a structured executive briefing, and answers follow-up questions using Retrieval-Augmented Generation (RAG).

The system is implemented as an explicit stateful workflow using **LangGraph**, with separate nodes for query understanding, arXiv retrieval, paper selection, PDF parsing, chunking, vector indexing, summarization, and question answering.

## 🚀 Features

- Accepts arXiv paper IDs, arXiv URLs, or natural-language research topics
- Retrieves papers using the official arXiv API
- Selects a relevant paper for topic-based queries
- Downloads and parses research-paper PDFs
- Splits papers into overlapping text chunks
- Generates local embeddings using FastEmbed
- Performs semantic search using FAISS
- Generates a structured executive briefing
- Supports interactive paper-specific QA
- Maintains QA conversation history
- Grounds answers in retrieved paper excerpts
- Handles retrieval, parsing, indexing, and generation failures gracefully

## 🏗️ Architecture

```text
                         User Query
                              |
                              v
                  ┌─────────────────────┐
                  │ Query Understanding │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │   arXiv Retrieval   │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │  Paper Ranking /    │
                  │     Selection       │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │ PDF Download +      │
                  │      Parsing        │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │    Text Chunking    │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │ FastEmbed Embedding │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │   FAISS Vector      │
                  │      Store          │
                  └──────────┬──────────┘
                             |
                             v
                  ┌─────────────────────┐
                  │ Map-Reduce Paper    │
                  │    Summarization    │
                  └──────────┬──────────┘
                             |
                             v
                       Executive
                        Briefing
                             |
                             v
                    ┌────────────────┐
                    │ Interactive QA │
                    └───────┬────────┘
                            |
                            v
                    Semantic Retrieval
                            |
                            v
                     Grounded Answer
```

## 🧩 LangGraph Workflow

The ingestion pipeline is implemented as an explicit graph rather than a single monolithic LLM prompt.

```text
START
  |
  v
query_understanding
  |
  v
arxiv_retrieval
  |
  v
paper_ranking
  |
  v
pdf_parser
  |
  v
chunking
  |
  v
vector_store
  |
  v
summarizer
  |
  v
END
```

The QA stage operates on the persisted vector store created during ingestion.

## 🧠 Shared Agent State

The LangGraph workflow maintains shared state across nodes:

```text
user_query
query_type
arxiv_results
selected_paper
pdf_path
parsed_text
chunks
vector_store_path
briefing
current_question
retrieved_chunks
answer
conversation_history
error
```

This allows information produced by one node to be consumed by later nodes while preserving the state of the research workflow.

## 📑 Executive Briefing

For every selected paper, the agent generates a structured briefing containing:

- Title
- Authors
- arXiv ID
- Publication date
- arXiv link
- Why the paper matters
- Problem statement
- Method
- Key results
- Limitations
- Follow-up questions

The goal is to provide a concise research-oriented overview instead of returning only a generic LLM summary.

## 💬 Retrieval-Augmented QA

After the paper is indexed, users can ask questions about the paper.

The QA pipeline follows these steps:

```text
User Question
      |
      v
Question Embedding
      |
      v
FAISS Similarity Search
      |
      v
Top Relevant Paper Chunks
      |
      v
LLM + Retrieved Evidence
      |
      v
Grounded Answer
```

The QA prompt instructs the model to use only the retrieved paper excerpts.

If the retrieved evidence does not contain the requested information, the agent responds:

> I couldn't find this information in the paper.

This helps keep the QA process grounded in the selected paper.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Agent Orchestration | LangGraph |
| Paper Source | Official arXiv API |
| PDF Parsing | PyMuPDF |
| Embeddings | FastEmbed |
| Vector Database | FAISS |
| LLM Provider | Groq |
| LLM Model | `openai/gpt-oss-120b` |
| Environment Management | Conda |
| Secrets Management | python-dotenv |

## 📂 Project Structure

```text
arxiv-digest-agent/
│
├── app/
│   ├── __init__.py
│   │
│   ├── nodes/
│   │   ├── arxiv.py
│   │   ├── chunking.py
│   │   ├── main.py
│   │   ├── pdf_parser.py
│   │   ├── qa.py
│   │   ├── query.py
│   │   ├── ranking.py
│   │   └── summarizer.py
│   │
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── graph.py
│   └── state.py
│
├── data/
│   ├── papers/
│   └── vectorstore/
│
├── examples/
├── tests/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

Generated PDFs and vector stores are excluded from version control.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/DebasisHX/arxiv-digest-agent.git
cd arxiv-digest-agent
```

### 2. Create the Conda environment

```bash
conda create -n arxiv-agent python=3.11 -y
conda activate arxiv-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔑 Environment Configuration

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_api_key_here
```

The API key is loaded using `python-dotenv`.

**Never commit `.env` or expose the API key publicly.**

The `.gitignore` file excludes `.env` from version control.

## ▶️ Running the Agent

Run:

```bash
python main.py
```

The agent accepts an arXiv paper ID, arXiv URL, or research topic.

### Example: arXiv ID

```text
1706.03762
```

### Example: arXiv URL

```text
https://arxiv.org/abs/1706.03762
```

### Example: Research Topic

```text
transformer architecture for machine translation
```

## 🧪 Example Run

For the paper:

```text
1706.03762
```

the agent retrieves:

```text
Attention Is All You Need
```

The workflow then:

1. Retrieves the paper metadata from arXiv.
2. Downloads the PDF.
3. Extracts the paper text using PyMuPDF.
4. Splits the text into overlapping chunks.
5. Generates embeddings using FastEmbed.
6. Builds a FAISS vector index.
7. Generates an executive briefing.
8. Starts an interactive QA session.

## 💡 Example Questions

### Question 1

```text
What dataset was used to evaluate the Transformer?
```

The agent retrieves relevant evidence from the paper and generates a grounded answer.

### Question 2

```text
What is the main architectural idea introduced in the paper?
```

The retrieval system finds relevant sections discussing self-attention and the Transformer architecture.

### Question 3

```text
What are the limitations of the approach?
```

The agent retrieves relevant evidence from the paper and answers based on the available text.

## 🔄 Error Handling

The system handles common failure cases including:

- Empty user queries
- Invalid arXiv identifiers
- No matching arXiv papers
- arXiv API failures
- PDF download failures
- Invalid PDF responses
- PDFs with no extractable text
- Empty or invalid chunks
- Embedding failures
- Vector-store failures
- LLM/API failures
- Missing retrieved evidence

Errors are propagated through the shared graph state instead of silently failing.

## ⚖️ Design Trade-offs

### FastEmbed + FAISS

The project uses local embeddings and FAISS instead of a hosted vector database.

**Advantages:**

- Free to use
- Simple local setup
- No separate vector database service
- Local vector indexing

**Trade-off:**

Local embedding generation can require significant memory and may be slower on lower-end hardware.

### Fixed-Size Chunking

The current implementation uses overlapping character-based chunks.

**Advantages:**

- Simple
- Predictable
- Easy to implement
- Works across different PDF layouts

**Trade-off:**

The approach does not fully understand academic section boundaries and may occasionally split related information across chunks.

### Map-Reduce Summarization

Long papers are divided into smaller sections. Each section is summarized independently before a final synthesis step.

**Advantages:**

- Handles long documents
- Reduces context-size pressure
- Allows processing of papers that are too large for one request

**Trade-off:**

It requires multiple LLM calls and therefore increases processing time and API usage.

### Groq LLM

Groq is used for hosted LLM inference.

**Advantages:**

- Fast inference
- No need to run a large generative model locally

**Trade-off:**

The system depends on external API availability and usage limits.

## ⚠️ Current Limitations

- PDF extraction quality depends on the structure of the source PDF.
- Fixed-size chunking may not preserve complete academic sections.
- Semantic retrieval may occasionally return less relevant chunks.
- Summary quality depends on the extracted text and LLM generation.
- The system currently focuses on arXiv papers.
- No authentication or deployment infrastructure is included.
- Scanned PDFs without an extractable text layer may not be processed correctly.

## 🔐 Security

API credentials are loaded through environment variables.

The following files and directories are excluded from Git:

```text
.env
data/papers/
data/vectorstore/
__pycache__/
.ipynb_checkpoints/
.virtual_documents/
```

Do not commit API keys or other secrets.

## 📌 Future Improvements

Potential improvements include:

- Section-aware academic document chunking
- Better paper-ranking algorithms
- Citation-aware retrieval
- Hybrid keyword + semantic search
- Reranking retrieved chunks
- Persistent conversation sessions
- Support for multiple papers
- Improved PDF layout extraction
- Evaluation datasets for retrieval and QA quality

## 👨‍💻 Author

**Debasish Dey**

B.Tech — Artificial Intelligence & Machine Learning  
Techno International New Town, Kolkata

GitHub: https://github.com/DebasisHX

## 📄 License

This project is licensed under the MIT License.
