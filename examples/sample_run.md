# Example Run

## Input

```text
1706.03762
```

## Selected Paper

**Attention Is All You Need**

- **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
- **arXiv ID:** 1706.03762
- **Published:** 2017-06-12
- **Link:** https://arxiv.org/abs/1706.03762

## Processing Pipeline

The agent performs the following steps:

```text
User Query
    ↓
Query Understanding
    ↓
arXiv Retrieval
    ↓
Paper Selection
    ↓
PDF Download
    ↓
PDF Parsing
    ↓
Text Chunking
    ↓
FastEmbed Embeddings
    ↓
FAISS Vector Store
    ↓
Executive Briefing
    ↓
Interactive QA
```

## Example QA 1

**Question:**

```text
What dataset was used to evaluate the Transformer?
```

**Answer:**

The Transformer was evaluated on the WMT 2014 translation benchmarks, including the English-to-German and English-to-French translation tasks.

---

## Example QA 2

**Question:**

```text
What is the main architectural idea introduced in the paper?
```

**Answer:**

The paper introduces the Transformer architecture, which replaces recurrent and convolutional sequence-processing components with attention-based mechanisms, particularly multi-head self-attention.

---

## Example QA 3

**Question:**

```text
What is the computational limitation of self-attention for long sequences?
```

**Answer:**

The paper describes self-attention as having computational complexity that grows quadratically with sequence length, making it more expensive for very long sequences.

---

## Grounding Behavior

The QA system retrieves relevant chunks from the paper using FAISS semantic similarity search before generating an answer.

The language model is instructed to use only the retrieved paper excerpts.

If the requested information cannot be found in the retrieved evidence, the system responds:

```text
I couldn't find this information in the paper.
```

This prevents the QA stage from intentionally relying on unrelated outside knowledge.
