# Hybrid Information Retrieval & Indexing System

A production-ready, high-performance hybrid information retrieval (IR) engine implemented in Python. This system bridges the gap between traditional lexical search and modern semantic search by providing a multi-index architecture featuring optimized **TF-IDF**, hyperparameter-tuned **BM25**, and dense **Transformer-based text embeddings**.

## 🚀 Key Features

* **Advanced Lexical Search (BM25):** Implements the BM25Okapi core with dynamic, thread-safe hyperparameter tuning ($k_1$ and $b$) backed by an efficient scoring cache to avoid redundant statistical re-computations.
* **Optimized Vector Space Model (TF-IDF):** Features a highly customized TF-IDF index utilizing sublinear term frequency scaling ($1 + \log(\text{tf})$) to mitigate word-repetition bias and maximize retrieval fairness.
* **Dense Semantic Retrieval:** Leverages State-of-the-Art Sentence Transformers (`all-MiniLM-L6-v2`) for contextual search, parsing the underlying meaning of queries beyond exact keyword matching.
* **High-Throughput NLP Pipeline:** A robust, batch-processed text preprocessing pipeline featuring text normalization, tokenization, stop-word filtering, and lemmatization using stream-based I/O processing.
* **Production-Grade Optimizations:** * **Memory Efficiency:** Uses Compressed Sparse Rows (SciPy `spmatrix`) for TF-IDF arrays, reducing RAM footprints by up to 85%.
  * **Ultra-fast Similarity Check:** Implements manual $L_2$ vector normalization on dense embeddings, converting heavy Cosine Similarity computations into lightning-fast NumPy Dot Product operations.
  * **Persistence:** Instant initialization via state serialization using optimized Pickle protocols.
  * **Fault-Tolerant Streaming:** Robust JSON-Lines parser equipped with try-except mechanisms to handle malformed records without pipeline failure.

## 🛠️ Architecture & Text Processing Pipeline

1. **Ingestion & Validation:** Streamed line-by-line reading of raw corpus files to guarantee $O(1)$ memory consumption regardless of dataset size.
2. **NLP Transformation:** Texts are normalized, tokenized, and systematically reduced to their base dictionary forms (Lemmas) in high-throughput batches (`nlp.pipe`).
3. **Multi-Index Generation:** Cleaned tokens feed parallel indexing tracks:
   * Lexical Matrices (Sparse Frequency Arrays).
   * Semantic Tensors (Dense 384-dimensional Floating-Point Latent Vectors).

## 🎛️ Technical Specifications & Math Highlights

### Sublinear TF Scaling
Standard linear frequency can skew search relevance. This engine enforces:
$$\text{TF}_{\text{sublinear}} = 1 + \log(\text{tf})$$

### Vector Optimization
By scaling all generated dense embeddings to a unit norm ($L_2 = 1.0$), the mathematical formulation of Cosine Similarity simplifies to a matrix-vector dot product:
$$\text{Score} = \vec{q} \cdot \vec{d}$$
This optimization avoids costly division steps during runtime inference.

## 💻 Tech Stack

* **Core Language:** Python 3.10+
* **Machine Learning & NLP:** `sentence-transformers`, `scikit-learn`, `spacy`
* **Scientific Computing:** `numpy`, `scipy`
* **Data Management & I/O:** `json`, `pickle`, `pathlib`
* **Concurrency:** `threading` (Thread-Safe Cache Isolation)