import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


# ==========================================
# 1. CORE VECTOR MATHEMATICS (FROM SCRATCH)
# ==========================================

def vector_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the dot (inner) product of two 1D vectors."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def vector_l2_norm(a: np.ndarray) -> float:
    """Computes the Euclidean L2 norm (magnitude) of a 1D vector."""
    return float(np.sqrt(np.sum(a ** 2)))


def vector_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalizes a 1D vector to unit length (L2 norm = 1.0)."""
    norm = vector_l2_norm(a)
    if norm < eps:
        return np.zeros_like(a)
    return a / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Computes cosine similarity between two 1D vectors with a zero-vector guard."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    norm_a = vector_l2_norm(a)
    norm_b = vector_l2_norm(b)

    if norm_a < eps or norm_b < eps:
        return 0.0

    return float(np.sum(a * b) / (norm_a * norm_b))


# ==========================================
# 2. MATRIX VECTORIZATION & TOP-K RETRIEVAL
# ==========================================

def cosine_similarity_matrix(
    query: np.ndarray, corpus: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Computes cosine similarity between a 1D query and a 2D corpus matrix.
    
    query: shape (D,)
    corpus: shape (N, D)
    Returns: 1D array of shape (N,)
    """
    if query.ndim != 1 or corpus.ndim != 2:
        raise ValueError("query must be 1D and corpus must be 2D")
    if query.shape[0] != corpus.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query ({query.shape[0]}) != corpus ({corpus.shape[1]})"
        )

    norm_query = vector_l2_norm(query)
    norm_corpus = np.linalg.norm(corpus, axis=1)

    if norm_query < eps:
        return np.zeros(corpus.shape[0], dtype=np.float32)

    dot_products = corpus @ query
    denominators = norm_corpus * norm_query
    denominators = np.where(denominators < eps, np.nan, denominators)

    return np.nan_to_num(dot_products / denominators, nan=0.0)


def retrieve_top_k(
    query: np.ndarray, corpus: np.ndarray, k: int = 3, eps: float = 1e-12
) -> list[tuple[int, float]]:
    """Ranks and retrieves the top-k most similar document indices and scores."""
    similarities = cosine_similarity_matrix(query, corpus, eps=eps)
    k_bounded = min(k, len(similarities))
    sorted_indices = np.argsort(similarities)[::-1][:k_bounded]
    return [(int(idx), float(similarities[idx])) for idx in sorted_indices]


# ==========================================
# 3. VERIFICATION & BENCHMARK SUITE
# ==========================================

def run_tests_and_benchmark():
    print("--- 1. Testing Core Math & Edge Cases ---")
    v1 = np.array([3.0, 4.0, 0.0])
    v2 = np.array([0.0, 6.0, 8.0])
    v_zero = np.array([0.0, 0.0, 0.0])

    print(f"Dot Product (v1 . v2)       : {vector_dot_product(v1, v2)}")
    print(f"L2 Norm (v1)                : {vector_l2_norm(v1)}")
    print(f"L2 Norm (v2)                : {vector_l2_norm(v2)}")
    print(f"Cosine Similarity (v1, v2)  : {cosine_similarity(v1, v2):.4f}")
    print(f"Zero-Vector Guard (v1, zero): {cosine_similarity(v1, v_zero):.4f}")

    print("\n--- 2. Testing Vectorized Top-K Retrieval ---")
    sample_corpus = np.array([
        [0.1, 0.9, 0.0],  # Doc 0
        [0.0, 0.0, 1.0],  # Doc 1
        [1.0, 1.0, 0.0],  # Doc 2
        [0.0, 0.0, 0.0],  # Doc 3 (zero vector)
    ])
    sample_query = np.array([0.5, 0.5, 0.0])
    top_results = retrieve_top_k(sample_query, sample_corpus, k=2)
    print(f"Top 2 matches for query {sample_query.tolist()}:")
    for doc_idx, score in top_results:
        print(f"  -> Doc {doc_idx} with score: {score:.4f}")

    print("\n--- 3. Running Large-Scale Benchmark (50,000 vectors) ---")
    n_docs, dim = 50_000, 1536
    print(f"Generating synthetic matrix: ({n_docs:,}, {dim}) in float32...")

    corpus = np.random.randn(n_docs, dim).astype(np.float32)
    query = np.random.randn(dim).astype(np.float32)
    query_2d = query.reshape(1, -1)

    # Precision verification against Scikit-Learn
    custom_scores = cosine_similarity_matrix(query, corpus)
    sklearn_scores = sklearn_cosine(query_2d, corpus)[0]

    precision_match = np.allclose(custom_scores, sklearn_scores, atol=1e-5)
    max_diff = float(np.max(np.abs(custom_scores - sklearn_scores)))
    print(f"Precision check vs Scikit-Learn : {precision_match} (Max absolute diff: {max_diff:.2e})")

    # Latency: Custom NumPy
    t0 = time.perf_counter()
    for _ in range(5):
        _ = cosine_similarity_matrix(query, corpus)
    custom_latency = ((time.perf_counter() - t0) / 5) * 1000
    print(f"Custom NumPy Matrix Cosine      : {custom_latency:.2f} ms")

    # Latency: Scikit-Learn
    t0 = time.perf_counter()
    for _ in range(5):
        _ = sklearn_cosine(query_2d, corpus)
    sklearn_latency = ((time.perf_counter() - t0) / 5) * 1000
    print(f"Scikit-Learn Pairwise Cosine    : {sklearn_latency:.2f} ms")

    # Latency: Pre-Normalized Unit Vectors (Dot product only)
    corpus_norm = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query)

    t0 = time.perf_counter()
    for _ in range(5):
        _ = corpus_norm @ query_norm
    fast_latency = ((time.perf_counter() - t0) / 5) * 1000
    print(f"Pre-Normalized Dot Product      : {fast_latency:.2f} ms")


if __name__ == "__main__":
    run_tests_and_benchmark()
