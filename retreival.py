def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    query_embedding = embed(query)  # your existing sentence-transformers call
    scores = cosine_similarity_search(query_embedding, your_vector_store)
    top_chunks = get_top_k(scores, top_k)
    return top_chunks  # list of text strings