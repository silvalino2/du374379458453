def build_rag_prompt(user_query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(context_chunks)
    prompt = f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know — do not make up information.

Context:
{context_block}

Question: {user_query}

Answer:"""
    return prompt