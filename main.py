import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Render injects env vars directly, no .env file needed

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from groq import Groq
from chroma_setup import add_fact, query_facts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your real Vercel domain before real launch
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND = os.environ.get("INFERENCE_BACKEND", "groq")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

SIMILARITY_THRESHOLD = 0.4  # placeholder — tune once real data + real queries exist


def build_rag_prompt(user_query: str, context_chunks: list[str]) -> str:
    context_block = "\n\n".join(context_chunks)
    return f"""Use the following context to answer the question. If the context doesn't contain the answer, say you don't know — do not make up information.

Context:
{context_block}

Question: {user_query}

Answer:"""


async def call_model(prompt: str) -> str:
    if BACKEND == "ollama":
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.1:8b", "prompt": prompt, "stream": False},
                timeout=60.0
            )
            return r.json()["response"]
    r = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        timeout=15.0
    )
    return r.choices[0].message.content


@app.get("/generate")
async def generate(prompt: str = ""):
    prompt = prompt.strip()

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt cannot be empty")

    if len(prompt) > 500:
        raise HTTPException(status_code=400, detail="prompt too long — keep under 500 characters")

    try:
        results = query_facts(prompt, n=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"retrieval failed: {str(e)}")

    good_matches = [(text, source) for score, text, source in results if score >= SIMILARITY_THRESHOLD]

    if not good_matches:
        return {"response": "I don't have information on that yet.", "sources": []}

    context_texts = [text for text, source in good_matches]
    full_prompt = build_rag_prompt(prompt, context_texts)

    try:
        response = await call_model(full_prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail="model backend unavailable — try again shortly")

    return {"response": response, "sources": [{"text": t, "source": s} for t, s in good_matches]}

@app.get("/search")
def search(q: str):
    return {"matches": query_facts(q)}


@app.get("/tokentest")
def tokentest(text: str):
    r = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": text}]
    )
    return {"text": text, "prompt_tokens": r.usage.prompt_tokens}
