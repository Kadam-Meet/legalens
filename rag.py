#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

# -------- Torch / HF --------
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# -------- Retrieval --------
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------- LCEL --------
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# =====================================
# CONFIG
# =====================================
DATA_DIR = "data"
INDEX_DIR = "index_data"
CHUNK_DIR = f"{INDEX_DIR}/chunks"

EMB_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "models/mistral_finetuned"

CHUNK_SIZE = 3000
OVERLAP = 300
TOP_BM25 = 20
TOP_DENSE = 20
FINAL_TOPK = 5

os.makedirs(CHUNK_DIR, exist_ok=True)

# =====================================
# LOAD MODELS
# =====================================
print("🔄 Loading models...")

embedder = SentenceTransformer(EMB_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    use_fast=True
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

if Path(LORA_PATH).exists():
    model = PeftModel.from_pretrained(model, LORA_PATH, token=HF_TOKEN)
    print("✅ LoRA Loaded")

model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    return_full_text=False   # VERY IMPORTANT
)


llm = HuggingFacePipeline(pipeline=pipe)

print("✅ Model Ready\n")

# =====================================
# INDEXING
# =====================================
def chunk_text(text):
    i = 0
    while i < len(text):
        yield text[i:i+CHUNK_SIZE].strip()
        i += CHUNK_SIZE - OVERLAP


def index_documents():
    print("📚 Indexing documents...")
    files = list(Path(DATA_DIR).rglob("*"))

    meta, corpus, vectors = [], [], []
    cid = 0

    for f in tqdm(files):
        if not f.is_file():
            continue

        text = f.read_text(errors="ignore")

        for chunk in chunk_text(text):
            if not chunk:
                continue

            chunk_path = Path(CHUNK_DIR) / f"chunk_{cid}.txt"
            chunk_path.write_text(chunk, encoding="utf-8")

            meta.append({
                "id": cid,
                "doc": f.name,
                "path": str(chunk_path)
            })

            corpus.append(chunk)
            vectors.append(chunk)
            cid += 1

    embeddings = embedder.encode(vectors, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    json.dump(corpus, open(f"{INDEX_DIR}/bm25.json", "w"))
    json.dump(meta, open(f"{INDEX_DIR}/meta.json", "w"))

    print(f"✅ Indexed {cid} chunks")

# =====================================
# HYBRID RETRIEVER FUNCTION
# =====================================
def hybrid_retrieve(query):

    if not Path(f"{INDEX_DIR}/faiss.index").exists():
        index_documents()

    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    corpus = json.load(open(f"{INDEX_DIR}/bm25.json"))
    meta = json.load(open(f"{INDEX_DIR}/meta.json"))

    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_ids = np.argsort(
        bm25.get_scores(query.split())
    )[::-1][:TOP_BM25]

    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, dense_ids = index.search(q_emb.astype(np.float32), TOP_DENSE)

    candidates = list(set(bm25_ids.tolist() + dense_ids[0].tolist()))

    texts, ids = [], []
    for cid in candidates:
        path = meta[cid]["path"]
        texts.append(Path(path).read_text(errors="ignore"))
        ids.append(cid)

    scores = reranker.predict([[query, t[:512]] for t in texts])

    ranked = sorted(
        zip(ids, scores),
        key=lambda x: x[1],
        reverse=True
    )[:FINAL_TOPK]

    docs = []
    for cid, _ in ranked:
        m = meta[cid]
        content = Path(m["path"]).read_text(errors="ignore")
        docs.append(content)

    return "\n\n".join(docs)

# Wrap retriever as Runnable
retriever_runnable = RunnableLambda(
    lambda inputs: hybrid_retrieve(inputs["question"])
)

# =====================================
# PROMPT
# =====================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a legal AI assistant. If the question is not related to legal topics, respond normally without using context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# =====================================
# LCEL CHAIN
# =====================================
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda inputs: hybrid_retrieve(inputs["question"])
    )
    | prompt
    | llm
)

# =====================================
# MEMORY (LCEL WAY)
# =====================================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# =====================================
# MAIN LOOP
# =====================================
def main():
    print("🧠 Pure LCEL Legal RAG Ready\n")

    session_id = "legal-session"

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        response = chain_with_memory.invoke(
            {"question": query},
            config={"configurable": {"session_id": session_id}}
        )

        print("\n📄 RESPONSE\n")
        print(response)

if __name__ == "__main__":
    main()
