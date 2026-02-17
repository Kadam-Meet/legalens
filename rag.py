#!/usr/bin/env python3
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# -------- Load Env --------
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

# -------- LangChain --------
from langchain.llms import HuggingFacePipeline
from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# LOAD MODELS
# =====================================
print("🔄 Loading embedding + reranker models...")
embedder = SentenceTransformer(EMB_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

print("🔄 Loading Mistral 7B...")

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

# Optional LoRA
if Path(LORA_PATH).exists():
    model = PeftModel.from_pretrained(model, LORA_PATH, token=HF_TOKEN)
    print("✅ LoRA Loaded")

model.eval()

# Wrap model for LangChain
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=pipe)

print("✅ Model Ready\n")

# =====================================
# CHUNKING
# =====================================
def chunk_text(text, page):
    i = 0
    while i < len(text):
        yield {"page": page, "text": text[i:i+CHUNK_SIZE].strip()}
        i += CHUNK_SIZE - OVERLAP

# =====================================
# INDEXING
# =====================================
def index_documents():
    print("📚 Indexing documents...")
    files = list(Path(DATA_DIR).rglob("*"))

    meta = []
    corpus = []
    vectors = []
    cid = 0

    for f in tqdm(files):
        if not f.is_file():
            continue

        text = f.read_text(errors="ignore")

        for ch in chunk_text(text, 1):
            if not ch["text"]:
                continue

            chunk_path = Path(CHUNK_DIR) / f"chunk_{cid}.txt"
            chunk_path.write_text(ch["text"], encoding="utf-8")

            meta.append({
                "id": cid,
                "doc": f.name,
                "page": 1,
                "path": str(chunk_path)
            })

            corpus.append(ch["text"])
            vectors.append(ch["text"])
            cid += 1

    embeddings = embedder.encode(vectors, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    json.dump(corpus, open(f"{INDEX_DIR}/bm25.json", "w"))
    json.dump(meta, open(f"{INDEX_DIR}/meta.json", "w"))

    print(f"✅ Indexed {cid} chunks")

# =====================================
# HYBRID RETRIEVER
# =====================================
class HybridRetriever(BaseRetriever):

    def get_relevant_documents(self, query):
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
        _, dense_ids = index.search(
            q_emb.astype(np.float32),
            TOP_DENSE
        )

        candidates = list(set(
            bm25_ids.tolist() + dense_ids[0].tolist()
        ))

        texts, ids = [], []
        for cid in candidates:
            path = meta[cid]["path"]
            texts.append(Path(path).read_text(errors="ignore"))
            ids.append(cid)

        scores = reranker.predict(
            [[query, t[:512]] for t in texts]
        )

        ranked = sorted(
            zip(ids, scores),
            key=lambda x: x[1],
            reverse=True
        )[:FINAL_TOPK]

        documents = []
        for cid, _ in ranked:
            m = meta[cid]
            content = Path(m["path"]).read_text(errors="ignore")

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": m["doc"],
                        "page": m["page"]
                    }
                )
            )

        return documents

# =====================================
# MEMORY
# =====================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# =====================================
# CHAIN
# =====================================
retriever = HybridRetriever()

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# =====================================
# MAIN LOOP
# =====================================
def main():
    print("🧠 Conversational Legal RAG Ready\n")

    while True:
        query = input(">> ")

        if query.lower() == "exit":
            break

        result = qa_chain({"question": query})

        print("\n📄 RESPONSE\n")
        print(result["answer"])

        print("\n📚 SOURCES\n")
        for doc in result["source_documents"]:
            print(f"{doc.metadata['source']} (Page {doc.metadata['page']})")

if __name__ == "__main__":
    main()
