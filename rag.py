#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# -------- OCR ----------
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# -------- Retrieval ----------
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------- LLM ----------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------- LangChain ----------
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# ================= CONFIG =================
DATA_DIR = "data"
INDEX_DIR = "index_data"
CHUNK_DIR = f"{INDEX_DIR}/chunks"

EMB_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "models/mistral_finetuned"

# Load HF token from environment variable for security
HF_TOKEN = os.getenv('HF_TOKEN', '')

CHUNK_SIZE = 3000
OVERLAP = 300
TOP_BM25 = 20
TOP_DENSE = 20
FINAL_TOPK = 5
# If true, when data/ is empty the script will build an index from training_json/.
# Set to False in production to avoid returning canned training examples for all queries.
FALLBACK_TO_TRAINING = False
# =========================================


# ------------------- INIT -------------------
os.makedirs(CHUNK_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer(EMB_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

# ---- Tokenizer (must come from base model) ----
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN,
    use_fast=True
)

# ---- 4-bit config (MUST match training) ----
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ---- Load base model in 4-bit ----
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

# ---- Attach LoRA adapter ----
# ---- Attach LoRA adapter (optional) ----
from pathlib import Path as _Path

_lora = LORA_PATH
if _Path(_lora).exists():
    try:
        model = PeftModel.from_pretrained(model, _lora, token=HF_TOKEN)
        print(f"✅ LoRA adapter loaded from {_lora}")
    except Exception as e:
        print(f"⚠️ Failed loading LoRA adapter from {_lora}: {e}")
else:
    # Attempt to load from Hugging Face hub (repo id). If it fails, continue with base model.
    try:
        model = PeftModel.from_pretrained(model, _lora, token=HF_TOKEN)
        print(f"✅ LoRA adapter loaded from hub '{_lora}'")
    except Exception as e:
        print(f"⚠️ LoRA adapter not found at {_lora}; continuing with base model. ({e})")

model.eval()

memory = ConversationBufferMemory(return_messages=True)

print("✅ Model loaded on:", next(model.parameters()).device)


# ---------------- OCR ----------------
def extract_text(path):
    ext = path.suffix.lower()
    pages = []

    if ext == ".pdf":
        try:
            with pdfplumber.open(path) as pdf:
                for i, p in enumerate(pdf.pages, 1):
                    pages.append({"page": i, "text": p.extract_text() or ""})
        except:
            images = convert_from_path(path)
            for i, img in enumerate(images, 1):
                pages.append({"page": i, "text": pytesseract.image_to_string(img)})

    elif ext in [".txt", ".md"]:
        pages.append({"page": 1, "text": path.read_text(errors="ignore")})

    return pages


# ---------------- Chunk ----------------
def chunk_text(text, page):
    i = 0
    while i < len(text):
        yield {"page": page, "text": text[i:i+CHUNK_SIZE].strip()}
        i += CHUNK_SIZE - OVERLAP


# ---------------- INDEX ----------------
def index_documents():
    files = list(Path(DATA_DIR).rglob("*")) if Path(DATA_DIR).exists() else []

    # If no user data found, optionally fallback to training_json samples
    if not files:
        if FALLBACK_TO_TRAINING:
            print("⚠️ No files in data/ — using training_json/ as fallback (FALLBACK_TO_TRAINING=True)...")
            tj_path = Path("training_json")
            if not tj_path.exists():
                raise RuntimeError("❌ No files in data/ and no training_json/ fallback available.")

            files = list(tj_path.glob("*.json"))
            if not files:
                raise RuntimeError("❌ training_json/ is empty — add files under data/ or set FALLBACK_TO_TRAINING accordingly.")
        else:
            raise RuntimeError("❌ No files in data/. Add documents to data/ or enable FALLBACK_TO_TRAINING to use training samples.")

    meta = []
    corpus = []
    vectors = []

    cid = 0
    for f in tqdm(files, desc="Indexing"):
        if f.suffix.lower() == ".json" and f.parent.name == "training_json":
            # load JSON array of training examples and convert to text chunks
            try:
                j = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue

            for i, item in enumerate(j):
                instr = item.get("instruction", "")
                ctx = item.get("context", "")
                out = item.get("output", {})
                out_text = json.dumps(out, ensure_ascii=False) if not isinstance(out, str) else out
                combined = f"Instruction: {instr}\nContext: {ctx}\nOutput: {out_text}"

                for ch in chunk_text(combined, i+1):
                    if not ch["text"]:
                        continue

                    chunk_path = Path(CHUNK_DIR) / f"chunk_{cid}.txt"
                    chunk_path.write_text(ch["text"], encoding="utf-8")

                    meta.append({
                        "id": cid,
                        "doc": f.name,
                        "page": i+1,
                        "path": str(chunk_path)
                    })

                    corpus.append(ch["text"])
                    vectors.append(ch["text"])
                    cid += 1
        else:
            for p in extract_text(f):
                for ch in chunk_text(p["text"], p["page"]):
                    if not ch["text"]:
                        continue

                    chunk_path = Path(CHUNK_DIR) / f"chunk_{cid}.txt"
                    chunk_path.write_text(ch["text"], encoding="utf-8")

                    meta.append({
                        "id": cid,
                        "doc": f.name,
                        "page": p["page"],
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


# ---------------- RETRIEVE ----------------
def retrieve(query):
    # Ensure index and corpus exist; if not, (re)build them
    if not Path(f"{INDEX_DIR}/faiss.index").exists() or not Path(f"{INDEX_DIR}/bm25.json").exists() or not Path(f"{INDEX_DIR}/meta.json").exists():
        print("⚠️ Index or corpus missing — building index now...")
        index_documents()

    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    corpus = json.load(open(f"{INDEX_DIR}/bm25.json"))
    meta = json.load(open(f"{INDEX_DIR}/meta.json"))

    if not corpus:
        print("⚠️ BM25 corpus is empty — attempting to rebuild index...")
        index_documents()
        index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
        corpus = json.load(open(f"{INDEX_DIR}/bm25.json"))
        meta = json.load(open(f"{INDEX_DIR}/meta.json"))

    if not corpus:
        raise RuntimeError("No documents found after indexing. Add files under the data/ directory and retry.")

    bm25 = BM25Okapi([c.split() for c in corpus])
    bm25_ids = np.argsort(bm25.get_scores(query.split()))[::-1][:TOP_BM25]

    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, dense_ids = index.search(q_emb.astype(np.float32), TOP_DENSE)

    candidates = list(set(bm25_ids.tolist() + dense_ids[0].tolist()))

    texts, ids = [], []
    for cid in candidates:
        path = meta[cid]["path"]
        if Path(path).exists():
            texts.append(Path(path).read_text(errors="ignore"))
            ids.append(cid)

    scores = reranker.predict([[query, t[:512]] for t in texts])
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:FINAL_TOPK]

    return ranked, meta


# ---------------- GENERATE ----------------
def generate(context, user_query, mode):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal AI assistant.

TASK: {mode}

CONTEXT:
{context}

USER QUERY:
{question}

Return clean, structured JSON output.
"""
    )

    final_prompt = prompt.format(
        context=context,
        question=user_query,
        mode=mode
    )

    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2
        )

        generated_tokens = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ---------------- MAIN ----------------
def main():
    if not Path(f"{INDEX_DIR}/faiss.index").exists():
        index_documents()

    print("\n🧠 Legal RAG Assistant Ready")

    while True:
        query = input(">> ").strip()
        if query.lower() == "exit":
            break

        if query.startswith("summary:"):
            mode = "Summarization"
            q = query.replace("summary:", "").strip()
        elif query.startswith("chat:"):
            mode = "Question Answering"
            q = query.replace("chat:", "").strip()
        else:
            mode = "Both Summarization + QA"
            q = query.replace("both:", "").strip()

        results, meta = retrieve(q)

        context = ""
        for cid, _ in results:
            m = meta[cid]
            text = Path(m["path"]).read_text(errors="ignore")
            context += f"\n--- {m['doc']} page {m['page']} ---\n{text}"

        response = generate(context, q, mode)
        print("\n📄 RESPONSE\n")
        print(response)


if __name__ == "__main__":
    main()