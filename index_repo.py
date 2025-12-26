import os
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer


CONFIG_PATH = Path(__file__).parent / "config.json"  # configuring
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

REPO_PATH = Path(config["repo_path"])
CHUNK_SIZE = config["chunk_size"]
CHUNK_OVERLAP = config["chunk_overlap"]


OUT_DIR = Path(__file__).parent / "faiss_db"
OUT_DIR.mkdir(exist_ok=True)
INDEX_PATH = OUT_DIR / "repo.index"
DOCS_PATH = OUT_DIR / "docs.json"


ALLOWED_EXT = {".py"}
SKIP_DIRS = {".venv", "env", "node_modules", "dist", "build", "__pycache__", "venv"}

def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if Path(name).suffix in ALLOWED_EXT:
                yield Path(dirpath) / name

def chunk_text(text: str, chunk_size: int, overlap: int):
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        yield text[start:end]
        if end == n:
            break
        start = end - overlap


def index_repo():
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    texts = []  # chunk texts for embedding

    print(f"Scanning repo at {REPO_PATH} ...")
    file_count = 0

    for file_path in iter_files(REPO_PATH):
        file_count += 1
        if file_count % 50 == 0:
            print(f"  Scanned {file_count} files...")

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for i, chunk in enumerate(chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)):
            docs.append({
                "path": str(file_path),
                "chunk_id": i,
                "text": chunk
            })
            texts.append(chunk)

    if not docs:
        print("No documents found to index.")
        return

    print(f"Total prepared chunks: {len(docs)}. Computing embeddings (can be slow on CPU)...")

    emb = model.encode(  # embeddings computing
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = emb.shape[1]
    print(f"Embedding dim: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print("Saving FAISS index and docs mapping...")
    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    print(f"Done. Saved:\n- {INDEX_PATH}\n- {DOCS_PATH}")


if __name__ == "__main__":
    index_repo()
