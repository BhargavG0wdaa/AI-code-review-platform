"""
Phase 5: repository RAG (Retrieval-Augmented Generation).

The reviewer only sees the diff. That's why the verifier keeps refusing real
bugs ("I can't see how this function is called"). RAG fixes that: we index the
whole repo into a vector database (Qdrant), then for a given PR we RETRIEVE the
most relevant code (e.g. the callers of a changed function) and feed it in as
extra context.

Pieces:
  - fastembed  -> turns code text into vectors (no PyTorch; ONNX, fast)
  - Qdrant     -> stores vectors and finds the nearest ones to a query
  - ast        -> splits each file into function/class chunks to index
"""

import ast

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://localhost:6333"
# bge-small: 384-dim, small + fast, good at code/text. Downloaded once on first use.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

_embedder: TextEmbedding | None = None


def _get_embedder() -> TextEmbedding:
    """Load the embedding model once and reuse it (the first call downloads it)."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def chunk_python(path: str, content: str) -> list[dict]:
    """Split a Python file into retrievable chunks — one per top-level function
    or class. Retrieving a whole function is far more useful as context than a
    random window of lines. Files that don't parse fall back to one whole chunk.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return [{"path": path, "name": path, "start": 1, "code": content}]

    lines = content.splitlines()
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start) or start
            code = "\n".join(lines[start - 1:end])
            chunks.append({"path": path, "name": node.name, "start": start, "code": code})
    if not chunks:  # module-level script with no defs — index the whole thing
        chunks.append({"path": path, "name": path, "start": 1, "code": content})
    return chunks


class RepoIndex:
    """A vector index of one repository's code, stored in a Qdrant collection."""

    def __init__(self, collection: str, url: str = QDRANT_URL):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.embedder = _get_embedder()

    def index(self, files: dict) -> int:
        """files: {path: content}. Rebuilds the collection from scratch and
        returns the number of chunks indexed."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        chunks = []
        for path, content in files.items():
            if path.endswith(".py") and content:
                chunks += chunk_python(path, content)
        if not chunks:
            return 0
        vectors = list(self.embedder.embed([c["code"] for c in chunks]))
        points = [
            PointStruct(id=i, vector=v.tolist(), payload=chunks[i])
            for i, v in enumerate(vectors)
        ]
        self.client.upsert(self.collection, points=points)
        return len(points)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the payloads of the k chunks most relevant to `query`."""
        qv = next(iter(self.embedder.embed([query]))).tolist()
        resp = self.client.query_points(self.collection, query=qv, limit=k)
        return [p.payload for p in resp.points]


if __name__ == "__main__":
    # Standalone smoke test: index a tiny repo and retrieve by meaning.
    repo = {
        "auth.py": (
            "def hash_password(pw):\n"
            "    return sha256(pw.encode()).hexdigest()\n\n"
            "def login(user, pw):\n"
            "    return hash_password(pw) == user.pw_hash\n"
        ),
        "math_utils.py": (
            "def divide(a, b):\n"
            "    return a / b\n\n"
            "def average(nums):\n"
            "    return divide(sum(nums), len(nums))\n"
        ),
    }
    idx = RepoIndex("smoke_test")
    n = idx.index(repo)
    print(f"indexed {n} chunks")
    print("\nquery: 'who calls divide / division by zero'")
    for hit in idx.search("division by zero in divide", k=2):
        print(f"  -> {hit['path']}::{hit['name']} (line {hit['start']})")
