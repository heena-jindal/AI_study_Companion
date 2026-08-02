"""
rag_service.py

Handles RAG's two distinct phases, which we specifically corrected in Q5:

1. INDEXING (runs ONCE, when a file is uploaded) -- extract_text_from_pdf,
   chunk_text, embed_and_store. This is NOT part of the per-query flow.

2. RETRIEVAL (runs EVERY TIME a query happens) -- retrieve_relevant_chunks.
   This is what /explain and /quiz will call before talking to the LLM.

Kept in its own file, separate from llm_service.py, because these are
genuinely different responsibilities: this file talks to ChromaDB and the
embedding model, llm_service.py talks to Groq. Later, when /explain and
/quiz need retrieval, they'll import from BOTH files.
"""

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Persistent client -- stores vectors on disk in ./chroma_db so uploaded
# notes survive a server restart. Without this, you'd have to re-upload
# every time you stop and restart Flask.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# NOTE: explicitly setting "hnsw:space": "cosine" -- without this, Chroma
# defaults to L2 (euclidean) distance, not cosine similarity. We've been
# discussing cosine similarity conceptually (Q2), so this makes the code
# actually match that, and lets us compute a real similarity score below
# to filter out weak/irrelevant matches.
collection = chroma_client.get_or_create_collection(
    name="study_notes",
    metadata={"hnsw:space": "cosine"},
)

# Below this similarity score, a "match" is too weak to trust as real
# grounding -- this is the fix for the binary search case, where a loosely
# related chunk (a generic algorithm definition) still got returned as a
# "top match" even though it wasn't a genuinely useful match.
MIN_SIMILARITY = 0.3

# Local, open-source embedding model (this is the Q8 tradeoff in practice):
# no API key, no per-call cost, runs on your own machine. Downloads once
# (~80MB) on first run, then reused from local cache after that.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(file_stream) -> str:
    """
    Q6: a PDF is fonts + layout + page structure, not plain text -- this
    pulls the readable text out before anything else can happen to it.
    """
    reader = PdfReader(file_stream)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> list:
    """
    Fixed-size chunking (Q3), by WORD count rather than raw characters --
    splitting on words avoids literally cutting a word in half.

    `overlap` is Q7's fix in code form: each new chunk re-includes the
    last `overlap` words from the previous chunk, so a sentence sitting
    right on a chunk boundary still appears whole in at least one chunk,
    instead of being split across two chunks that each only have half of it.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap  # step forward, but re-include overlap words
    return chunks


def embed_and_store(chunks: list, source_name: str):
    """
    Embeds every chunk and stores it in ChromaDB. Per Q4: for each chunk,
    ChromaDB ends up storing the text itself, its embedding vector, a
    unique ID, and metadata (here: which file it came from, and its
    position) -- all four pieces, not just the raw text.
    """
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{source_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source_name, "chunk_index": i} for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )


def has_indexed_content() -> bool:
    """
    Checks if ANYTHING has been uploaded/indexed yet. Needed because
    /explain and /quiz will always try to use notes if available (per your
    choice) -- but on a fresh install, before any /upload call, the
    collection is empty. Querying an empty collection is either an error
    or meaningless, so we check this FIRST and skip retrieval entirely if
    nothing's been indexed, falling back to general LLM knowledge instead.
    """
    return collection.count() > 0


def retrieve_relevant_chunks(query: str, top_k: int = 3) -> list:
    """
    Runs on EVERY query, not at upload time (the Q5 distinction). Embeds
    the query using the SAME model used for the chunks -- this has to
    match, since vectors from two different embedding models aren't
    comparable to each other. Asks ChromaDB for the top_k chunks by
    cosine similarity (Q2), then filters out any that are too weak a
    match to actually be useful -- fixes the earlier bug where a barely-
    related chunk still got treated as "grounded" context.
    """
    available = collection.count()
    if available == 0:
        return []
    safe_top_k = min(top_k, available)

    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=safe_top_k,
        include=["documents", "distances"],
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # In cosine space, Chroma returns DISTANCE, where distance = 1 - similarity.
    # So similarity = 1 - distance. We only keep chunks that clear the bar --
    # everything else gets dropped instead of being force-used as "context."
    relevant = [
        doc
        for doc, dist in zip(documents, distances)
        if (1 - dist) >= MIN_SIMILARITY
    ]

    return relevant