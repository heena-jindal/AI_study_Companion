"""
test_retrieval.py

Quick standalone test -- NOT part of the app, just a way to check that
retrieve_relevant_chunks() actually returns sensible results before we
wire it into /explain and /quiz. Run from the PROJECT ROOT with:
python tests/test_retrieval.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so rag_service etc. are importable

from rag_service import retrieve_relevant_chunks

# Change this to something you know your DSA notes actually cover
query = "types of AI and generative AI"

results = retrieve_relevant_chunks(query, top_k=3)

print(f"Query: {query}\n")
for i, chunk in enumerate(results, 1):
    print(f"--- Chunk {i} ---")
    print(chunk)
    print()