"""
test_faithfulness.py

Standalone test for Part 6's evaluation piece -- NOT part of the running
app, same pattern as test_retrieval.py. Ties together Part 3 (retrieval),
Part 1 (generation), and evaluation_service.py: retrieves context for a
topic, generates an explanation from it, then checks whether that
explanation actually stayed faithful to the retrieved context.

Run from the PROJECT ROOT with: python tests/test_faithfulness.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so rag_service etc. are importable

from rag_service import retrieve_relevant_chunks, has_indexed_content
from llm_service import get_explanation
from evaluation_service import check_faithfulness

# Change this to a topic you know is covered in your uploaded notes
topic = "types of AI"

if not has_indexed_content():
    print("Nothing indexed yet -- run /upload first.")
else:
    chunks = retrieve_relevant_chunks(topic, top_k=3)
    context = "\n\n".join(chunks)

    print(f"Topic: {topic}\n")
    print(f"Retrieved {len(chunks)} chunk(s) as context.\n")

    answer = get_explanation(topic, context=context)
    print(f"Generated answer:\n{answer}\n")

    verdict = check_faithfulness(answer, context)
    print(f"Faithfulness check: {verdict['faithful']}")
    print(f"Reasoning: {verdict['explanation']}")