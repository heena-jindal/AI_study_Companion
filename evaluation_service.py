"""
evaluation_service.py

Part 6, Q1: checking retrieval quality (Part 3's MIN_SIMILARITY) only
proves relevant chunks were FOUND -- not that the LLM actually used them
correctly when generating an answer. This file adds that second,
separate check: "does this answer actually match what the context says?"

This is intentionally NOT wired into every /explain or /quiz call by
default -- per Q4 (cost/latency tradeoffs), running an extra LLM call to
judge every single response would double your API usage for something
that's mainly useful for spot-checking and debugging, not for every
live request. Use it selectively, the way you'd write a test.
"""

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


def check_faithfulness(answer: str, context: str) -> dict:
    """
    LLM-as-judge (Q1): makes a SEPARATE call asking the model to check
    whether `answer` is actually supported by `context`, or whether it
    contains claims the context doesn't back up -- Q2's definition of a
    RAG-specific hallucination.

    Returns {"faithful": bool, "explanation": str} -- a verdict plus the
    reasoning, so a failure is debuggable, not just a flat true/false.
    """
    judge_prompt = (
        "You are a strict fact-checker. Below is a CONTEXT and an ANSWER "
        "that was supposedly generated using that context. Determine if "
        "every claim in the ANSWER is actually supported by the CONTEXT. "
        "If the answer adds information, facts, or claims that are NOT "
        "present in or inferable from the context, that counts as "
        "unfaithful (hallucinated), even if the added information "
        "happens to be true in general.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Respond with exactly one line starting with 'FAITHFUL: yes' or "
        "'FAITHFUL: no', followed by a one-sentence explanation."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,  # this is a judgment call, not a creative task --
                        # we want maximum consistency, not variety
        max_tokens=1000,  # generous budget: gpt-oss-120b is a REASONING
        # model -- it spends tokens "thinking" (visible separately as
        # response.choices[0].message.reasoning) BEFORE writing the
        # actual visible verdict. A complex judgment task over a full
        # context + answer can burn through a small budget on reasoning
        # alone, leaving nothing for the real answer -- which is exactly
        # what happened with the original max_tokens=150.
    )

    verdict_text = response.choices[0].message.content
    finish_reason = response.choices[0].finish_reason

    if not verdict_text:
        # Don't silently call this "unfaithful" -- an empty response
        # means the judge never actually reached a verdict (most likely
        # cut off mid-reasoning), which is a DIFFERENT problem than the
        # answer actually failing the check.
        return {
            "faithful": None,
            "explanation": (
                f"Judge returned no verdict (finish_reason='{finish_reason}'). "
                f"Likely ran out of tokens mid-reasoning -- try raising max_tokens further."
            ),
        }

    is_faithful = "faithful: yes" in verdict_text.lower()

    return {
        "faithful": is_faithful,
        "explanation": verdict_text,
    }