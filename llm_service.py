"""
llm_service.py

This file is the ONLY place in the whole project that talks to the LLM.
Everything else (Flask routes, later the frontend) calls functions from here.
Keeping this separate means later when you add RAG or agents, you're not
rewriting Flask routes -- you're just changing what happens inside these
functions.

CONCEPT CHECKPOINT (read after you get this working, not before):
- We are calling a "chat completion" endpoint. Every LLM API call is
  stateless -- the model has no memory between calls. That's why we build
  the full "messages" list fresh every single time.
- The "system" message sets behavior/personality for the whole conversation.
  The "user" message is the actual input. This split matters a lot in
  prompt engineering.
"""

import os
import json
from typing import List
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Loads variables from your .env file (like GROQ_API_KEY) into the environment
load_dotenv()

# One client, reused everywhere. Reading the key from an env var (not hardcoding
# it) means this code is safe to push to GitHub without leaking your key.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# NOTE: llama-3.3-70b-versatile is deprecated on Groq's free/developer tier,
# shutdown date 08/16/26. Using openai/gpt-oss-120b instead, Groq's recommended
# replacement -- also one of the models that supports strict Structured
# Outputs (see get_quiz below).
MODEL = "openai/gpt-oss-120b"


# ---- Schema for the quiz feature ----
# This is the "decide the fields before writing the prompt" step we discussed.
# `model_config = ConfigDict(extra="forbid")` makes Pydantic generate
# "additionalProperties": false in the JSON Schema -- Groq's strict mode
# REQUIRES this on every object, plus every field being required. Without
# this, strict mode either gets rejected or silently falls back to
# best-effort matching -- which is exactly the bug we just hit.

class QuizQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str  # must exactly match one of the 4 strings in "options"
    difficulty: str  # "easy" | "medium" | "hard"


class QuizResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: List[QuizQuestion]


def get_explanation(topic: str, context: str = None) -> str:
    """
    Takes a topic or a chunk of notes, returns a simple explanation.

    If `context` is provided (retrieved chunks from your uploaded notes),
    the prompt changes to ground the answer in that context specifically,
    instead of the LLM's general training knowledge -- this is Q9's
    "prompt injection of context" in actual code. If context is None
    (nothing indexed yet), it falls back to exactly Part 1's original
    behavior -- general knowledge, no notes involved.
    """

    if context:
        system_prompt = (
            "You are a patient tutor explaining concepts to a college student "
            "preparing for exams and interviews. You will be given CONTEXT "
            "retrieved from the student's own notes. Base your explanation "
            "primarily on this context. If the context doesn't fully cover "
            "the topic, you may supplement with general knowledge, but say "
            "so explicitly. Explain clearly and simply, using short "
            "sentences. Keep the explanation under 200 words. Do not use "
            "headers or bullet lists -- write it as plain explanatory "
            "paragraphs."
        )
        user_prompt = f"CONTEXT FROM NOTES:\n{context}\n\nExplain this topic: {topic}"
    else:
        system_prompt = (
            "You are a patient tutor explaining concepts to a college student "
            "preparing for exams and interviews. Explain clearly and simply, "
            "using short sentences and a concrete example where useful. "
            "Keep the explanation under 200 words. Do not use headers or bullet "
            "lists -- write it as plain explanatory paragraphs."
        )
        user_prompt = f"Explain this topic: {topic}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,  # lower = more focused/consistent, higher = more varied
        max_tokens=400,
    )

    # response.choices[0] is the model's first (and only, since we didn't
    # ask for multiple) completion. .message.content is the actual text.
    return response.choices[0].message.content


def get_quiz(topic: str, num_questions: int = 3, context: str = None) -> dict:
    """
    Takes a topic, returns a dict matching QuizResponse's schema.

    Same context-injection pattern as get_explanation() -- if notes have
    been indexed and relevant chunks were retrieved, the quiz gets
    generated FROM that material specifically, instead of the LLM's
    general knowledge. This is what fixes the "1949 revolt" ambiguity
    problem you hit earlier -- if your notes are about a specific event,
    the quiz gets grounded in that specific context instead of the LLM
    guessing which globally-famous interpretation you meant.
    """

    system_prompt = (
        "You are a quiz generator for a college student studying for exams "
        "and interviews. Generate quiz questions that test real understanding, "
        "not just definition recall."
    )

    if context:
        user_prompt = (
            f"CONTEXT FROM NOTES:\n{context}\n\n"
            f"Generate {num_questions} multiple-choice quiz questions based "
            f"on this context, about: {topic}. Vary the difficulty across "
            f"easy, medium, and hard. Make sure correct_answer is an exact "
            f"copy of one of the 4 strings in options."
        )
    else:
        user_prompt = (
            f"Generate {num_questions} multiple-choice quiz questions about: {topic}. "
            f"Vary the difficulty across easy, medium, and hard. Make sure "
            f"correct_answer is an exact copy of one of the 4 strings in options."
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "quiz_response",
                "strict": True,  # <-- the fix: without this, matching was best-effort only
                "schema": QuizResponse.model_json_schema(),
            },
        },
    )

    raw_json = response.choices[0].message.content
    quiz_data = json.loads(raw_json)  # safe: strict mode guarantees valid JSON

    # This is the extra validation layer we discussed in Q1 (Part 2) --
    # the schema guarantees each question HAS a correct_answer field and
    # an options field, but NOT that options has exactly 4 items --
    # array length isn't enforced by strict mode's constrained decoding,
    # only object-level required fields are. Confirmed by a real dropped-
    # option bug found during Part 5 testing. So we check both things
    # ourselves: right number of options, AND correct_answer actually
    # being one of them.
    for q in quiz_data["questions"]:
        if len(q["options"]) != 4:
            raise ValueError(
                f"Expected 4 options, got {len(q['options'])} for "
                f"question: {q['question']}"
            )
        if q["correct_answer"] not in q["options"]:
            raise ValueError(
                f"correct_answer '{q['correct_answer']}' not found in "
                f"options for question: {q['question']}"
            )

    return quiz_data