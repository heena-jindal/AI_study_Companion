"""
app.py

The Flask backend. Right now it does exactly one thing: receives a topic
from the user, passes it to llm_service.get_explanation(), and sends the
explanation back as JSON.

Run it with:  python app.py
Test it with: curl -X POST http://127.0.0.1:5000/explain \
                    -H "Content-Type: application/json" \
                    -d '{"topic": "sliding window technique"}'
"""

from flask import Flask, request, jsonify
from llm_service import get_explanation, get_quiz
from rag_service import (
    extract_text_from_pdf,
    chunk_text,
    embed_and_store,
    retrieve_relevant_chunks,
    has_indexed_content,
)
from tracking_service import init_db, record_attempt, get_weak_topics
from agent_service import run_agent

app = Flask(__name__)
init_db()  # creates the quiz_attempts table if it doesn't exist yet


@app.route("/upload", methods=["POST"])
def upload():
    """
    The ONE-TIME indexing endpoint (Q5). Accepts a PDF, extracts its text,
    chunks it, embeds every chunk, and stores it all in ChromaDB. This has
    to run and finish successfully BEFORE retrieval can find anything --
    /explain and /quiz will search whatever's already been indexed here.
    """
    if "file" not in request.files:
        return jsonify(
            {"error": "No file provided. Send it as form-data with key 'file'"}
        ), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported right now"}), 400

    try:
        text = extract_text_from_pdf(file)

        if not text.strip():
            return jsonify(
                {"error": "Could not extract any text from this PDF -- "
                          "it may be scanned/image-based rather than text-based"}
            ), 400

        chunks = chunk_text(text)
        embed_and_store(chunks, source_name=file.filename)

    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    return jsonify({
        "message": f"Successfully indexed {file.filename}",
        "chunks_created": len(chunks),
    })


@app.route("/explain", methods=["POST"])
def explain():
    data = request.get_json()

    # Basic validation -- always check what the user actually sent before
    # passing it to the LLM. This isn't an LLM concept, it's just good
    # backend hygiene, but it matters just as much here.
    if not data or "topic" not in data or not data["topic"].strip():
        return jsonify({"error": "Please provide a non-empty 'topic' field"}), 400

    topic = data["topic"].strip()

    # Per your choice: ALWAYS use notes if any are indexed. has_indexed_content()
    # guards against querying an empty ChromaDB collection before any /upload
    # has happened -- in that case we just fall back to general knowledge.
    context = None
    if has_indexed_content():
        relevant_chunks = retrieve_relevant_chunks(topic, top_k=3)
        if relevant_chunks:
            context = "\n\n".join(relevant_chunks)

    try:
        explanation = get_explanation(topic, context=context)
    except Exception as e:
        # If the Groq API call fails (bad key, rate limit, network issue),
        # we don't want the whole server to crash -- we return a clean
        # error instead.
        return jsonify({"error": f"LLM call failed: {str(e)}"}), 500

    return jsonify({
        "topic": topic,
        "explanation": explanation,
        "grounded_in_notes": context is not None,  # lets the frontend show this later
    })


@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.get_json()

    if not data or "topic" not in data or not data["topic"].strip():
        return jsonify({"error": "Please provide a non-empty 'topic' field"}), 400

    topic = data["topic"].strip()
    num_questions = data.get("num_questions", 3)  # optional, defaults to 3

    # Bounds check -- reject nonsensical values BEFORE spending an API call
    # on them. Note this check happens before the try/except below: it's not
    # an LLM failure, it's bad input, so it gets its own clear 400 error
    # instead of being lumped in with API errors.
    if not isinstance(num_questions, int) or isinstance(num_questions, bool):
        return jsonify({"error": "'num_questions' must be an integer"}), 400

    if num_questions < 1 or num_questions > 10:
        return jsonify(
            {"error": "'num_questions' must be between 1 and 10"}
        ), 400

    try:
        context = None
        if has_indexed_content():
            relevant_chunks = retrieve_relevant_chunks(topic, top_k=3)
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)

        quiz_data = get_quiz(topic, num_questions, context=context)
        quiz_data["grounded_in_notes"] = context is not None
    except ValueError as e:
        # This is OUR validation catching a logical mismatch (correct_answer
        # not actually in options) -- schema-valid but logically wrong.
        return jsonify({"error": f"Quiz validation failed: {str(e)}"}), 500
    except Exception as e:
        # Catches API failures: rate limits, network issues, etc.
        return jsonify({"error": f"LLM call failed: {str(e)}"}), 500

    return jsonify(quiz_data)


@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    """
    This is the endpoint that answers Q6: /quiz alone never finds out
    whether the user got a question right. The frontend calls THIS
    endpoint after the user picks an answer, sending back what they chose
    -- we compare it against correct_answer and persist the result via
    tracking_service.record_attempt(). This one write is what makes
    "weak topic" tracking possible at all.
    """
    data = request.get_json()

    required_fields = ["topic", "question", "difficulty", "user_answer", "correct_answer"]
    if not data or any(field not in data for field in required_fields):
        return jsonify({
            "error": f"Please provide all required fields: {', '.join(required_fields)}"
        }), 400

    is_correct = data["user_answer"] == data["correct_answer"]

    try:
        record_attempt(
            topic=data["topic"],
            question=data["question"],
            difficulty=data["difficulty"],
            is_correct=is_correct,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to record attempt: {str(e)}"}), 500

    return jsonify({
        "is_correct": is_correct,
        "correct_answer": data["correct_answer"],
    })


@app.route("/weak-topics", methods=["GET"])
def weak_topics():
    """
    The actual payoff of Part 4: reads back everything /submit-answer has
    recorded, groups it by topic, and returns topics sorted weakest-first.
    This is what turns raw stored rows into something genuinely useful --
    "what should I actually study more?" -- rather than just data sitting
    unused in a database file.
    """
    try:
        topics = get_weak_topics(min_attempts=2)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch weak topics: {str(e)}"}), 500

    return jsonify({"weak_topics": topics})


@app.route("/agent", methods=["POST"])
def agent():
    """
    The Part 5 entry point. Unlike /explain, /quiz, /weak-topics (which
    you call directly and chain yourself), THIS endpoint takes a single
    natural-language message and lets the agent decide what to do --
    exactly the "quiz me on my weak topics" example from the diagram.
    """
    data = request.get_json()

    if not data or "message" not in data or not data["message"].strip():
        return jsonify({"error": "Please provide a non-empty 'message' field"}), 400

    message = data["message"].strip()

    try:
        result = run_agent(message)
    except Exception as e:
        return jsonify({"error": f"Agent run failed: {str(e)}"}), 500

    return jsonify(result)  # {"message": ..., "quiz_data": ... or None}


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "AI Study Companion backend is running"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)