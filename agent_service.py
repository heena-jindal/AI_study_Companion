"""
agent_service.py

Part 5: turns everything built in Parts 1-4 into TOOLS an LLM chooses
between, instead of endpoints you call and chain yourself. Built with
LangGraph, implementing exactly the ReAct loop from Q2 and the diagram
we drew together: reason -> act (call a tool) -> observe -> loop back to
reason, until the agent decides it has enough to respond.
"""

import json
import logging
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_service import get_explanation, get_quiz
from rag_service import retrieve_relevant_chunks, has_indexed_content
from tracking_service import get_weak_topics

# Part 6, Q5: a real record of every tool call the agent makes -- this is
# what turns "the agent decided something" into evidence you can actually
# look back at, for debugging AND for explaining your project honestly
# in an interview, the same way you've been explaining today's real bugs
# to me with concrete logs instead of vague descriptions.
logging.basicConfig(
    filename="agent_decisions.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)
agent_logger = logging.getLogger("agent_decisions")


def _get_rag_context(topic: str) -> Optional[str]:
    """
    Same RAG-injection logic used in app.py's /explain and /quiz,
    factored out here so the agent's tools ground answers in uploaded
    notes exactly the same way Parts 1-3 already do, instead of
    duplicating that logic separately.
    """
    if not has_indexed_content():
        return None
    chunks = retrieve_relevant_chunks(topic, top_k=3)
    return "\n\n".join(chunks) if chunks else None


# ---- Tools: Q3 and Q6 made real. Each function below becomes a
# structured definition (name + description + parameters) that gets sent
# to the LLM. The DOCSTRING is the tool description -- its exact wording
# directly affects which tool the agent picks, which is exactly what we
# covered in Q6.

@tool
def explain_topic(topic: str) -> str:
    """Explain a concept or topic in simple terms, grounded in the
    user's uploaded notes if relevant ones exist. Use this when the user
    wants to understand or learn about something, not be tested on it."""
    agent_logger.info(f"TOOL CALLED: explain_topic(topic='{topic}')")
    context = _get_rag_context(topic)
    return get_explanation(topic, context=context)


@tool(response_format="content_and_artifact")
def generate_quiz(topic: str, num_questions: int = 3):
    """Generate a multiple-choice quiz on a specific topic, grounded in
    the user's uploaded notes if relevant ones exist. Use this when the
    user wants to be tested or quizzed on a topic they already know or
    have just specified."""
    agent_logger.info(
        f"TOOL CALLED: generate_quiz(topic='{topic}', num_questions={num_questions})"
    )
    context = _get_rag_context(topic)
    quiz = get_quiz(topic, num_questions, context=context)
    # content = short text summary the LLM sees and can talk about.
    # artifact = the EXACT structured quiz dict, untouched, that we'll
    # pull straight out in run_agent() below -- never re-typed by the
    # LLM, so it can't silently drop a question or an option like we
    # just witnessed happen twice out of three runs.
    summary = f"Generated a {len(quiz['questions'])}-question quiz on '{topic}'."
    return summary, quiz


@tool
def check_weak_topics() -> list:
    """Check which topics the user has historically performed poorly on,
    based on past quiz attempts. Use this FIRST whenever the user asks
    to be quizzed on their weak areas, mistakes, or what they should
    study more -- BEFORE generating any quiz -- so the quiz targets an
    actual weak topic instead of a random guess."""
    agent_logger.info("TOOL CALLED: check_weak_topics()")
    result = get_weak_topics(min_attempts=1)
    agent_logger.info(f"  -> result: {result}")
    return result


tools = [explain_topic, generate_quiz, check_weak_topics]


# ---- LangGraph: State, Nodes, Edges -- Q4 made real instead of abstract.

class AgentState(TypedDict):
    # This IS the "state" from Q4 -- the shared data that flows through
    # every node. add_messages means each node APPENDS to this list
    # rather than overwriting it, so the full reasoning trail is kept.
    messages: Annotated[list, add_messages]


# NOTE: same model as llm_service.py (openai/gpt-oss-120b), since it's
# confirmed to support both Structured Outputs (Part 2) and tool calling.
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a study companion agent for a college student. You have "
    "tools to explain topics, generate quizzes, and check the user's "
    "weak topics from past quiz history. If the user asks to be quizzed "
    "on their weak areas or what they should study, ALWAYS call "
    "check_weak_topics first, then generate a quiz on the weakest topic "
    "it finds. If they ask to be quizzed on a specific named topic, skip "
    "straight to generate_quiz. If they ask to understand or learn "
    "something, use explain_topic. After calling generate_quiz, just "
    "briefly tell the student what topic the quiz covers -- the exact "
    "questions will be shown separately, so do not retype them yourself."
))


def agent_node(state: AgentState):
    """
    The 'Reason' step (Q2) / the purple 'Agent reasoning' box in the
    diagram. The LLM looks at the conversation so far and decides:
    call a tool, or respond directly with no further tool calls.
    """
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ToolNode is LangGraph's built-in 'Act' + 'Observe' step -- it actually
# EXECUTES whichever tool the LLM decided to call, and feeds the result
# back into state as a new message. This is the Q3 distinction, in code:
# the LLM only DECIDED which tool and what arguments; ToolNode is what
# actually runs the real Python function and gets the real result.
tool_node = ToolNode(tools)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
# tools_condition checks the last message: did the LLM request a tool
# call? If yes -> route to "tools". If no (LLM is ready to answer) -> END.
# This conditional edge, plus the edge below sending "tools" back to
# "agent", together ARE the loop drawn in the diagram.
graph_builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "agent")

agent_graph = graph_builder.compile()


def run_agent(user_message: str) -> dict:
    """
    Entry point app.py calls. Runs the full graph to completion (which
    may loop through "agent" -> "tools" -> "agent" multiple times, like
    the diagram showed).

    Returns a dict with:
    - "message": the agent's natural-language response (topic framing,
      explanations, weak-topic summaries -- fine to be LLM-generated prose)
    - "quiz_data": the EXACT structured quiz dict from generate_quiz's
      artifact, if a quiz was generated this turn -- pulled directly from
      the tool's real output, never retyped by the LLM. This is what
      fixes the dropped-question/dropped-option bug: quiz_data is either
      the real thing or None, never a lossy paraphrase.
    """
    result = agent_graph.invoke({"messages": [("user", user_message)]})
    messages = result["messages"]

    quiz_data = None
    for msg in reversed(messages):
        if getattr(msg, "name", None) == "generate_quiz" and hasattr(msg, "artifact"):
            quiz_data = msg.artifact
            break

    return {
        "message": messages[-1].content,
        "quiz_data": quiz_data,
    }