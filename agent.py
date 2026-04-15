"""
LangGraph ReAct agent — provider-agnostic.

Same graph topology as Phase 2 of the GCP project. The only difference:
ChatVertexAI → init_chat_model, which supports openai/anthropic/groq/etc.
via two env vars (LLM_PROVIDER, LLM_MODEL). No code changes to switch models.
"""

from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from config import settings
from tools import TOOLS

SYSTEM_PROMPT = """You are an expert GDPR legal analyst. You have access to three tools:

1. search_gdpr_documents — searches your knowledge base of GDPR regulation text and EDPB guidelines
2. get_gdpr_article — quickly retrieves the key provisions of a specific GDPR article by number
3. web_search — searches the web for recent enforcement actions, new guidelines, or news

How to reason:
- For specific article questions, start with get_gdpr_article for speed
- For broader questions or when you need exact regulatory text, use search_gdpr_documents
- For recent developments (fines, new guidance, enforcement), use web_search
- Call multiple tools if a question requires information from different sources
- Always cite your sources (article numbers, document names) in your final answer
- Distinguish between hard legal requirements and best-practice recommendations
- If information is insufficient, say so — do not speculate

You support multi-turn conversations. Use the conversation history for follow-up questions."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph():
    # init_chat_model resolves the right LangChain integration at runtime:
    #   provider="openai"    → ChatOpenAI    (needs OPENAI_API_KEY)
    #   provider="anthropic" → ChatAnthropic (needs ANTHROPIC_API_KEY)
    #   provider="groq"      → ChatGroq      (needs GROQ_API_KEY, very fast)
    llm = init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
    ).bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        return {"messages": [llm.invoke(messages)]}

    def run_tools(state: AgentState) -> dict:
        last: AIMessage = state["messages"][-1]
        tool_map = {t.name: t for t in TOOLS}
        results = []
        for call in last.tool_calls:
            try:
                result = tool_map[call["name"]].invoke(call["args"])
            except Exception as e:
                result = f"Tool error: {e}"
            results.append(
                ToolMessage(content=str(result), tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": results}

    def should_continue(state: AgentState) -> Literal["run_tools", "__end__"]:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "run_tools"
        return "__end__"

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("run_tools", run_tools)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", should_continue)
    graph.add_edge("run_tools", "call_model")

    return graph.compile(checkpointer=MemorySaver())
