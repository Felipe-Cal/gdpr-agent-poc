"""
Chainlit chat UI for the GDPR agent.

Chainlit gives us streaming, tool call visibility, and source citations with
zero frontend code. What the client sees:
  - Tokens stream in real time as the LLM generates
  - Each tool call appears as a collapsible step showing input and output
  - Source documents appear as expandable elements below the answer

Run locally:
    chainlit run app.py

Deploy to Railway:
    push to GitHub → connect repo in Railway → set env vars → deploy
"""

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent import build_graph


@cl.on_chat_start
async def on_chat_start():
    # Build one graph per session — MemorySaver keeps conversation history
    # scoped to this graph instance (thread_id is the Chainlit session ID).
    graph = build_graph()
    cl.user_session.set("graph", graph)
    cl.user_session.set("config", {
        "configurable": {"thread_id": cl.user_session.get("id")}
    })

    await cl.Message(
        content=(
            "Hello! I'm a GDPR legal analyst. I can answer questions about:\n\n"
            "- Lawful bases for processing (Article 6)\n"
            "- Data subject rights (Articles 15–22)\n"
            "- DPO requirements (Article 37)\n"
            "- Breach notification (Articles 33–34)\n"
            "- Data transfers outside the EU (Article 44)\n"
            "- Any other GDPR question\n\n"
            "Ask me anything."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    config = cl.user_session.get("config")

    # The answer message — we'll stream tokens into it
    answer_msg = cl.Message(content="")
    await answer_msg.send()

    # Track open tool steps so we can close them when the tool finishes
    # key: LangGraph run_id, value: cl.Step
    open_steps: dict[str, cl.Step] = {}

    async for event in graph.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        config=config,
        version="v2",
    ):
        kind = event["event"]

        # ── Tool call started ─────────────────────────────────────────────
        if kind == "on_tool_start":
            tool_name = event.get("name", "tool")
            step = cl.Step(name=tool_name, type="tool")
            # Show what the agent is passing to the tool
            step.input = str(event["data"].get("input", ""))
            await step.send()
            open_steps[event["run_id"]] = step

        # ── Tool call finished ────────────────────────────────────────────
        elif kind == "on_tool_end":
            step = open_steps.pop(event["run_id"], None)
            if step:
                output = str(event["data"].get("output", ""))
                # Truncate long outputs (e.g. full document chunks)
                step.output = output[:2000] + ("…" if len(output) > 2000 else "")
                await step.update()

        # ── LLM token streaming ───────────────────────────────────────────
        elif kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Only stream final-answer tokens — skip tool-call JSON fragments
            if chunk.content and not getattr(chunk, "tool_calls", None):
                await answer_msg.stream_token(chunk.content)

    await answer_msg.update()
