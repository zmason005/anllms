"""
Chat server -- runs the tool-use loop against a LiteLLM proxy (OpenAI-
compatible endpoint) and serves a simple local web chat page.

Requires LITELLM_API_KEY and LITELLM_BASE_URL environment variables,
pointing at the project's self-hosted LiteLLM proxy. This app runs
OUTSIDE claude.ai, so it needs its own credentials, routed through the
proxy rather than calling any model vendor directly.

Which model is used is controlled by the ANLLMS_MODEL environment
variable (defaults to "gemini-flash" -- set this to whatever alias is
configured on the LiteLLM proxy, e.g. "gemini-flash", "mistral-large").

Run with:
    export LITELLM_API_KEY=sk-...
    export LITELLM_BASE_URL=https://litellm-proxy-700813965617.us-east1.run.app
    export ANLLMS_MODEL=gemini-flash   # optional, defaults below
    python -m chat.server
Then open http://localhost:5000 in a browser.
"""

from __future__ import annotations

import json
import os

from flask import Flask, jsonify, request, send_from_directory

from chat.logging_utils import blocks_to_dicts, log_turn
from chat.tools import TOOL_DEFINITIONS, ChatSession

SYSTEM_PROMPT = """You are a dairy nutrition assistant built on the NASEM \
(2021) Nutrient Requirements of Dairy Cattle model.

SCOPE: You can ONLY answer questions about LACTATING dairy cows using the \
tools available to you: dry matter intake, energy (NEL) requirement and \
supply, protein (MP) requirement and supply, all 13 NASEM minerals, \
vitamins A/D/E, and water requirement. You do NOT cover dry cows, \
heifers, or other species. If asked about any of these, say clearly and \
plainly that this is not yet supported, rather than guessing or estimating.

IMPORTANT: mineral and vitamin "balance" numbers come directly from the \
underlying reference model's own supply calculation, NOT from \
independently-cited supply equations the way requirements are. If a user \
asks specifically how mineral/vitamin supply is calculated, say this \
plainly rather than implying the same level of citation-backed detail as \
the requirement side.

RULES:
- Never state a number that didn't come from a tool call. If you don't \
have a tool for something, say so.
- When you give a numeric answer, mention the underlying equation number \
if the tool result includes one (e.g. "via Equation 2-2").
- If the user asks "why" a number is what it is, use explain_component \
to get the real citation/assumptions rather than making up a reason. For \
minerals use component='mineral_<Symbol>' (e.g. 'mineral_Ca'), for \
vitamins use 'vitamin_<Symbol>' (e.g. 'vitamin_E'), for water use 'water'.
- Use search_feed_ingredient before calculate_lactating_cow_requirements \
if you are not certain an ingredient name matches the feed library exactly.
- Keep answers concise and in plain language -- this is a chat interface, \
not a report.
- If no ration is given, the calculation tool falls back to a standard \
reference diet -- say so plainly and offer to use the user's real diet \
instead, rather than presenting the placeholder result as if it were \
based on their actual ration.
"""

# LiteLLM proxy expects OpenAI-style tool schemas: {"type": "function",
# "function": {"name": ..., "description": ..., "parameters": ...}}.
# TOOL_DEFINITIONS in tools.py stays in Anthropic's flat shape (name /
# description / input_schema) since that's still the source of truth;
# this reshapes it for the wire format this client needs.
def _to_openai_tools(tool_defs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tool_defs
    ]


OPENAI_TOOL_DEFINITIONS = _to_openai_tools(TOOL_DEFINITIONS)

DEFAULT_MODEL = os.environ.get("ANLLMS_MODEL", "gemini-flash")

app = Flask(__name__, static_folder="static")

# NOTE: single global session -- fine for a local single-user chat window,
# not a multi-user deployment. See ChatSession docstring.
session = ChatSession()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    import openai

    api_key = os.environ.get("LITELLM_API_KEY")
    base_url = os.environ.get("LITELLM_BASE_URL")
    if not api_key or not base_url:
        return jsonify({
            "error": "LITELLM_API_KEY and LITELLM_BASE_URL environment variables must both be set."
        }), 500

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    body = request.get_json()
    history = body.get("history", [])  # list of {role, content}
    user_message = body["message"]

    # OpenAI convention: system prompt is a message in the array, not a
    # separate top-level param.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]

    # Tool-use loop: keep calling the model until it stops requesting tools.
    for _ in range(10):  # hard cap to avoid a runaway loop
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            tools=OPENAI_TOOL_DEFINITIONS,
            messages=messages,
        )

        choice = response.choices[0]
        assistant_message = choice.message

        if choice.finish_reason != "tool_calls":
            reply_text = assistant_message.content or ""
            messages.append({"role": "assistant", "content": reply_text})
            log_turn(user_message, blocks_to_dicts([assistant_message]), [])
            # Drop the system prompt before handing history back to the
            # client -- it gets re-added on the next turn.
            return jsonify({"reply": reply_text, "history": messages[1:]})

        # Record the assistant's tool-call turn in OpenAI's message shape.
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": blocks_to_dicts(assistant_message.tool_calls or []),
        })

        tool_results = []
        for tool_call in assistant_message.tool_calls or []:
            tool_args = json.loads(tool_call.function.arguments)
            result = session.dispatch(tool_call.function.name, tool_args)
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
            tool_results.append(tool_result_message)
            messages.append(tool_result_message)

        log_turn(user_message, blocks_to_dicts([assistant_message]), tool_results)

    return jsonify({"error": "Too many tool-use steps without a final answer."}), 500


if __name__ == "__main__":
    if not os.environ.get("LITELLM_API_KEY") or not os.environ.get("LITELLM_BASE_URL"):
        print("WARNING: LITELLM_API_KEY / LITELLM_BASE_URL are not both set. The chat endpoint will fail.")
    app.run(host="0.0.0.0", port=5000)
