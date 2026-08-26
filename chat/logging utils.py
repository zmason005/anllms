"""
Chat transcript logging -- opt-in, for test sessions.

Also provides blocks_to_dicts(), which converts OpenAI-style SDK
objects (ChatCompletionMessage, ChatCompletionMessageToolCall, etc.,
as returned via the LiteLLM proxy) into plain JSON-serializable dicts.
This exists because these are pydantic model objects, not plain dicts
-- storing them directly in the message history and passing that to
Flask's jsonify() will raise a TypeError on any turn that includes a
tool call. blocks_to_dicts() is used both to fix that (see server.py)
and to write log entries, so there's one place that does this
conversion, not two.

(Previously this wrapped Anthropic SDK content blocks -- TextBlock,
ToolUseBlock, etc. -- before the migration to the LiteLLM proxy. The
same duck-typed model_dump() check below works for both, since both
SDKs use pydantic models.)

TEST-PHASE LOGGING POLICY (current): logs ARE committed to the repo,
by explicit project decision, because test sessions are verified to
contain no real farm or animal data. This is NOT the policy for
commercial/production use -- see docs/architecture.md's open items for
what changes before real customer data is ever logged (no git commit,
a proper backend store, disclosure/consent, retention limits, access
control and encryption at rest, and legal review).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def blocks_to_dicts(content) -> list[dict]:
    """Turn a list of SDK content blocks (or plain dicts) into plain dicts."""
    result = []
    for block in content:
        if isinstance(block, dict):
            result.append(block)
        elif hasattr(block, "model_dump"):
            result.append(block.model_dump())
        else:
            result.append({"type": "unknown", "repr": repr(block)})
    return result


LOGGING_ENABLED = os.environ.get("ANLLMS_CHAT_LOG", "0") == "1"
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_FILE = _LOG_DIR / f"session_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.jsonl"


def log_turn(user_message: str, assistant_content, tool_results: list[dict]) -> None:
    """Append one turn (user message + assistant reply/tool calls + tool results) as one JSON line."""
    if not LOGGING_ENABLED:
        return
    _LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_message": user_message,
        "assistant_content": blocks_to_dicts(assistant_content),
        "tool_results": tool_results,
    }
    with open(_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
