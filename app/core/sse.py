"""Server-Sent Events wire format.

An event is `event: <name>` + `data: <payload>` + a blank line. The payload is
always one line of JSON: `json.dumps` escapes newlines, so answer text with
line breaks can never be mistaken for the end of an event, and the client
parses every event the same way.
"""

import json
from typing import Any


def format_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
