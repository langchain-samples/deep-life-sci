"""Prompts, split by who reads them.

`system.py` is the root agent's; `subagents.py` holds the three analyst leaves. They are
re-exported here so callers keep importing from one place, and so a prompt can move
between files without touching `agent.py`.
"""

from research_agent.prompts.subagents import (
    ABSTRACT_ANALYST,
    FIGURE_ANALYST,
    FULL_TEXT_ANALYST,
)
from research_agent.prompts.system import SYSTEM_PROMPT

__all__ = [
    "ABSTRACT_ANALYST",
    "FIGURE_ANALYST",
    "FULL_TEXT_ANALYST",
    "SYSTEM_PROMPT",
]
