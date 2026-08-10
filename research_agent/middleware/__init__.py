"""Middleware wrapping the agent's tool calls.

Order matters where these are installed in `agent.py`: `ArtifactMiddleware` sweeps once
the call it wraps has returned, and `LoopLagProbe` sits innermost so the wall time it
reports is the `eval` itself rather than the artifact sweep that follows it.
"""

from research_agent.middleware.artifacts import ArtifactMiddleware
from research_agent.middleware.perf import LoopLagProbe, install_logging

__all__ = ["ArtifactMiddleware", "LoopLagProbe", "install_logging"]
