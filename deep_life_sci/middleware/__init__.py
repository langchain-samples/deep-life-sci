"""Middleware wrapping the agent's tool calls.

Order matters where these are installed in `agent.py`: `ArtifactMiddleware` sweeps once
the call it wraps has returned, and `LoopLagProbe` sits innermost so the wall time it
reports is the `eval` itself rather than the artifact sweep that follows it.
"""

from deep_life_sci.middleware.artifacts import ArtifactMiddleware
from deep_life_sci.middleware.cadence import UpdateCadence
from deep_life_sci.middleware.perf import LoopLagProbe, install_logging

__all__ = ["ArtifactMiddleware", "LoopLagProbe", "UpdateCadence", "install_logging"]
