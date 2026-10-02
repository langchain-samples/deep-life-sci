"""Whose thread a run is in, for what a thread keeps outside the agent server.

A thread's sandbox (graph.py) and the durable copies of its uploads (middleware/uploads.py)
are keyed by thread, outside the server's own database. On a deployment with sign-in that
key has to include the owner. Thread ids are client-chosen UUIDs, so once a thread is deleted
(by its owner, or by the checkpointer TTL) anyone who knows its id — from a shared link, say
— can create it again, while its sandbox (kept for days after it stops) and its uploads
(kept for weeks in the store) are still there. Keyed by owner as well, a recreated id
reaches nothing of the old thread's.

The owner is the signed-in caller: auth.py lets a signed-in user run only threads it owns.
It comes from the identity the server attaches to every run's config, under reserved keys a
request cannot set. Workspace members (LangSmith API keys, Studio) are not signed-in users;
for them the thread's own `owner` metadata, which the server merges into the run's, keeps a
member who runs a user's thread on that user's sandbox. With neither — a local server, the
CLI, a thread no signed-in user made — the thread id alone is the key, as it always was.
"""

from __future__ import annotations

import hashlib
from typing import Any

# What auth.py's `authenticate` grants a signed-in user, and so what marks one here.
USER = "oidc-user"


def thread_owner(config: Any) -> str | None:
    """The identity of the signed-in user whose thread this run is in, if there is one."""
    configurable = config.get("configurable") or {}
    if USER in (configurable.get("langgraph_auth_permissions") or ()):
        identity = configurable.get("langgraph_auth_user_id")
        return str(identity) if identity else None
    owner = (config.get("metadata") or {}).get("owner")
    return owner if isinstance(owner, str) and owner else None


def thread_scope(config: Any) -> str:
    """What a thread's sandbox and stored uploads are keyed by: the thread id, plus a tag for
    its owner when it has one. "default" for the CLI, which has no thread.

    The tag is a digest, never the identity itself, which may be an email address and would
    otherwise end up in sandbox names.
    """
    thread = str((config.get("configurable") or {}).get("thread_id") or "default")
    owner = thread_owner(config)
    if not owner:
        return thread
    return f"{thread}.{hashlib.sha256(owner.encode()).hexdigest()[:12]}"
