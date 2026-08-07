"""Build the sandbox snapshot the agent boots from.

Run this once (and again whenever you want to change the library set):

    uv run build_snapshot.py

It boots a plain sandbox, installs the scientific Python stack, and freezes the result
as a named snapshot. `agent.py` then boots from that snapshot, which turns a ~30s
per-run `pip install` into a ~1s start.

Put the printed name in `.env` as SANDBOX_SNAPSHOT_NAME.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from langsmith.sandbox import SandboxClient  # noqa: E402

SNAPSHOT_NAME = os.environ.get("SANDBOX_SNAPSHOT_NAME", "pubmed-py")

# Pinned so a rebuild doesn't silently move the library versions under the agent.
# Bump deliberately, then rebuild.
PACKAGES = [
    "numpy==2.5.1",
    "pandas==3.0.5",
    "scipy==1.18.0",
    "matplotlib==3.11.1",
]

# The workspace is baked in so the agent never has to create it, and so a prompt that
# says "save plots to /workspace/out" is true the instant the sandbox boots.
BUILD = (
    "mkdir -p /workspace/out && "
    f"pip install --break-system-packages --quiet {' '.join(PACKAGES)}"
)

VERIFY = (
    'python3 -c "import numpy, pandas, scipy, matplotlib; '
    "matplotlib.use('Agg'); import matplotlib.pyplot as plt; "
    "plt.plot([1, 2, 3]); plt.savefig('/workspace/out/_smoke.png'); "
    "print(numpy.__version__, pandas.__version__, scipy.__version__, "
    'matplotlib.__version__)"'
)


def main() -> None:
    client = SandboxClient()

    for existing in client.list_snapshots(name_contains=SNAPSHOT_NAME):
        if existing.name == SNAPSHOT_NAME:
            print(f"[snapshot] deleting existing {SNAPSHOT_NAME} ({existing.id})")
            client.delete_snapshot(existing.id)

    t0 = time.monotonic()
    with client.sandbox(idle_ttl_seconds=900) as sandbox:
        print(f"[build] sandbox up in {time.monotonic() - t0:.1f}s")

        t1 = time.monotonic()
        result = sandbox.run(BUILD, timeout=900)
        if result.exit_code != 0:
            raise SystemExit(f"install failed (exit {result.exit_code}):\n{result.stderr}")
        print(f"[build] installed in {time.monotonic() - t1:.1f}s")

        # Verify inside the sandbox we are about to freeze, not after the fact — a
        # snapshot of a broken environment is worse than no snapshot.
        check = sandbox.run(VERIFY, timeout=120)
        if check.exit_code != 0:
            raise SystemExit(f"verification failed:\n{check.stderr}")
        print(f"[build] verified: {check.stdout.strip()}")

        # Drop the smoke-test artifact so the snapshot's /workspace/out starts empty.
        sandbox.run("rm -f /workspace/out/_smoke.png")

        t2 = time.monotonic()
        snapshot = client.capture_snapshot(sandbox.name, SNAPSHOT_NAME, timeout=600)
        print(f"[build] captured in {time.monotonic() - t2:.1f}s")

    print(f"\n[snapshot] ready: name={snapshot.name} id={snapshot.id}")
    print(f"[snapshot] set SANDBOX_SNAPSHOT_NAME={snapshot.name} in .env")


if __name__ == "__main__":
    main()
