import { useRunProgress, useStreamContext } from "@/providers/Stream";
import { AssistantMessageLoading } from "./messages/ai";

/**
 * What the agent is doing, while it is doing it.
 *
 * This agent orchestrates inside a single `eval` call, so from the transcript's point of
 * view a run is one tool call that has not come back yet — and with tool calls hidden, which
 * is how `scripts/dev.py` opens the app, there is nothing on screen at all. The graph
 * narrates its own inner calls instead (`research_agent/middleware/progress.py`), and
 * `providers/Stream.tsx` keeps the latest line.
 *
 * Both halves are deliberate. The dots stay up for the whole run rather than stopping at the
 * first AI message, because the first AI message here arrives seconds into something that
 * takes minutes. The text is what makes them mean anything; without an event yet — or on a
 * turn the agent answers straight from context — the dots stand alone, which is upstream's
 * behaviour.
 */
export function RunStatus() {
  const { isLoading } = useStreamContext();
  const progress = useRunProgress();

  if (!isLoading) return null;

  return (
    <div className="mr-auto flex items-center gap-3">
      <AssistantMessageLoading />
      {progress && (
        <span className="animate-in fade-in-0 text-muted-foreground text-sm">
          {progress}
        </span>
      )}
    </div>
  );
}
