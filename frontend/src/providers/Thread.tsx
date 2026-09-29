import { validate } from "uuid";
import { Thread } from "@langchain/langgraph-sdk";
import { useQueryState } from "nuqs";
import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useState,
  Dispatch,
  SetStateAction,
} from "react";
import { createClient } from "./client";

interface ThreadContextType {
  getThreads: () => Promise<Thread[]>;
  threads: Thread[];
  setThreads: Dispatch<SetStateAction<Thread[]>>;
  threadsLoading: boolean;
  setThreadsLoading: Dispatch<SetStateAction<boolean>>;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

function getThreadSearchMetadata(
  assistantId: string,
): { graph_id: string } | { assistant_id: string } {
  if (validate(assistantId)) {
    return { assistant_id: assistantId };
  } else {
    return { graph_id: assistantId };
  }
}

export function ThreadProvider({ children }: { children: ReactNode }) {
  const envApiUrl: string | undefined = process.env.NEXT_PUBLIC_API_URL;
  const envAssistantId: string | undefined =
    process.env.NEXT_PUBLIC_ASSISTANT_ID;
  const envAuthScheme: string | undefined = process.env.NEXT_PUBLIC_AUTH_SCHEME;

  const [apiUrl] = useQueryState("apiUrl", {
    defaultValue: envApiUrl || "",
  });
  const [assistantId] = useQueryState("assistantId");
  const [authScheme] = useQueryState("authScheme", {
    defaultValue: envAuthScheme || "",
  });
  const [threads, setThreads] = useState<Thread[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(false);

  const getThreads = useCallback(async (): Promise<Thread[]> => {
    const resolvedAssistantId = assistantId || envAssistantId;
    if (!apiUrl || !resolvedAssistantId) return [];
    const client = createClient(apiUrl, authScheme || undefined);

    // The sidebar needs exactly one field of thread state — the first human
    // message, for the title. Asking for all of `values` drags this agent's QuickJS
    // heap snapshot down with it (125 MB for 40 threads, measured). See frontend/CLAUDE.md.
    const query = {
      metadata: {
        ...getThreadSearchMetadata(resolvedAssistantId),
      },
      limit: 100,
    };
    let threads: Thread[];
    try {
      threads = await client.threads.search({
        ...query,
        select: ["thread_id", "created_at", "updated_at", "metadata", "status"],
        extract: { first_message: "values.messages[0]" },
      });
    } catch {
      // Older servers have neither `select` nor `extract`. Slow beats empty.
      threads = await client.threads.search(query);
    }

    // Put the extracted message back where the thread list already looks for it.
    return threads.map((t) =>
      t.extracted?.first_message
        ? ({ ...t, values: { messages: [t.extracted.first_message] } } as Thread)
        : t,
    );
  }, [apiUrl, assistantId, authScheme, envAssistantId]);

  const value = {
    getThreads,
    threads,
    setThreads,
    threadsLoading,
    setThreadsLoading,
  };

  return (
    <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>
  );
}

export function useThreads() {
  const context = useContext(ThreadContext);
  if (context === undefined) {
    throw new Error("useThreads must be used within a ThreadProvider");
  }
  return context;
}
