**Concept: a Deep Agents implementation for life sciences clients with skills and tooling to act as a research assistant for biologists**

Principle: This is a demo. Keep everything as simple as possible.

**Necessary tools**

- Boolean PubMed search, returning list of PMIDs with paper names, lead author, and publication year. This is `esearch` then `esummary`—two calls, and esummary carries all the metadata we need. Can dump to filesystem as JSON or CSV if more than a certain number of results.
- Retrieve PubMed abstracts by PMID, batched via `efetch`. Caches locally on retrieval so nothing is fetched twice.
- Ask question of an abstract using a subagent. The orchestrating code batch-fetches the abstracts first and passes the text into each subagent's prompt, so subagents do no I/O of their own. Must be able to fan out a large number of subagents.

**Constraint driving the above:** NCBI rate-limits E-utilities to 3 requests/sec (10 with an API key). That's why abstracts are batch-fetched up front instead of pulled per-subagent—fifty subagents each hitting the API would just collect 429s.

The API has enough sharp edges to be worth its own writeup—silent query mangling, malformed PMIDs returning the wrong paper, hard batch caps that fail as HTTP 200. See `pubmed_api_notes/` (gitignored) for per-service notes; the tools need to account for those, but they don't belong in this doc.

To start with, we’ll just do abstract, since that’s always just text. Papers, with their associated multimodal and multi-file-type data will come later.

Rather than exposing tools to the model directly, we’ll have it compose them in code via LangChain’s interpreter (https://docs.langchain.com/oss/python/deepagents/interpreters). The interpreter runs JavaScript in QuickJS with no network, filesystem, or shell access, so the PubMed calls themselves still have to be real Python tools—they reach the interpreter through programmatic tool calling (`ptc=[...]`), which exposes them as async functions on a `tools` global (`pubmed_search` → `tools.pubmedSearch(...)`). So: two thin Python tools, and each gets a segment in the prompt with a reference JS snippet showing how to call and compose it. Put the built-in filesystem tools in the allowlist too, so the agent can check the cache and write out result sets.

Watch the interpreter defaults—`timeout` is 5s, which a fan-out will blow through immediately. `max_result_chars` (4000) truncates what comes back to the model.

Subagents via LangChain’s dynamic subagents (https://docs.langchain.com/oss/python/deepagents/dynamic-subagents), dispatched from interpreter code with `task()`. Give them a `response_schema` so the synthesis step gets structured objects instead of prose to parse. Cheap model on the leaves (Haiku for abstract Q&A, Sonnet at the root) is worth showing—it’s the cost story for a client running thousands of abstracts.

Both the interpreter and dynamic subagents are beta; flag that to clients.

Web search beyond PubMed will come later.