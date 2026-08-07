**Concept: a Deep Agents implementation for life sciences clients with skills and tooling to act as a research assistant for biologists**

Principle: This is a demo. Keep everything as simple as possible.

**Necessary tools**

- Boolean PubMed search, returning list of PMIDs, likely including associated paper names, lead author name, and publication year, plus maybe others—agent can pick. Can dump to filesystem as JSON or CSV if more than a certain number of results.
- Retrieve PubMed abstract or paper by PMID. Caches locally by default upon retrieval so does not need to be fetched multiple times.
- Ask question of paper or abstract using subagent. Checks for cached paper before retrieving using API. Must be able to fan out a large number of subagents.

To start with, we’ll just do abstract, since that’s always just text. Papers, with their associated multimodal and multi-file-type data will come later.

Instead of formal tools, we’ll use code execution via LangChain’s interpreter (https://docs.langchain.com/oss/python/deepagents/interpreters). Each “tool” is a segment in the prompt giving a reference code snippet the agent can execute to execute the associated op.

Subagents via LangChain’s dynamic subagents (https://docs.langchain.com/oss/python/deepagents/dynamic-subagents).

Web search beyond PubMed will come later.