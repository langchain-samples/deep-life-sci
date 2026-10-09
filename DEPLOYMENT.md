# Deploy to LangSmith

To run the agent server in the cloud with [LangSmith Deployment](https://docs.langchain.com/langsmith/deployments),
first complete the [Quickstart](README.md#quickstart), then run:

```bash
uv run scripts/deploy.py
```

This creates a deployment named `deep-life-sci-cloud`, or updates it if it exists (use `--name` to pick
another). Models are whatever your models file says when you deploy; edit it and deploy
again to change them.

## Cost

The default is a Dedicated Small deployment, always on, about $390 a month. 

For a cheaper deployment to try things out, use `--type serverless`: about $62 a month at most, depending on usage.
The LangSmith Plus plan includes one free. It scales to zero when idle, so the first request after
a quiet spell has higher latency.

Model calls and sandboxes are billed separately. See
[pricing](https://www.langchain.com/pricing) for current rates.

## Sign-in

Set `OIDC_ISSUER` and `OIDC_CLIENT_ID` in `.env` (see `.env.example`) before
deploying, and the deployment serves the chat UI at `<deployment URL>/app/`, where people
sign in with your organization's identity provider: Entra ID, Okta, or any other OpenID
Connect provider. Each person sees only their own conversations.

Without them, the deployment accepts only LangSmith API keys from your workspace, so it is
for your own use. Chat with it from the local chat UI using the deployment URL shown on its
LangSmith page:

```bash
uv run scripts/dev.py --remote https://<your-deployment>.langgraph.app
```

## Retention

A deployment deletes conversations after 90 days without use. To change, change
`THREAD_TTL_MINUTES` in `deep_life_sci/paths.py` and both `checkpointer.ttl` and `store.ttl`
in `langgraph.json` together.
