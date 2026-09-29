import { Client } from "@langchain/langgraph-sdk";

export function createClient(apiUrl: string, authScheme: string | undefined) {
  return new Client({
    apiUrl,
    ...(authScheme && {
      defaultHeaders: {
        "X-Auth-Scheme": authScheme,
      },
    }),
  });
}
