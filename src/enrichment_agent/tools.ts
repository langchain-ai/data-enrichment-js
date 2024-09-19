/**
 * Tools for data enrichment.
 *
 * This module contains functions that are directly exposed to the LLM as tools.
 * These tools can be used for tasks such as web searching and scraping.
 * Users can edit and extend these tools as needed.
 */
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { RunnableConfig } from "@langchain/core/runnables";

import { ensureConfiguration } from "./configuration.js";
import { AnyRecord, StateAnnotation } from "./state.js";
import { StructuredTool, tool } from "@langchain/core/tools";
import { curry, getTextContent, loadChatModel } from "./utils.js";
import {
  AIMessage,
  isBaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { z } from "zod";

async function search(
  { query }: { query: string },
  config: RunnableConfig,
): Promise<Array<AnyRecord> | null> {
  /**
   * Search for general results.
   *
   * This function performs a search using the Tavily search engine, which is designed
   * to provide comprehensive, accurate, and trusted results. It's particularly useful
   * for answering questions about current events.
   */
  const configuration = ensureConfiguration(config);
  const wrapped = new TavilySearchResults({
    maxResults: configuration.maxSearchResults,
  });
  const result = await wrapped.invoke(query, config);
  return result as Array<AnyRecord> | null;
}

const INFO_PROMPT = `You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>`;

async function scrapeWebsite(
  {
    url,
    __state,
  }: {
    url: string;
    __state?: typeof StateAnnotation.State;
  },
  config: RunnableConfig,
): Promise<string> {
  /**
   * Scrape and summarize content from a given URL.
   */
  const response = await fetch(url);
  const content = await response.text();
  const truncatedContent = content.slice(0, 50000);
  const configuration = ensureConfiguration(config);
  const p = INFO_PROMPT.replace(
    "{info}",
    JSON.stringify(__state?.extractionSchema, null, 2),
  )
    .replace("{url}", url)
    .replace("{content}", truncatedContent);

  const rawModel = await loadChatModel(configuration.model);
  const result = await rawModel.invoke(p, { callbacks: config?.callbacks });
  return getTextContent(result.content);
}

export const createToolNode = (tools: StructuredTool[]) => {
  const toolNode = async (
    state: typeof StateAnnotation.State,
    config: RunnableConfig,
  ) => {
    const message = state.messages[state.messages.length - 1];
    const outputs = await Promise.all(
      (message as AIMessage).tool_calls?.map(async (call) => {
        const tool = tools.find((tool) => tool.name === call.name);
        try {
          if (tool === undefined) {
            throw new Error(`Tool "${call.name}" not found.`);
          }
          const newCall = {
            ...call,
            args: {
              __state: state,
              ...call.args,
            },
          };
          const output = await tool.invoke(
            { ...newCall, type: "tool_call" },
            config,
          );
          if (isBaseMessage(output) && output._getType() === "tool") {
            return output;
          } else {
            return new ToolMessage({
              name: tool.name,
              content:
                typeof output === "string" ? output : JSON.stringify(output),
              tool_call_id: call.id ?? "",
            });
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (e: any) {
          return new ToolMessage({
            content: `Error: ${e.message}\n Please fix your mistakes.`,
            name: call.name,
            tool_call_id: call.id ?? "",
            status: "error",
          });
        }
      }) ?? [],
    );

    return { messages: outputs };
  };
  return toolNode;
};

const searchTool = tool(search, {
  name: "search",
  description: "Search the internet for information on a given topic",
  schema: z.object({
    query: z.string().describe("The search query to look up"),
  }),
});

// Exposed to the
export const TOOLS = [
  searchTool,
  tool(curry(scrapeWebsite, { __state: undefined }), {
    name: "scrapeWebsite",
    description: "Scrape content from a given website URL",
    schema: z.object({
      url: z.string().url().describe("The URL of the website to scrape"),
    }),
  }),
];

export const toolNode = createToolNode([
  searchTool,
  tool(scrapeWebsite, {
    name: "scrapeWebsite",
    description: "Scrape content from a given website URL",
    schema: z.object({
      url: z.string().url().describe("The URL of the website to scrape"),
      __state: z.any(),
    }),
  }),
]);
