/**
 * Tools for data enrichment.
 *
 * This module contains functions that are directly exposed to the LLM as tools.
 * These tools can be used for tasks such as web searching and scraping.
 * Users can edit and extend these tools as needed.
 */
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { RunnableConfig } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";

import { INFO_PROMPT } from "./prompts.js";
import { ensureConfiguration } from "./configuration.js";
import { StateAnnotation } from "./state.js";
import { getTextContent, loadChatModel } from "./utils.js";
import {
  AIMessage,
  isBaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { z } from "zod";

/**
 * Initialize tools within a function so that they have access to the current
 * state and config at runtime.
 */
function initializeTools(
  state?: typeof StateAnnotation.State,
  config?: RunnableConfig,
) {
  /**
   * Search for general results.
   *
   * This function performs a search using the Tavily search engine, which is designed
   * to provide comprehensive, accurate, and trusted results. It's particularly useful
   * for answering questions about current events.
   */
  const configuration = ensureConfiguration(config);
  const searchTool = new TavilySearchResults({
    maxResults: configuration.maxSearchResults,
  });

  async function scrapeWebsite({ url }: { url: string }): Promise<string> {
    /**
     * Scrape and summarize content from a given URL.
     */
    const response = await fetch(url);
    const content = await response.text();
    const truncatedContent = content.slice(0, 50000);
    const p = INFO_PROMPT.replace(
      "{info}",
      JSON.stringify(state?.extractionSchema, null, 2),
    )
      .replace("{url}", url)
      .replace("{content}", truncatedContent);

    const rawModel = await loadChatModel(configuration.model);
    const result = await rawModel.invoke(p);
    return getTextContent(result.content);
  }

  const scraperTool = tool(scrapeWebsite, {
    name: "scrapeWebsite",
    description: "Scrape content from a given website URL",
    schema: z.object({
      url: z.string().url().describe("The URL of the website to scrape"),
    }),
  });

  return [searchTool, scraperTool];
}

export const toolNode = async (
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
) => {
  const message = state.messages[state.messages.length - 1];
  // Initialize the tools within the context of the node so that the tools
  // have the current state of the graph and the config in scope.
  // See: https://js.langchain.com/docs/how_to/tool_runtime
  const tools = initializeTools(state, config);
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

// No state or config required here since these are just bound to the chat model
// and are only used to define schema.
// The tool node above will actually call the functions.
export const MODEL_TOOLS = initializeTools();
