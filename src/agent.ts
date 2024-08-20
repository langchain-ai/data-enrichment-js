import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  isBaseMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { StructuredTool, tool } from "@langchain/core/tools";
import { StateGraph } from "@langchain/langgraph";
import { z } from "zod";
import { State, StateAnnotation } from "./utils/state.js";
import { curry, getTextContent } from "./utils/utils.js";
import { initChatModel } from "langchain/chat_models/universal";

/**
 * Web Research Workflow
 *
 * This script implements a workflow for conducting web research on a given topic.
 * It uses AI-powered tools to search the web, scrape websites, and compile information
 * into a structured format.
 */

// Default configuration
const DEFAULT_CONFIG = {
  /** The name of the AI model to use */
  MODEL_NAME: "claude-3-5-sonnet-20240620",
  /** Maximum number of search results to return */
  MAX_SEARCH_RESULTS: 10,
  /** Maximum number of times the Info tool can be called before "__end__"ing the workflow */
  MAX_INFO_TOOL_CALLS: 3,
};

const rawModel = await initChatModel(DEFAULT_CONFIG.MODEL_NAME);

/**
 * Main prompt template for the AI agent.
 * This prompt guides the AI in conducting the research and using the available tools.
 */
const main_prompt = `You are doing web research on behalf of a user. You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- \`Search\`: call a search tool and get back some results
- \`ScrapeWebsite\`: scrape a website and get relevant notes about the given request. This will update the notes above.
- \`Info\`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}`;

const info_prompt = `You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

{content}`;

// Define the tools
const searchTool = new TavilySearchResults({
  maxResults: DEFAULT_CONFIG.MAX_SEARCH_RESULTS,
});

function createWebScraper(prompt: string) {
  /**
   * Tool for scraping a specific webpage.
   * This tool is used to gain more detailed information from a specific
   * URL that the agent has found compared to the general search results.
   */

  const scrapeWebsite = async ({
    url,
    __state,
  }: {
    url: string;
    __state?: Partial<State>;
  }): Promise<string> => {
    const state = __state as State;
    // Note: WebBaseLoader is not directly available in TypeScript.
    // You might need to use a different method to fetch and parse web content.
    const response = await fetch(url);
    const content = await response.text();
    const p = prompt
      .replace("{info}", JSON.stringify(state?.schema?.shape, null, 2))
      .replace("{url}", url)
      .replace("{content}", content);
    const result = await rawModel.withRetry().invoke(p);
    return getTextContent(result.content);
  };

  return scrapeWebsite;
}
const narrowFn = curry(createWebScraper(info_prompt), { __state: {} });
const scrapeWebsiteTool = tool(narrowFn, {
  name: "ScrapeWebsite",
  description: "Used to scrape a website",
  schema: z.object({
    url: z.string().describe("URL to scrape"),
  }),
});

// Create the full tool
const scrapeWebsiteToolFull = tool(createWebScraper(info_prompt), {
  name: "ScrapeWebsite",
  description: "Used to scrape a website THE FULL WAY",
  schema: z.object({
    url: z.string().describe("URL to scrape"),
    __state: z.any(),
  }),
});

/**
 * Schema for structuring and validating the AI's judgment on the current results.
 * This schema ensures that the AI provides reasoning and a clear decision
 * on whether the gathered information is satisfactory.
 */
const infoIsSatisfactory = z
  .object({
    reason: z
      .array(z.string())
      .describe(
        "First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons.",
      ),
    isSatisfactory: z
      .boolean()
      .describe(
        "After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching.",
      ),
  })
  .describe("Submit a judgment of whether the result is satisfactory or not");

// Define Nodes:

/**
 * Main agent node: Decides next action based on current state.
 * This function processes the current state and generates the next action for the AI.
 *
 * @param state - The current state of the research workflow
 * @returns An object containing the next set of messages to be processed
 */
const callModel = async (
  state: State,
): Promise<{
  messages: BaseMessage[];
  info?: z.infer<typeof state.schema>;
}> => {
  const infoTool = tool(async (_args: z.infer<typeof state.schema>) => {}, {
    name: "Info",
    description: "Call this when you have gathered all the relevant info",
    schema: state.schema,
  });

  const p = main_prompt
    .replace("{info}", JSON.stringify(state.schema.shape, null, 2))
    .replace("{topic}", state.topic);
  const messages = [new HumanMessage(p), ...state.messages];
  const model = rawModel.bindTools([scrapeWebsiteTool, searchTool, infoTool], {
    tool_choice: "any",
  });
  const response = await model.invoke(messages);
  const info =
    response.tool_calls
      ?.filter((tc) => tc.name == "Info")
      .map((tc) => state.schema.parse(tc?.args))?.[0] ?? undefined;
  return { messages: [response], info };
};

/**
 * Checker node: Validates the gathered information.
 * This function assesses whether the collected information is satisfactory or if more research is needed.
 *
 * @param state - The current state of the research workflow
 * @returns Either a set of messages to continue research or the final info object if satisfactory
 */
const callChecker = async (
  state: State,
): Promise<
  { messages: BaseMessage[] } | { info: z.infer<typeof state.schema> }
> => {
  const p = main_prompt
    .replace("{info}", JSON.stringify(state.schema.shape, null, 2))
    .replace("{topic}", state.topic);
  const messages = [new HumanMessage(p), ...state.messages.slice(0, -1)];
  const presumedInfo = state.info;
  const checker_prompt = `I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}`;
  const p1 = checker_prompt.replace(
    "{presumed_info}",
    JSON.stringify(presumedInfo ?? {}, null, 2),
  );
  messages.push(new HumanMessage(p1));

  const response = await rawModel
    .withStructuredOutput(infoIsSatisfactory)
    .invoke(messages);

  if (response.isSatisfactory) {
    try {
      return { info: presumedInfo };
    } catch (e) {
      return {
        messages: [
          new ToolMessage({
            tool_call_id:
              (state.messages[state.messages.length - 1] as AIMessage)
                .tool_calls?.[0]?.id || "",
            content: `Invalid response: ${e}`,
            name: "Info",
          }),
        ],
      };
    }
  } else {
    return {
      messages: [
        new ToolMessage({
          tool_call_id:
            (state.messages[state.messages.length - 1] as AIMessage)
              .tool_calls?.[0]?.id || "",
          content: JSON.stringify(response),
          name: "Info",
          artifact: response,
        }),
      ],
    };
  }
};

/**
 * Bad agent node: Handles cases where the agent makes invalid tool calls.
 * This function generates error messages when the agent doesn't follow the expected behavior.
 *
 * @param state - The current state of the research workflow
 * @returns An object containing error messages
 */
const badAgent = (state: State): { messages: BaseMessage[] } => {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  if ((lastMessage?.tool_calls?.length || 0) > 0) {
    return {
      messages:
        lastMessage.tool_calls?.map((call) => {
          return new ToolMessage({
            tool_call_id: call.id || "",
            content: "You must call one, and only one, tool!",
            name: call.name,
          });
        }) || [],
    };
  }
  return {
    messages: [
      new HumanMessage(
        "You must call one, and only one, tool! You can call the `Info` tool to finish the task.",
      ),
    ],
  };
};

/**
 * Routing function: Determines the next node after the agent's action.
 * This function decides whether to proceed with tool execution, check results, or handle errors.
 *
 * @param state - The current state of the research workflow
 * @returns The name of the next node to execute
 */
const route_after_agent = (
  state: State,
): "badAgent" | "callChecker" | "toolNode" | "__end__" => {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  const numRounds = state.messages.filter(
    (m) => ((m._getType() as string) === "tool" && m.name === "Info") || false,
  ).length;

  if (!lastMessage.tool_calls || lastMessage.tool_calls.length !== 1) {
    return "badAgent";
  } else if (lastMessage.tool_calls[0].name === "Info") {
    if (numRounds > 2) {
      return "__end__";
    }
    return "callChecker";
  } else {
    return "toolNode";
  }
};

/**
 * Routing function: Determines whether to continue research or "__end__" the workflow.
 * This function decides if the gathered information is satisfactory or if more research is needed.
 *
 * @param state - The current state of the research workflow
 * @returns Either "callModel" to continue research or "__end__" to finish the workflow
 */
const route_after_checker = (state: State): "__end__" | "callModel" => {
  if (state.info) {
    return "__end__";
  }
  return "callModel";
};

const createToolNode = (tools: StructuredTool[]) => {
  const toolNode = async (state: State) => {
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
          const output = await tool.invoke({ ...newCall, type: "tool_call" });
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
          });
        }
      }) ?? [],
    );

    return { messages: outputs };
  };
  return toolNode;
};

// Create the graph
const workflow = new StateGraph(StateAnnotation)
  .addNode("callModel", callModel)
  .addNode("callChecker", callChecker)
  .addNode("badAgent", badAgent)
  .addNode("toolNode", createToolNode([searchTool, scrapeWebsiteToolFull]))
  .addEdge("__start__", "callModel")
  .addConditionalEdges("callModel", route_after_agent)
  .addEdge("toolNode", "callModel")
  .addConditionalEdges("callChecker", route_after_checker)
  .addEdge("badAgent", "callModel");

export const graph = workflow.compile();
graph.name = "ResearchTopic";
