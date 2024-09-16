/**
 * Define a data enrichment agent.
 *
 * Works with a chat model with tool calling support.
 */

import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { StateGraph } from "@langchain/langgraph";
import { z } from "zod";
import {
  AnyRecord,
  InputStateAnnotation,
  State,
  StateAnnotation,
} from "./state.js";
import { loadChatModel } from "./utils.js";
import { ensureConfiguration } from "./configuration.js";
import { MAIN_PROMPT } from "./prompts.js";
import { toolNode, TOOLS } from "./tools.js";
import { RunnableConfig } from "@langchain/core/runnables";

// Define the nodes

async function callAgentModel(
  state: State,
  config?: RunnableConfig,
): Promise<{
  messages: BaseMessage[];
  info?: AnyRecord;
  loopStep: number;
}> {
  const configuration = ensureConfiguration(config);
  const infoTool = tool(async (_args: AnyRecord) => {}, {
    name: "Info",
    description: "Call this when you have gathered all the relevant info",
    schema: state.extractionSchema,
  });

  const p = MAIN_PROMPT.replace(
    "{info}",
    JSON.stringify(state.extractionSchema, null, 2),
  ).replace("{topic}", state.topic);
  const messages = [new HumanMessage(p), ...state.messages];
  const rawModel = await loadChatModel(configuration.modelName);
  if (!rawModel.bindTools) {
    throw new Error("Chat model does not support tool binding");
  }
  const model = rawModel.bindTools([...TOOLS, infoTool]);
  const response = await model.invoke(messages);

  let info;
  if (
    ((response as AIMessage)?.tool_calls &&
      (response as AIMessage).tool_calls?.length) ||
    0
  ) {
    for (const tool_call of (response as AIMessage).tool_calls || []) {
      if (tool_call.name === "Info") {
        info = tool_call.args;
        break;
      }
    }
  }

  // If info was called, the agent is submitting a response. Ensure
  // it's not also trying to use other functions.
  if (info) {
    (response as AIMessage).tool_calls = (
      response as AIMessage
    ).tool_calls?.filter((tool_call) => tool_call.name === "Info");
  }

  return {
    messages: [response],
    info,
    loopStep: 1,
  };
}

const InfoIsSatisfactory = z.object({
  reason: z
    .array(z.string())
    .describe(
      "First, provide reasoning for why this is either good or bad as a final result. Must include at least 3 reasons.",
    ),
  is_satisfactory: z
    .boolean()
    .describe(
      "After providing your reasoning, provide a value indicating whether the result is satisfactory. If not, you will continue researching.",
    ),
});

async function callChecker(
  state: State,
  config?: RunnableConfig,
): Promise<{ messages: BaseMessage[] } | { info: AnyRecord }> {
  const configuration = ensureConfiguration(config);
  const p = MAIN_PROMPT.replace(
    "{info}",
    JSON.stringify(state.extractionSchema, null, 2),
  ).replace("{topic}", state.topic);
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

  const rawModel = await loadChatModel(configuration.modelName);
  const boundModel = rawModel.withStructuredOutput(InfoIsSatisfactory);
  const response = await boundModel.invoke(messages);

  const lm = state.messages[state.messages.length - 1];
  if (!(lm._getType() === "ai")) {
    throw new Error(
      `${callChecker.name} expects the last message in the state to be an AI message with tool calls. Got: ${lm._getType()}`,
    );
  }
  const lastMessage = lm as AIMessage;

  if (response.is_satisfactory) {
    try {
      return { info: presumedInfo };
    } catch (e) {
      return {
        messages: [
          new ToolMessage({
            tool_call_id: lastMessage.tool_calls?.[0]?.id || "",
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
          tool_call_id: lastMessage.tool_calls?.[0]?.id || "",
          content: JSON.stringify(response),
          name: "Info",
          additional_kwargs: { artifact: response },
        }),
      ],
    };
  }
}

function routeAfterAgent(state: State): "callChecker" | "tools" | "__end__" {
  const lastMessage = state.messages[state.messages.length - 1];

  if (
    ((lastMessage as AIMessage)?.tool_calls?.length || 0) > 0 &&
    (lastMessage as AIMessage)?.tool_calls?.[0]?.name === "Info"
  ) {
    return "callChecker";
  } else {
    return "tools";
  }
}

function routeAfterChecker(
  state: State,
  config?: RunnableConfig,
): "__end__" | "callAgentModel" {
  const configuration = ensureConfiguration(config);
  if (state.loopStep < configuration.maxInfoToolCalls) {
    if (!state.info) {
      return "callAgentModel";
    }
    const lastMessage = state.messages[state.messages.length - 1];
    if (
      lastMessage._getType() === "tool" &&
      (lastMessage as ToolMessage).status === "error"
    ) {
      // Research deemed unsatisfactory
      return "callAgentModel";
    }
    // It's great!
    return "__end__";
  } else {
    return "__end__";
  }
}

// Create the graph
const workflow = new StateGraph({
  stateSchema: StateAnnotation,
  input: InputStateAnnotation,
})
  .addNode("callAgentModel", callAgentModel)
  .addNode("callChecker", callChecker)
  .addNode("tools", toolNode)
  .addEdge("__start__", "callAgentModel")
  .addConditionalEdges("callAgentModel", routeAfterAgent)
  .addEdge("tools", "callAgentModel")
  .addConditionalEdges("callChecker", routeAfterChecker);

export const graph = workflow.compile();
graph.name = "ResearchTopic";
