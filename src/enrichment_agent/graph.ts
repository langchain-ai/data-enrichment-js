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
import { RunnableConfig } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { StateGraph } from "@langchain/langgraph";
import { z } from "zod";

import {
  ConfigurationAnnotation,
  ensureConfiguration,
} from "./configuration.js";
import { AnyRecord, InputStateAnnotation, StateAnnotation } from "./state.js";
import { toolNode, TOOLS } from "./tools.js";
import { loadChatModel } from "./utils.js";

/**
 * Calls the primary Language Model (LLM) to decide on the next research action.
 *
 * This function performs the following steps:
 * 1. Initializes configuration and sets up the 'Info' tool, which is the user-defined extraction schema.
 * 2. Prepares the prompt and message history for the LLM.
 * 3. Initializes and configures the LLM with available tools.
 * 4. Invokes the LLM and processes its response.
 * 5. Handles the LLM's decision to either continue research or submit final info.
 *
 * @param state - The current state of the research process.
 * @param config - Optional configuration for the runnable.
 * @returns A Promise resolving to an object containing:
 *   - messages: An array of BaseMessage objects representing the LLM's response.
 *   - info: An optional AnyRecord containing the extracted information if the LLM decided to submit final info.
 *   - loopStep: A number indicating the current step in the research loop.
 */

async function callAgentModel(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<typeof StateAnnotation.Update> {
  const configuration = ensureConfiguration(config);
  // First, define the info tool. This uses the user-provided
  // json schema to define the research targets
  const infoTool = tool(async (_args: AnyRecord) => {}, {
    name: "Info",
    description: "Call this when you have gathered all the relevant info",
    schema: state.extractionSchema,
  });
  // Next, load the model
  const rawModel = await loadChatModel(configuration.model);
  if (!rawModel.bindTools) {
    throw new Error("Chat model does not support tool binding");
  }
  const model = rawModel.bindTools([...TOOLS, infoTool], {
    tool_choice: "any",
  });

  // Format the schema into the configurable system prompt
  const p = configuration.prompt
    .replace("{info}", JSON.stringify(state.extractionSchema, null, 2))
    .replace("{topic}", state.topic);
  const messages = [{ role: "user", content: p }, ...state.messages];

  // Next, we'll call the model.
  const response: AIMessage = await model.invoke(messages);
  const responseMessages = [response];

  // If the model has collected enough information to fill uot
  // the provided schema, great! It will call the "Info" tool
  // We've decided to track this as a separate state variable
  let info;
  if ((response?.tool_calls && response.tool_calls?.length) || 0) {
    for (const tool_call of response.tool_calls || []) {
      if (tool_call.name === "Info") {
        info = tool_call.args;
        // If info was called, the agent is submitting a response.
        // (it's not actually a function to call, it's a schema to extract)
        // To ensure that the graph doesn'tend up in an invalid state
        // (where the AI has called tools but no tool message has been provided)
        // we will drop any extra tool_calls.
        response.tool_calls = response.tool_calls?.filter(
          (tool_call) => tool_call.name === "Info",
        );
        break;
      }
    }
  } else {
    // If LLM didn't respect the tool_choice
    responseMessages.push(
      new HumanMessage("Please respond by calling one of the provided tools."),
    );
  }

  return {
    messages: responseMessages,
    info,
    // This increments the step counter.
    // We configure a max step count to avoid infinite research loops
    loopStep: 1,
  };
}

/**
 * Validate whether the current extracted info is satisfactory and complete.
 */
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
  improvement_instructions: z
    .string()
    .optional()
    .describe(
      "If the result is not satisfactory, provide clear and specific instructions on what needs to be improved or added to make the information satisfactory. This should include details on missing information, areas that need more depth, or specific aspects to focus on in further research.",
    ),
});

/**
 * Validates the quality of the data enrichment agent's output.
 *
 * This function performs the following steps:
 * 1. Prepares the initial prompt using the main prompt template.
 * 2. Constructs a message history for the model.
 * 3. Prepares a checker prompt to evaluate the presumed info.
 * 4. Initializes and configures a language model with structured output.
 * 5. Invokes the model to assess the quality of the gathered information.
 * 6. Processes the model's response and determines if the info is satisfactory.
 *
 * @param state - The current state of the research process.
 * @param config - Optional configuration for the runnable.
 * @returns A Promise resolving to an object containing either:
 *   - messages: An array of BaseMessage objects if the info is not satisfactory.
 *   - info: An AnyRecord containing the extracted information if it is satisfactory.
 */
async function reflect(
  state: typeof StateAnnotation.State,
  config: RunnableConfig,
): Promise<{ messages: BaseMessage[] } | { info: AnyRecord }> {
  const configuration = ensureConfiguration(config);
  const presumedInfo = state.info; // The current extracted result
  const lm = state.messages[state.messages.length - 1];
  if (!(lm._getType() === "ai")) {
    throw new Error(
      `${reflect.name} expects the last message in the state to be an AI message with tool calls. Got: ${lm._getType()}`,
    );
  }
  const lastMessage = lm as AIMessage;

  // Load the configured model & provide the reflection/critique schema
  const rawModel = await loadChatModel(configuration.model);
  const boundModel = rawModel.withStructuredOutput(InfoIsSatisfactory);
  // Template in the conversation history:
  const p = configuration.prompt
    .replace("{info}", JSON.stringify(state.extractionSchema, null, 2))
    .replace("{topic}", state.topic);
  const messages = [
    { role: "user", content: p },
    ...state.messages.slice(0, -1),
  ];

  const checker_prompt = `I am thinking of calling the info tool with the info below. \
Is this good? Give your reasoning as well. \
You can encourage the Assistant to look at specific URLs if that seems relevant, or do more searches.
If you don't think it is good, you should be very specific about what could be improved.

{presumed_info}`;
  const p1 = checker_prompt.replace(
    "{presumed_info}",
    JSON.stringify(presumedInfo ?? {}, null, 2),
  );
  messages.push({ role: "user", content: p1 });

  // Call the model
  const response = await boundModel.invoke(messages);
  if (response.is_satisfactory && presumedInfo) {
    return {
      info: presumedInfo,
      messages: [
        new ToolMessage({
          tool_call_id: lastMessage.tool_calls?.[0]?.id || "",
          content: response.reason.join("\n"),
          name: "Info",
          artifact: response,
          status: "success",
        }),
      ],
    };
  } else {
    return {
      messages: [
        new ToolMessage({
          tool_call_id: lastMessage.tool_calls?.[0]?.id || "",
          content: `Unsatisfactory response:\n${response.improvement_instructions}`,
          name: "Info",
          artifact: response,
          status: "error",
        }),
      ],
    };
  }
}

/**
 * Determines the next step in the research process based on the agent's last action.
 *
 * @param state - The current state of the research process.
 * @returns "reflect" if the agent has called the "Info" tool to submit findings,
 *          "tools" if the agent has called any other tool or no tool at all.
 */
function routeAfterAgent(
  state: typeof StateAnnotation.State,
): "callAgentModel" | "reflect" | "tools" | "__end__" {
  const lastMessage: AIMessage = state.messages[state.messages.length - 1];

  // If for some reason the last message is not an AIMessage
  // (if you've modified this template and broken one of the assumptions)
  // ensure the system doesn't crash but instead tries to recover by calling the agent model again.
  if (lastMessage._getType() !== "ai") {
    return "callAgentModel";
  }

  // If the "Info" tool was called, then the model provided its extraction output. Reflect on the result
  if (lastMessage.tool_calls && lastMessage.tool_calls[0]?.name === "Info") {
    return "reflect";
  }

  // The last message is a tool call that is not "Info" (extraction output)
  return "tools";
}

/**
 * Schedules the next node after the checker's evaluation.
 *
 * This function determines whether to continue the research process or end it
 * based on the checker's evaluation and the current state of the research.
 *
 * @param state - The current state of the research process.
 * @param config - The configuration for the research process.
 * @returns "__end__" if the research should end, "callAgentModel" if it should continue.
 */
function routeAfterChecker(
  state: typeof StateAnnotation.State,
  config?: RunnableConfig,
): "__end__" | "callAgentModel" {
  const configuration = ensureConfiguration(config);
  const lastMessage = state.messages[state.messages.length - 1];

  if (state.loopStep < configuration.maxLoops) {
    if (!state.info) {
      return "callAgentModel";
    }
    if (lastMessage._getType() !== "tool") {
      throw new Error(
        `routeAfterChecker expected a tool message. Received: ${lastMessage._getType()}.`,
      );
    }
    if ((lastMessage as ToolMessage).status === "error") {
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
const workflow = new StateGraph(
  {
    stateSchema: StateAnnotation,
    input: InputStateAnnotation,
  },
  ConfigurationAnnotation,
)
  .addNode("callAgentModel", callAgentModel)
  .addNode("reflect", reflect)
  .addNode("tools", toolNode)
  .addEdge("__start__", "callAgentModel")
  .addConditionalEdges("callAgentModel", routeAfterAgent)
  .addEdge("tools", "callAgentModel")
  .addConditionalEdges("reflect", routeAfterChecker);

export const graph = workflow.compile();
graph.name = "ResearchTopic";
