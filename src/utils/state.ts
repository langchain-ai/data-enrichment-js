import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { type BaseMessage } from "@langchain/core/messages";
import { z } from "zod";
/**
 * A graph's StateAnnotation defines three main thing:
 * 1. The structure of the data to be passed between nodes (which "channels" to read from/write to and their types)
 * 2. Default values each field
 * 3. Rducers for the state's. Reducers are functions that determine how to apply updates to the state.
 * See [Reducers](https://langchain-ai.github.io/langgraphjs/concepts/low_level/#reducers) for more information.
 */

export const StateAnnotation = Annotation.Root({
  /**
   * Messages track the primary execution state of the agent.
   *
   * Typically accumulates a pattern of:
   *
   * 1. HumanMessage - user input
   * 2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
   *     information
   * 3. ToolMessage(s) - the responses (or errors) from the executed tools
   *
   *     (... repeat steps 2 and 3 as needed ...)
   * 4. AIMessage without .tool_calls - agent responding in unstructured
   *     format to the user.
   *
   * 5. HumanMessage - user responds with the next conversational turn.
   *
   *     (... repeat steps 2-5 as needed ... )
   *
   * Merges two lists of messages, updating existing messages by ID.
   *
   * By default, this ensures the state is "append-only", unless the
   * new message has the same ID as an existing message.
   *
   * Returns:
   *     A new list of messages with the messages from \`right\` merged into \`left\`.
   *     If a message in \`right\` has the same ID as a message in \`left\`, the
   *     message from \`right\` will replace the message from \`left\`.`
   */
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  topic: Annotation<string>(),
  /**
   * The info state trackes the current extracted data for the given topic,
   * conforming to the provided schema.
   */
  info: Annotation<z.infer<z.ZodObject<z.ZodRawShape>>>(),
  /**
   * The schema defines the information the agent is tasked with filling out.
   */
  schema: Annotation<z.ZodObject<z.ZodRawShape>>(),
  // Feel free to add additional attributes to your state as needed.
  // Common examples include retrieved documents, extracted entities, API connections, etc.
});

export type State = typeof StateAnnotation.State;
