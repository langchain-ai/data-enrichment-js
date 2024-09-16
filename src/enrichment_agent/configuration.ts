/**
 * Define the configurable parameters for the agent.
 */

import { RunnableConfig } from "@langchain/core/runnables";
import { MAIN_PROMPT } from "./prompts.js";

export interface Configuration {
  /**
   * The name of the language model to use for the agent.
   *
   * Should be in the form: provider/model-name.
   */
  modelName: string;

  /**
   * The main prompt template to use for the agent's interactions.
   *
   * Expects two template literals: ${info} and ${topic}.
   */
  prompt: string;

  /**
   * The maximum number of search results to return for each search query.
   */
  maxSearchResults: number;

  /**
   * The maximum number of times the Info tool can be called during a single interaction.
   */
  maxInfoToolCalls: number;

  /**
   * The maximum number of interaction loops allowed before the agent terminates.
   */
  maxLoops: number;
}

export function ensureConfiguration(config?: RunnableConfig): Configuration {
  /**
   * Ensure the defaults are populated.
   */
  const configurable = (config?.configurable as Record<string, any>) ?? {};
  return {
    modelName: configurable.modelName ?? "anthropic/claude-3-5-sonnet-20240620",
    prompt: configurable.prompt ?? MAIN_PROMPT,
    maxSearchResults: configurable.maxSearchResults ?? 10,
    maxInfoToolCalls: configurable.maxInfoToolCalls ?? 3,
    maxLoops: configurable.maxLoops ?? 6,
  };
}
