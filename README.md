# LangGraph Studio Data Enrichment Template

[![CI](https://github.com/langchain-ai/data-enrichment-js/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/data-enrichment-js/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/data-enrichment-js/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/data-enrichment-js/actions/workflows/integration-tests.yml)
[![Open in - LangGraph Studio](https://img.shields.io/badge/Open_in-LangGraph_Studio-00324d.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4NS4zMzMiIGhlaWdodD0iODUuMzMzIiB2ZXJzaW9uPSIxLjAiIHZpZXdCb3g9IjAgMCA2NCA2NCI+PHBhdGggZD0iTTEzIDcuOGMtNi4zIDMuMS03LjEgNi4zLTYuOCAyNS43LjQgMjQuNi4zIDI0LjUgMjUuOSAyNC41QzU3LjUgNTggNTggNTcuNSA1OCAzMi4zIDU4IDcuMyA1Ni43IDYgMzIgNmMtMTIuOCAwLTE2LjEuMy0xOSAxLjhtMzcuNiAxNi42YzIuOCAyLjggMy40IDQuMiAzLjQgNy42cy0uNiA0LjgtMy40IDcuNkw0Ny4yIDQzSDE2LjhsLTMuNC0zLjRjLTQuOC00LjgtNC44LTEwLjQgMC0xNS4ybDMuNC0zLjRoMzAuNHoiLz48cGF0aCBkPSJNMTguOSAyNS42Yy0xLjEgMS4zLTEgMS43LjQgMi41LjkuNiAxLjcgMS44IDEuNyAyLjcgMCAxIC43IDIuOCAxLjYgNC4xIDEuNCAxLjkgMS40IDIuNS4zIDMuMi0xIC42LS42LjkgMS40LjkgMS41IDAgMi43LS41IDIuNy0xIDAtLjYgMS4xLS44IDIuNi0uNGwyLjYuNy0xLjgtMi45Yy01LjktOS4zLTkuNC0xMi4zLTExLjUtOS44TTM5IDI2YzAgMS4xLS45IDIuNS0yIDMuMi0yLjQgMS41LTIuNiAzLjQtLjUgNC4yLjguMyAyIDEuNyAyLjUgMy4xLjYgMS41IDEuNCAyLjMgMiAyIDEuNS0uOSAxLjItMy41LS40LTMuNS0yLjEgMC0yLjgtMi44LS44LTMuMyAxLjYtLjQgMS42LS41IDAtLjYtMS4xLS4xLTEuNS0uNi0xLjItMS42LjctMS43IDMuMy0yLjEgMy41LS41LjEuNS4yIDEuNi4zIDIuMiAwIC43LjkgMS40IDEuOSAxLjYgMi4xLjQgMi4zLTIuMy4yLTMuMi0uOC0uMy0yLTEuNy0yLjUtMy4xLTEuMS0zLTMtMy4zLTMtLjUiLz48L3N2Zz4=)](https://langgraph-studio.vercel.app/templates/open?githubUrl=https://github.com/langchain-ai/data-enrichment-js)

This is a starter project to help you get started with developing a data enrichment agent using [LangGraph.js](https://github.com/langchain-ai/langgraphjs) in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio).

![](/static/studio.png)

It contains an example graph exported from `src/enrichment_agent/graph.ts` that implements a research assistant capable of automatically gathering information on various topics from the web.

## What it does

The enrichment agent:

1. Takes a research **topic** and requested **extractionSchema** as input
2. Searches the web for relevant information
3. Reads and extracts key details from websites
4. Organizes the findings into the requested structured format
5. Validates the gathered information for completeness and accuracy

By default, it's set up to gather information based on the user-provided schema passed through the `extractionSchema` key in the state.

## Getting Started

You will need the latest versions of `@langchain/langgraph` and `@langchain/core`. See these instructions for help upgrading an [existing project](https://langchain-ai.github.io/langgraphjs/how-tos/manage-ecosystem-dependencies/).

Assuming you have already [installed LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download), to set up:

1. Create a `.env` file.

```bash
cp .env.example .env
```

2. Define required API keys in your `.env` file.

The primary [search tool](./src/enrichment_agent/tools.ts) [^1] used is [Tavily](https://tavily.com/). Create an API key [here](https://app.tavily.com/sign-in).

<!--
Setup instruction auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
-->

<details>
<summary>Setup for `modelName`</summary>
The `llm` configuration defaults are shown below:

```yaml
modelName: anthropic/claude-3-5-sonnet-20240620
```

Follow the instructions below to get set up, or pick one of the additional options.

### Anthropic Chat Models

To use Anthropic's chat models:

1. Sign up for an [Anthropic API key](https://console.anthropic.com/) if you haven't already.
2. Once you have your API key, add it to your `.env` file:

```
ANTHROPIC_API_KEY=your-api-key
```

### Fireworks Chat Models

To use Fireworks AI's chat models:

1. Sign up for a [Fireworks AI account](https://app.fireworks.ai/signup) and obtain an API key.
2. Add your Fireworks AI API key to your `.env` file:

```
FIREWORKS_API_KEY=your-api-key
```

#### OpenAI Chat Models

To use OpenAI's chat models:

1. Sign up for an [OpenAI API key](https://platform.openai.com/signup).
2. Once you have your API key, add it to your `.env` file:

```
OPENAI_API_KEY=your-api-key
```

</details>

<!--
End setup instructions
-->

3. Customize whatever you'd like in the code.
4. Open the folder LangGraph Studio!

## How to customize

1. **Customize research targets**: Provide a custom `extractionSchema` when calling the graph to gather different types of information.
2. **Select a different model**: We default to anthropic (claude-3-5-sonnet-20240620). You can select a compatible chat model using `provider/model-name` via configuration. Example: `openai/gpt-4o-mini`.
3. **Customize the prompt**: We provide a default prompt in [src/enrichment_agent/prompts.ts](./src/enrichment_agent/prompts.ts). You can easily update this via configuration in the studio.

You can also quickly extend this template by:

- Adding new tools and API connections in [src/enrichment_agent/tools.ts](./src/enrichment_agent/tools.ts). These are just any TypeScript functions.
- Adding additional steps in [src/enrichment_agent/graph.ts](./src/enrichment_agent/graph.ts). Concerned about hallucination? Add a fact-checking step!

## Development

While iterating on your graph, you can edit past state and rerun your app from past states to debug specific nodes. Local changes will be automatically applied via hot reload. Try adding an interrupt before the agent calls tools, updating the default system message in [src/enrichment_agent/utils.ts](./src/enrichment_agent/utils.ts) to take on a persona, or adding additional nodes and edges!

Follow up requests will be appended to the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

You can find the latest (under construction) docs on [LangGraph.js](https://langchain-ai.github.io/langgraphjs/) here, including examples and other references. Using those guides can help you pick the right patterns to adapt here for your use case.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates.

[^1]: https://js.langchain.com/docs/concepts#tools

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "agent": {
      "type": "object",
      "properties": {
        "modelName": {
          "type": "string",
          "default": "anthropic/claude-3-5-sonnet-20240620",
          "description": "The name of the language model to use for the agent. Should be in the form: provider/model-name.",
          "environment": [
            {
              "value": "anthropic/claude-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.0",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-2.1",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-5-sonnet-20240620",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-haiku-20240307",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-opus-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-3-sonnet-20240229",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "anthropic/claude-instant-1.2",
              "variables": "ANTHROPIC_API_KEY"
            },
            {
              "value": "fireworks/gemma2-9b-it",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-70b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-70b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-8b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3-8b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-405b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-405b-instruct-long",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-70b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/llama-v3p1-8b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x22b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x7b-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mixtral-8x7b-instruct-hf",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/mythomax-l2-13b",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/phi-3-vision-128k-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/phi-3p5-vision-instruct",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/starcoder-16b",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "fireworks/yi-large",
              "variables": "FIREWORKS_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0125",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0301",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-1106",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-3.5-turbo-16k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0125-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-1106-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0314",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-32k-0613",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-turbo-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4-vision-preview",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o",
              "variables": "OPENAI_API_KEY"
            },
            {
              "value": "openai/gpt-4o-mini",
              "variables": "OPENAI_API_KEY"
            }
          ]
        }
      },
      "environment": [
        "TAVILY_API_KEY"
      ]
    }
  }
}
-->
