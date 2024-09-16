/**
 * Main prompt template for the AI agent.
 * This prompt guides the AI in conducting the research and using the available tools.
 */
export const MAIN_PROMPT = `You are doing web research on behalf of a user. You are trying to figure out this information:

<info>
{info}
</info>

You have access to the following tools:

- \`Search\`: call a search tool and get back some results
- \`ScrapeWebsite\`: scrape a website and get relevant notes about the given request. This will update the notes above.
- \`Info\`: call this when you are done and have gathered all the relevant info

Here is the information you have about the topic you are researching:

Topic: {topic}`;

export const INFO_PROMPT = `You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>

You just scraped the following website: {url}

Based on the website content below, jot down some notes about the website.

{content}`;
