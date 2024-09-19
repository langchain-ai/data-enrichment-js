import { describe, it, expect } from "@jest/globals";
import { graph } from "../src/enrichment_agent/graph.js";

describe("Researcher", () => {
  it("should initialize and compile the graph", () => {
    expect(graph).toBeDefined();
    expect(graph.name).toBe("ResearchTopic");
  });

  const extractionSchema = {
    type: "object",
    properties: {
      founder: {
        type: "string",
        description: "The name of the company founder.",
      },
      websiteUrl: {
        type: "string",
        description:
          "Website URL of the company, e.g.: https://openai.com/, or https://microsoft.com",
      },
      products_sold: {
        type: "array",
        items: { type: "string" },
        description: "A list of products sold by the company.",
      },
    },
    required: ["founder", "websiteUrl", "products_sold"],
  };

  it("Simple runthrough", async () => {
    const res = await graph.invoke({
      topic: "LangChain",
      extractionSchema: extractionSchema,
    });

    expect(res.info).toBeDefined();
    expect(res.info.founder.toLowerCase()).toContain("harrison");
  }, 100_000);

  const arrayExtractionSchema = {
    type: "object",
    properties: {
      providers: {
        type: "array",
        items: {
          type: "object",
          properties: {
            name: { type: "string", description: "Company name" },
            technology_summary: {
              type: "string",
              description:
                "Brief summary of their chip technology for LLM training",
            },
            current_market_share: {
              type: "string",
              description:
                "Estimated current market share percentage or position",
            },
            future_outlook: {
              type: "string",
              description:
                "Brief paragraph on future prospects and developments",
            },
          },
          required: [
            "name",
            "technology_summary",
            "current_market_share",
            "future_outlook",
          ],
        },
        description: "List of top chip providers for LLM Training",
      },
      overall_market_trends: {
        type: "string",
        description: "Brief paragraph on general trends in the LLM chip market",
      },
    },
    required: ["providers", "overall_market_trends"],
  };

  it("Researcher list type", async () => {
    const res = await graph.invoke({
      topic: "Top 5 chip providers for LLM training",
      extractionSchema: arrayExtractionSchema,
    });

    const info = res.info;
    expect(info.providers).toBeDefined();
    expect(Array.isArray(info.providers)).toBe(true);
    expect(info.providers.length).toBe(5);

    const nvidiaPresent = info.providers.some(
      (provider: { name: string }) =>
        provider.name.toLowerCase().trim() === "nvidia",
    );
    expect(nvidiaPresent).toBe(true);

    info.providers.forEach(
      (provider: {
        name: any;
        technology_summary: any;
        current_market_share: any;
        future_outlook: any;
      }) => {
        expect(provider.name).toBeDefined();
        expect(provider.technology_summary).toBeDefined();
        expect(provider.current_market_share).toBeDefined();
        expect(provider.future_outlook).toBeDefined();
      },
    );

    expect(info.overall_market_trends).toBeDefined();
    expect(typeof info.overall_market_trends).toBe("string");
    expect(info.overall_market_trends.length).toBeGreaterThan(0);
  }, 100_000);
});
