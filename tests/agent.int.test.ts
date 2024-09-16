import { describe, it, expect } from "@jest/globals";
import { graph } from "../src/enrichment_agent/graph.js";
describe("Researcher", () => {
  it("should initialize and compile the graph", () => {
    expect(graph).toBeDefined();
    expect(graph.name).toBe("ResearchTopic");
  });

  it("Simple runthrough", async () => {
    const enrichmentSchema = {
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
      },
      required: ["founder", "websiteUrl"],
    };
    const res = await graph.invoke({
      topic: "LangChain",
      extractionSchema: enrichmentSchema,
    });
    expect(res.info).toBeDefined();
    const info = res.info;
    expect((info.founder as string).toLowerCase()).toContain("harrison");
  }, 100_000);
});
