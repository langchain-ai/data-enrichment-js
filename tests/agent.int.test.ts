import { describe, it, expect } from "@jest/globals";
import { graph } from "../src/agent.js";
import { z } from "zod";
describe("Researcher", () => {
  it("should initialize and compile the graph", () => {
    expect(graph).toBeDefined();
    expect(graph.name).toBe("ResearchTopic");
  });
  
  it("Simple runthrough", async () => {
    const enrichmentSchema = z.object({
      founder: z.string().describe("The name of the company founder."),
      websiteUrl: z
        .string()
        .describe(
          "Website URL of the company, e.g.: https://openai.com/, or https://microsoft.com",
        ),
    });
    const res = await graph.invoke({
      topic: "LangChain",
      schema: enrichmentSchema,
    });
    expect(res.info).toBeDefined();
    const info = res.info;
    expect((info.founder as string).toLowerCase()).toContain("harrison");
  }, 100_000);
});
