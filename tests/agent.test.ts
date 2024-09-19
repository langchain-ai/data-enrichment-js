import { describe, it, expect } from "@jest/globals";
import { graph } from "../src/enrichment_agent/graph.js";

describe("Web Research Agent", () => {
  beforeAll(() => {
    process.env.TAVILY_API_KEY = "dummy";
  });

  it("should initialize and compile the graph", () => {
    expect(graph).toBeDefined();
    expect(graph.name).toBe("ResearchTopic");
  });

  // TODO: Add more test cases for individual nodes, routing logic, tool integration, and output validation
});
