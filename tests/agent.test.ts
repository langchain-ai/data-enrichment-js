import { describe, it, expect } from "@jest/globals";
import { graph } from "../src/agent.js";

describe("Web Research Agent", () => {
  it("should initialize and compile the graph", () => {
    expect(graph).toBeDefined();
    expect(graph.name).toBe("ResearchTopic");
  });

  // TODO: Add more test cases for individual nodes, routing logic, tool integration, and output validation
});
