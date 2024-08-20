import {
  MessageContent,
  MessageContentComplex,
} from "@langchain/core/messages";

export function curry<
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  F extends (...args: any[]) => any,
  P extends Partial<Parameters<F>[0]> = Partial<Parameters<F>[0]>,
>(fn: F, partialArg: P) {
  return function (
    this: unknown,
    arg: Omit<Parameters<F>[0], keyof P> & Partial<P>,
    ...rest: Parameters<F> extends [unknown, ...infer R] ? R : never
  ): ReturnType<F> {
    const mergedArg = { ...partialArg, ...arg } as Parameters<F>[0];
    return fn.apply(this, [mergedArg, ...rest]) as ReturnType<F>;
  };
}

/**
 * Helper function to extract text content from a complex message.
 *
 * @param content - The complex message content to process
 * @returns The extracted text content
 */
function getSingleTextContent(content: MessageContentComplex) {
  if (content?.type === "text") {
    return content.text;
  } else if (content.type === "array") {
    return content.content.map(getSingleTextContent).join(" ");
  }
  return "";
}

/**
 * Helper function to extract text content from various message types.
 *
 * @param content - The message content to process
 * @returns The extracted text content
 */
export function getTextContent(content: MessageContent): string {
  if (typeof content === "string") {
    return content;
  } else if (Array.isArray(content)) {
    return content.map(getSingleTextContent).join(" ");
  }
  return "";
}
