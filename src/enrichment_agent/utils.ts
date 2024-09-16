import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
  MessageContent,
  MessageContentComplex,
} from "@langchain/core/messages";
import { initChatModel } from "langchain/chat_models/universal";

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

/**
 * Load a chat model from a fully specified name.
 * @param fullySpecifiedName - String in the format 'provider/model' or 'provider/account/provider/model'.
 * @returns A Promise that resolves to a BaseChatModel instance.
 */
export async function loadChatModel(
  fullySpecifiedName: string,
): Promise<BaseChatModel> {
  const index = fullySpecifiedName.indexOf("/");
  if (index === -1) {
    // If there's no "/", assume it's just the model
    return await initChatModel(fullySpecifiedName);
  } else {
    const provider = fullySpecifiedName.slice(0, index);
    const model = fullySpecifiedName.slice(index + 1);
    return await initChatModel(model, { modelProvider: provider });
  }
}
