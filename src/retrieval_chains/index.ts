import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatOpenAI } from "@langchain/openai";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";

export async function callChatWithDocumentContext(model: ChatOpenAI) {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the user's question.
        Context: {context}
        Question: {input}
    `);

    const loader = new CheerioWebBaseLoader('https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language')
    const documents = await loader.load();

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
        outputParser: new StringOutputParser(),
    });

    const response = await chain.invoke({
        input: `What's the LCEL?`,
        context: documents
    });

    return response;
}