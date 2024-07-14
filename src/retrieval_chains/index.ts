import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";


export async function callChatWithDocumentContext(model: ChatOpenAI) {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the user's question.
        Context: {context}
        Question: {input}
    `);

    const loader = new CheerioWebBaseLoader('https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language')
    const documents = await loader.load();
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });
    const splitDocuments = await textSplitter.splitDocuments(documents);
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocuments,
        new OpenAIEmbeddings()
    );

    const combineDocumentsChain = await createStuffDocumentsChain({
        llm: model,
        prompt,
        outputParser: new StringOutputParser(),
    });

    const chain = await createRetrievalChain({
        retriever: vectorStore.asRetriever(),
        combineDocsChain: combineDocumentsChain
    });


    const response = await chain.invoke({
        input: `What's the LCEL?`,
        context: documents
    });

    return response;
}