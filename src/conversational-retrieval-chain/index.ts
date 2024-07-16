import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";


const createVectorStoreFromUrl = async (input: { url: string }) => {
    const { url } = input;

    const loader = new CheerioWebBaseLoader(url)
    const documents = await loader.load();
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });
    const splitDocuments = await textSplitter.splitDocuments(documents);
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, new OpenAIEmbeddings());

    return vectorStore;
}

const createChain = async (model: ChatOpenAI, vectorStore: MemoryVectorStore) => {
    const prompt = ChatPromptTemplate.fromTemplate(`
        Answer the user's question.
        Context: {context}
        Question: {input}
    `);

    const combineDocumentsChain = await createStuffDocumentsChain({ llm: model, prompt });
    const conversationalChain = await createRetrievalChain({ retriever: vectorStore.asRetriever(), combineDocsChain: combineDocumentsChain });

    return conversationalChain;
}

export async function callConversationHistoryWithDocumentContext(model: ChatOpenAI) {
    const vectorStore = await createVectorStoreFromUrl({ url: 'https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language' });    
    const chain = await createChain(model, vectorStore);
    const response = await chain.invoke({ input: `What's the LCEL?` });

    return response;
}