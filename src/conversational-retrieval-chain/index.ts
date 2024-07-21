import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { BaseMessage, AIMessage, HumanMessage } from "@langchain/core/messages";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";


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
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', `Answer the user's question based on the following context: {context}`],
        new MessagesPlaceholder('chat_history'),
        ['user', '{input}'],
    ]);

    // Retriever prompt created to be used in the HistoryAwareRetriever
    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder('chat_history'),
        ['user', '{input}'],
        ['user', 'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation']
    ]);

    const combineDocumentsChain = await createStuffDocumentsChain({ llm: model, prompt });

    // This history aware retriever "buffs" the normal retriever, enabling the documents seach to use both the user input and the chat history to search relevant data
    const historyAwareRetriever = await createHistoryAwareRetriever({
        llm: model,
        retriever: vectorStore.asRetriever(),
        rephrasePrompt: retrieverPrompt
    });
    const conversationalChain = await createRetrievalChain({ retriever: historyAwareRetriever, combineDocsChain: combineDocumentsChain });

    return conversationalChain;
}

export async function callConversationHistoryWithDocumentContext(model: ChatOpenAI) {
    const vectorStore = await createVectorStoreFromUrl({ url: 'https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language' });    
    const chain = await createChain(model, vectorStore);

    const chatHistory: BaseMessage[] = [
        new HumanMessage('Hello!'),
        new AIMessage('Hi, how can I help you?'),
        new HumanMessage('My name is Rodrigo!'),
        new AIMessage('Hello! Nice to meet you. How can I help you?'),
        new HumanMessage('What are you!'),
        new AIMessage(`I'm a very helpful AI that can help you with many topics.`),
    ];

    const response = await chain.invoke({ 
        input: `Can you answer what is the LCEL? Besides that, can you tell me what's my name?`,
        chat_history: chatHistory,
    });

    return response.answer;
}