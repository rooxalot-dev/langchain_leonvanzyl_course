import readline from 'readline';
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { createRetrieverTool } from "langchain/tools/retriever";
import { WebBrowser } from "langchain/tools/webbrowser";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';


const createVectorStoreFromUrl = async (input: { url: string }) => {
    const { url } = input;

    const loader = new CheerioWebBaseLoader(url)
    const documents = await loader.load();
    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 200, chunkOverlap: 20 });
    const splitDocuments = await textSplitter.splitDocuments(documents);
    const vectorStore = await MemoryVectorStore.fromDocuments(splitDocuments, new OpenAIEmbeddings());

    return vectorStore;
}

export async function callAgentWithTools(model: ChatOpenAI, tavilyApiKey: string) {
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', `You're a helpful assistant call Max.`],
        new MessagesPlaceholder('chat_history'),
        ['human', '{input}'],
        new MessagesPlaceholder('agent_scratchpad'), // This placeholer needs to be inserted, since it'll be used internally and automatically by the agent and tools
    ]);

    const vectorStore = await createVectorStoreFromUrl({ url: 'https://js.langchain.com/v0.2/docs/concepts/#langchain-expression-language' });

    const retrieverTool = createRetrieverTool(vectorStore.asRetriever(), {
        name: 'lcel_search', // The name of the tool we're creating, it can be anything, but it's good to be a good identifier
        description: 'Use this tool when searching for information about Langchain Expression Language (LCEL)' // Tool's description, the more descritive the better since this is how the AI will know when to use it
    });

    // Create and assign tools
    const tools: any = [
        //new WebBrowser({ model, embeddings: new OpenAIEmbeddings(), verbose: true }),
        new TavilySearchResults({ apiKey: tavilyApiKey }),
        retrieverTool,
    ];

    // Create agent
    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        prompt: prompt,
        tools
    });

    // Create and execute agent
    const agentExecutor = new AgentExecutor({ agent, tools });

    const chatHistory: Array<BaseMessage> = [];

    // Create chatbot-like interation
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const askQuestion = () => {
        rl.question('User: ', async (question) => {
            if (question.toLowerCase() === 'exit') {
                rl.close();
                return;
            }

            const response = await agentExecutor.invoke({ input: question, chat_history: chatHistory });
            console.log('Agent: ', response.output);

            chatHistory.push(new HumanMessage(question));
            chatHistory.push(new AIMessage(response.output));

            askQuestion();
        });
    };

    askQuestion();
}