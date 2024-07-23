import readline from 'readline';
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { WebBrowser } from "langchain/tools/webbrowser";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";


export async function callAgentWithTools(model: ChatOpenAI, tavilyApiKey: string) {
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', `You're a helpful assistant call Max.`],
        ['human', '{input}'],
        new MessagesPlaceholder('agent_scratchpad'), // This placeholer needs to be inserted, since it'll be used internally and automatically by the agent and tools
    ]);

    // Create and assign tools
    const tools: any = [
        //new WebBrowser({ model, embeddings: new OpenAIEmbeddings(), verbose: true }),
        new TavilySearchResults({ apiKey: tavilyApiKey })
    ];

    // Create agent
    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        prompt: prompt,
        tools
    });

    // Create and execute agent
    const agentExecutor = new AgentExecutor({ agent, tools });

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    

    const askQuestion = () => {
        rl.question('User: ', async (question) => {
            if (question.toLowerCase() === 'exit') rl.close();

            const response = await agentExecutor.invoke({ input: question });
            console.log('Agent: ', response.output);

            askQuestion();
        });
    };

    askQuestion();
}