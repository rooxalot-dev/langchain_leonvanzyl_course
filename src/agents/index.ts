import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createOpenAIFunctionsAgent, AgentExecutor } from "langchain/agents";
import { WebBrowser } from "langchain/tools/webbrowser";


export async function callAgentWithTools(model: ChatOpenAI) {
    const prompt = ChatPromptTemplate.fromMessages([
        ['system', `You're a helpful assistant call Max.`],
        new MessagesPlaceholder('agent_scratchpad'), // This placeholer needs to be inserted, since it'll be used internally and automatically by the agent and tools
        ['human', '{input}'],
    ]);

    // Create and assign tools
    const tools: any = [
        new WebBrowser({ model, embeddings: new OpenAIEmbeddings(), verbose: true }),
    ];

    // Create agent
    const agent = await createOpenAIFunctionsAgent({
        llm: model,
        prompt: prompt,
        tools
    });

    // Create and execute agent
    const agentExecutor = new AgentExecutor({ agent, tools });
    const response = await agentExecutor.invoke({ input: `
        Consider that the current year is ${new Date().getFullYear()}, answer the following question: 'What are the latest news for Alan Wake 2?'
    ` });

    return response;
}