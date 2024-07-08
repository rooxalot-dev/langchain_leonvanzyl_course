import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug
    });

    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        You are a comedian. Tell me a joke based on the following word: {inputWord}
    `);

    const chain = templatePrompt.pipe(model);
    const aiChainResponse = await chain.invoke({ inputWord: 'Amazon' });
    const usedTokens = aiChainResponse.usage_metadata?.total_tokens;
    const content = aiChainResponse.content;

    console.log(`Used Tokens: ${usedTokens} - Content: ${content}`);
};

start();