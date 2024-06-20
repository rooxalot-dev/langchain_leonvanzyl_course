import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug

    });
    const aiResponse = await model.invoke('Which OpenAI version am I using right now?');
    const usedTokens = aiResponse.usage_metadata?.total_tokens;
    const content = aiResponse.content;
    console.log(`Used Tokens: ${usedTokens} - Content: ${content}`);
};

start();