import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug
    });

    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        You are a comedian. Tell me a joke on 2 two lines or more, based on the following word: {inputWord}
    `);
    const stringParser = new StringOutputParser();

    const chain = templatePrompt.pipe(model).pipe(stringParser);
    const aiChainStringResponse = await chain.invoke({ inputWord: 'Amazon' });

    console.log(aiChainStringResponse);
    //console.log(`Used Tokens: ${usedTokens} - Content: ${content}`);
};

start();