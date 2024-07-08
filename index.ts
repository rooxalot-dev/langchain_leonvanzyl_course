import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser, CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug
    });

    async function callStringOutputParser() {
        const templatePrompt = ChatPromptTemplate.fromTemplate(`
            You are a comedian. Tell me a joke on two lines or more, based on the following word: {inputWord}. Never return less than two line.
        `);
        const stringParser = new StringOutputParser();
    
        const chain = templatePrompt.pipe(model).pipe(stringParser);
        const aiChainStringResponse = await chain.invoke({ inputWord: 'Amazon' });
    
        return aiChainStringResponse;
    }

    async function callListOutputParser() {
        const templatePrompt = ChatPromptTemplate.fromTemplate(`
            Provide at least 5 synonyms, separated by comma, for the following word: {word}. Never return less than 5 synonyms.
        `);
        const listParser = new CommaSeparatedListOutputParser();
    
        const chain = templatePrompt.pipe(model).pipe(listParser);
        const aiChainListResponse = await chain.invoke({ word: 'Door' });
    
        return aiChainListResponse;
    }

    // console.log(await callStringOutputParser());
    console.log(await callListOutputParser());
};

start();