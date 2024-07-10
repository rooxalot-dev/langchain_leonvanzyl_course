import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser, CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { ChatOpenAI } from '@langchain/openai';

export async function callStringOutputParser(model: ChatOpenAI) {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        You are a comedian. Tell me a joke on two lines or more, based on the following word: {inputWord}. Never return less than two line.
    `);
    const stringParser = new StringOutputParser();

    const chain = templatePrompt.pipe(model).pipe(stringParser);
    const aiChainStringResponse = await chain.invoke({ inputWord: 'Amazon' });

    return aiChainStringResponse;
}

export async function callListOutputParser(model: ChatOpenAI) {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        Provide at least 10 synonyms, separated by comma, for the following word: {word}.
    `);
    const listParser = new CommaSeparatedListOutputParser();

    const chain = templatePrompt.pipe(model).pipe(listParser);
    const aiChainListResponse = await chain.invoke({ word: 'Door' });

    return aiChainListResponse;
}

export async function callStructuredOutputParser(model: ChatOpenAI) {
    
}