import { ChatOpenAI } from '@langchain/openai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser, CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { StructuredOutputParser } from 'langchain/output_parsers';
import { z } from 'zod';


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

export async function callStructuredParser(model: ChatOpenAI) {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting instructions: {format_instructions}
        Phrase: {phrase}
    `)

    // This parser is more useful for more simple objects. It's important to always inform on the value the description of what the field will be for.
    // For more complex ones such as arrays, is better to use the .fromZodSchema method
    const structuredParser = StructuredOutputParser.fromNamesAndDescriptions({
        name: 'The name of the person',
        age: 'The age of the person'
    });

    const chain = templatePrompt.pipe(model).pipe(structuredParser);

    const response = await chain.invoke({
        phrase: 'Max is 30 years old and is an alcoholic',
        format_instructions: structuredParser.getFormatInstructions(),
    });

    // This method return a serie of instructions that makes the AI model understand clearly that the object returned must be a JSON
    const instructions = structuredParser.getFormatInstructions();

    return {  ...response } ;
}

export async function callZodStructuredParser(model: ChatOpenAI) {
    const templatePrompt = ChatPromptTemplate.fromTemplate(`
        Extract information from the following phrase.
        Formatting instructions: {format_instructions}
        Phrase: {phrase}
    `)

    // It's important to always descrive what the field will be for using the .describe() method
    const structuredParser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe('Name of the recipe'),
            ingredients: z.array(z.string()).describe('Recipe\'s ingredients')
        })
    );

    const chain = templatePrompt.pipe(model).pipe(structuredParser);

    const response = await chain.invoke({
        phrase: `chicken and broccoli pasta 
            350g broccoli, cut into small florets, stems cut into matchsticks
            300g dried curly fettuccine pasta
            1⁄2 barbecued chicken, skin and bones removed, meat coarsely shredded
            2 x 80g Boursin Garlic & Herbs cheese
            125g (3⁄4 cup) sun-dried tomatoes, coarsely chopped`,
        format_instructions: structuredParser.getFormatInstructions(),
    });

    // This method return a serie of instructions that makes the AI model understand clearly that the object returned must be a JSON
    const instructions = structuredParser.getFormatInstructions();

    return {  ...response } ;
}