import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { callListOutputParser, callStringOutputParser, callStructuredParser } from './output_parsers';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug
    });

    //console.log(await callStringOutputParser(model));
    //console.log(await callListOutputParser(model));
    console.log(await callStructuredParser(model));
};

start();