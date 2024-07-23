import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';
import { callListOutputParser, callStringOutputParser, callStructuredParser, callZodStructuredParser } from './output_parsers';
import { callChatWithDocumentContext } from './retrieval_chains';
import { callConversationHistoryWithDocumentContext } from './conversational-retrieval-chain';
import { callAgentWithTools } from './agents';

const start = async () => {
    // OpenAI Key is on env
    const model = new ChatOpenAI({
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
        maxTokens: 1000,
        verbose: false // debug
    });

    const tavilyApiKey = process.env.TAVILY_API_KEY || '';

    //console.log(await callStringOutputParser(model));
    //console.log(await callListOutputParser(model));
    //console.log(await callStructuredParser(model));
    //console.log(await callZodStructuredParser(model));
    //console.log(await callChatWithDocumentContext(model));
    //console.log(await callConversationHistoryWithDocumentContext(model));
    console.log(await callAgentWithTools(model, tavilyApiKey));

};

start();