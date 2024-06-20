import 'dotenv/config';

import { ChatOpenAI } from '@langchain/openai';

const start = async () => {
    const model = new ChatOpenAI(); // OpenAI Key is on env
    const response = await model.stream('Which OpenAI version am I using right now?');

    for await (const chunk of response) {
        process.stdout.write(String(chunk?.content))
    }
};

start();