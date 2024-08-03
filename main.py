from langchain_openai import OpenAIEmbeddings

from langchain_community.llms import Ollama
from langchain_openai.llms import OpenAI

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory

import os
from create_database import *


chunks = create_chunks("ollama-in-context-learning/PDFs/",replace_newlines=True)

print(len(chunks))

embeddings = OpenAIEmbeddings()

#save_database(embeddings,chunks,path="ollama-in-context-learning/Chroma")

db = load_database(embeddings,path="ollama-in-context-learning/Chroma")

model = OpenAI(
    model = "gpt-4o-mini"
)


memory = ConversationBufferMemory()

conversation = ConversationChain(llm = model, verbose = True, memory = memory)

count = 1

while True:
    query = input("Enter a question: ")
    results = query_database(query, db, num_responses = 20)



    prompt = """
    Answer the question only based on the following context:



    Here is the context you can use to help you answer the questions:
    {context}



    ------------


    If you do not know the answer, do not make up an answer, just say you do not know. 
    
    Answer the question based on the above context. Here is the question:
    
    {question}"""

    response, response_text = get_response(query, results, prompt, model)

    print(f"\n{response}\n\n\n-----------------------------------------------------------------")
