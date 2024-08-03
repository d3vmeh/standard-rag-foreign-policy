from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from create_database import *


def get_response(query,context,llm):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
    #print(context_text)
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an experienced advisor and international diplomat who is assisting the US government in foreign policy. You use natural language "
         "to answer questions based on structured data, unstructured data, and community summaries. You are thoughtful and thorough in your responses."
        ),
        (
            "user",
            """Answer the question only based on the following context:
            {context}


            Here is the question:
            {question}"""
        ),
        ]
        )
    
    chain = (
         {"context": lambda x: context_text, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
    )
    #response_text = model.invoke(prompt)
    response_text = chain.invoke(query)
    #sources = [doc.metadata.get("source", None) for doc, _score in context]
    formatted_response = f"Response: {response_text}\n"#Sources: {sources}"
    return formatted_response, response_text


embeddings = OpenAIEmbeddings()


"""
Run to Create/Update Chrome DB
"""
#chunks = create_chunks(replace_newlines=True)
#print(len(chunks))
#save_database(embeddings,chunks,path="standard-rag-foreign-policy/Chroma")

db = load_database(embeddings,path="standard-rag-foreign-policy/Chroma")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

while True:
    query = input("Enter a question: ")
    results = query_database(query, db, num_responses = 100)
    response, response_text = get_response(query, results, llm)
    print(f"\n{response}\n\n\n-----------------------------------------------------------------")
