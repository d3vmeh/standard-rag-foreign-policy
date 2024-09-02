from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from create_database import *
import matplotlib.pyplot as plt
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage


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




print("Ready to answer questions")

st.title("Standard RAG AI Foreign Policy Assistant")



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

user_query = st.chat_input("Enter a question")

if user_query != None and user_query != "":

    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        context = query_database(user_query, db)
        print(context)
        response = get_response(user_query, context, llm)
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(response))
# while True:
#     query = input("Enter a question: ")
#     results = query_database(query, db, num_responses = 100)
#     response, response_text = get_response(query, results, llm)
#     print(f"\n{response}\n\n\n-----------------------------------------------------------------")
    