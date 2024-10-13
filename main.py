from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from create_database import *
import matplotlib.pyplot as plt
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import shelve
from wordcloud import WordCloud
from PIL import Image

def get_response(query,context,llm):
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in context])
    #print(context_text)
    prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            """
        You are an experienced advisor and international diplomat assisting the US government in shaping foreign policy. 
        Your role is to provide insightful and comprehensive answers to inquiries by using context provided to you from 
        textual sources including books, news articles, and speech transcripts.Please respond thoughtfully and thoroughly, 
        ensuring that your answers reflect a deep understanding of global issues and diplomatic nuances.
        """
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
    wordcloud = create_wordcloud(context_text)

    return formatted_response, response_text

def create_wordcloud(text):
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    word_cloud.to_file('wordcloud.png')
    return word_cloud

def load_chat_history():
    with shelve.open("conversation_history") as db:
        return db.get("messages",[])
    
def save_chat_history(messages):
    with shelve.open("conversation_history") as db:
        db["messages"] = messages


embeddings = OpenAIEmbeddings()


"""
Run to Create/Update Chrome DB
"""
#chunks = create_chunks(replace_newlines=True)
#print(len(chunks))
#save_database(embeddings,chunks,path="standard-rag-foreign-policy/Chroma")

db = load_database(embeddings,path="standard-rag-foreign-policy/Chroma")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620",temperature=0.6)




print("Ready to answer questions")

st.title("Standard RAG AI Foreign Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()


with st.sidebar:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        context = query_database(prompt,db)
        #print(context)
        formatted_response, full_response = get_response(prompt,context,llm)
        message_placeholder.markdown(full_response)   
        try:
            img = Image.open('wordcloud.png')
            st.image(img, caption='Wordcloud of the response', use_column_width=True)
        except:
            pass
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})

save_chat_history(st.session_state.messages)








# st.title("Standard RAG AI Foreign Policy Assistant")



# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# chat_placeholder = st.container()
# prompt_placeholder = st.form("chat-form")

# user_query = st.chat_input("Enter a question")

# if user_query != None and user_query != "":

#     st.session_state.chat_history.append(HumanMessage(user_query))

#     with st.chat_message("Human"):
#         st.markdown(user_query)

#     with st.chat_message("AI"):
#         context = query_database(user_query, db)
#         print(context)
#         response = get_response(user_query, context, llm)
#         st.markdown(response)

#     st.session_state.chat_history.append(AIMessage(response))
# while True:
#     query = input("Enter a question: ")
#     results = query_database(query, db, num_responses = 100)
#     response, response_text = get_response(query, results, llm)
#     print(f"\n{response}\n\n\n-----------------------------------------------------------------")
    
