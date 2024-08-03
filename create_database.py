from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate

import os


def load_documents():
    filenames = os.listdir("documents/")

    documents = []
    print("Loading text documents")
    for f in filenames:
        if f.split(".")[-1] == "txt":
            path = os.path.join("documents/", f)
            f = open(path, "r", encoding="utf-8")
            text = f.read()
            document = Document(text)
            documents.append(document)
    print("Loaded text documents")

    print("Loading PDFs")
    doc_loader = PyPDFDirectoryLoader("documents/")
    docs = doc_loader.load()
    print("Loaded PDFs")
    documents += docs
    return documents


def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 60, length_function = len, is_separator_regex  = False)
    return text_splitter.split_documents(documents)

def create_chunks(replace_newlines=False):
    document = load_documents()
    chunks = split_docs(document)
    if replace_newlines == True:
        for i in range(len(chunks)):
            chunks[i].page_content = chunks[i].page_content.replace("\n","")
        return chunks
    
    return chunks
    
def save_database(embeddings, chunks, path="standard-rag-foreign-policy/Chroma"):    
    database = Chroma.from_documents(chunks,embeddings,persist_directory=path)
    database.persist()
    print(f"Saved {len(chunks)} chunks to Chroma")


def load_database(embeddings, path="standard-rag-foreign-policy/Chroma"):
    database = Chroma(persist_directory=path,embedding_function=embeddings)
    return database

def query_database(query, database, num_responses = 3, similarity_threshold = 0.5):
    results = database.similarity_search_with_relevance_scores(query,k=num_responses)
    try:
        if results[0][1] < similarity_threshold:
            print("Could not find results")
    except:
        print("Error")
    return results



