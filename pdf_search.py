from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
import os
import asyncio

os.environ["LANGSMITH_TRACING"]="true"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.load():
        pages.append(page)
    return pages

file_path = "docs/python-crash-course.pdf"

# Define the tools for the agent to use

@tool
def search_pdf(query: str, file_path="docs/python-crash-course.pdf") -> list:
    """ hi there"""
    load_pdf(file_path)
    pages = load_pdf(file_path)
    vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
    docs = vector_store.similarity_search(query, k=4)
    p_list = []
    for doc in docs:
        p_list.append(doc.page_content)
    return p_list