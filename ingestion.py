from config import *
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load(pdf_path):
    loader=PyPDFLoader(pdf_path)
    docs=loader.load()


    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)

    return docs,parent_splitter,child_splitter
