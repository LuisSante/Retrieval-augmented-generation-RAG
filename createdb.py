from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
load_dotenv()

model_name = "text-embedding-ada-002"
DATA_PATH = "./pdf/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chromadb"

if OPENAI_API_KEY is None:
    print("API Key not found. Please ensure OPENAI_API_KEY is set in the environment.")
else:
    print("API Key successfully loaded.")

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_text(documents: list[Document], chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len, 
        add_start_index=True, 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks 

def save_to_chroma(chunks: list[Document], model_name: str):
    #chroma_dir = f"{CHROMA_PATH}_{model_name}"
    chroma_dir = CHROMA_PATH
    if os.path.exists(chroma_dir):
        shutil.rmtree(chroma_dir)
    
    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY),
        persist_directory=chroma_dir
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_dir}.")
    return chroma_dir


def generate_data_store(model_name: str, chunk_size, chunk_overlap):
    documents = load_documents()
    chunks = split_text(documents,chunk_size, chunk_overlap)
    chroma_dir = save_to_chroma(chunks, model_name)
    return chroma_dir

generate_data_store(model_name, chunk_size=500, chunk_overlap=200)
