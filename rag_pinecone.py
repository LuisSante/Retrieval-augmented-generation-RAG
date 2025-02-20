from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

PINECONE_API_KEY = ""

load_dotenv() 
# Función para cargar documentos
pc = Pinecone(
    api_key=PINECONE_API_KEY, environment="us-east-1-aws")


if os.getenv("INDEX_NAME") not in pc.list_indexes().names():
    pc.create_index(
        name=os.getenv("INDEX_NAME"),
        dimension=1536,  # Ajustar según la dimensión de tus embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(os.getenv("INDEX_NAME"))

def load_documents():
    DATA_PATH = "./pdf/"
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Dividir texto en chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# Guardar chunks en Pinecone
def save_to_pinecone(chunks: list[Document]):
    embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Inicializar la base de datos de vectores Pinecone
    vector_store = Pinecone.from_documents(
        documents=chunks,
        embedding=embedding_function,
        index_name=os.getenv("INDEX_NAME"),
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"), 
    )
    print(f"Saved {len(chunks)} chunks to Pinecone.")

# Consultar Pinecone usando RAG


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""
def query_rag(query_text: str):
    embedding_function = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = Pinecone(
        # documents=chunks,
        index_name=os.getenv("INDEX_NAME"),
        embedding_function=embedding_function,
        api_key=os.getenv("PINECONE_API_KEY"),
        environment="us-east-1-aws",
    )

    # Buscar documentos relevantes
    results = vector_store.similarity_search(
        query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "No matching results found."

    context_text = "\n\n - -\n\n".join(
        [doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generar respuesta usando el modelo de lenguaje
    model = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _ in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

# Ejecutar pipeline completo


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_pinecone(chunks)


# Ejemplo de consulta
query = "¿Qué es el bus en una interconexión de memoria compartida?"
formatted_response = query_rag(query)
print(formatted_response)
