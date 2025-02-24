from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os
import shutil  

DATA_PATH = "./pdf/"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma"

# Modelos de embeddings disponibles
modelos_embeddings_openai = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]

# Rango de chunk_size
chunk_sizes = range(200, 1001, 100)

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_text(documents: list[Document], chunk_size: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.2),
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document], chunk_size: int):
    for model in modelos_embeddings_openai:
        dir_name = f"{CHROMA_PATH}_{chunk_size}_{model}"
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

        db = Chroma.from_documents( 
            chunks,
            OpenAIEmbeddings(model=model, openai_api_key=OPENAI_API_KEY),
            persist_directory=dir_name
        )
        db.persist()
        print(f"Saved {len(chunks)} chunks to {dir_name} using {model}.")


def generate_data_store():
    documents = load_documents()
    for chunk_size in chunk_sizes:
        chunks = split_text(documents, chunk_size)
        save_to_chroma(chunks, chunk_size)

PROMPT_TEMPLATE = """
Você é um assistente jurídico altamente especializado.  
Responda estritamente com base no seguinte contexto extraído dos documentos:  

### 📌 **Contexto Fornecido:**
{context}  

---

Agora, responda à seguinte pergunta de forma objetiva, clara e fundamentada:  

**🔹 Pergunta:** {question}  

### 🔍 **Instruções:**  
1 **Baseie-se exclusivamente no contexto fornecido.**  
2 **Se a resposta não estiver no contexto, informe isso claramente.**  
3 **Se houver múltiplos pontos relevantes, estruture a resposta em tópicos.**  
4 **Use uma linguagem formal e precisa, como um parecer jurídico.**  

---

### 📝 **Resposta:**
"""


def query_rag(query_text):
    best_response = None
    best_score = -1
    best_model = None
    best_chunk_size = None

    for chunk_size in chunk_sizes:
        for model in modelos_embeddings_openai:
            dir_name = f"{CHROMA_PATH}_{chunk_size}_{model}"
            embedding_function = OpenAIEmbeddings(model=model, openai_api_key=OPENAI_API_KEY)
            db = Chroma(persist_directory=dir_name, embedding_function=embedding_function)
            results = db.similarity_search_with_relevance_scores(query_text, k=5)

            if len(results) == 0 or results[0][1] < 0.75:
                print(f"Chunk Size {chunk_size}, Model {model}: No relevant results found.")
                continue

            context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)

            model_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
            response_text = model_llm.invoke(prompt)
            score = results[0][1]

            if score > best_score:
                best_score = score
                best_response = response_text
                best_model = model
                best_chunk_size = chunk_size

    if best_response is None:
        return "No suitable response found."
    
    return f"Best Chunk Size: {best_chunk_size}\nBest Model: {best_model}\n\nResponse: {best_response}\n\nScore: {best_score}"


# Generar el datastore con todos los modelos y chunk_sizes
generate_data_store()

# Ejecutar una consulta y encontrar la mejor configuración
query = "Qual é o contexto da ação ordinária que levou ao recurso extraordinário da União?"
response = query_rag(query)
print(response)