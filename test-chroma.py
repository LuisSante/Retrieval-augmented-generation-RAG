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
CHROMA_PATH = "chromadb"

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

PROMPT_TEMPLATE = """
Você é um assistente jurídico altamente especializado.  
Responda estritamente com base no seguinte contexto extraído dos documentos:  

### **Contexto Fornecido:**  
{context}  

---

Agora, responda à seguinte pergunta de forma objetiva, clara e fundamentada:  

**🔹 Pergunta:** {question}  

### **Instruções:**  
1. **Baseie-se exclusivamente no contexto fornecido.**  
2. **Se a resposta não estiver no contexto, informe isso claramente.**  
3. **Se houver múltiplos pontos relevantes, estruture a resposta em tópicos.**  
4. **Identifique e destaque as leis, artigos ou regulamentos mencionados no contexto.**  
5. **Use uma linguagem formal e precisa, como um parecer jurídico.**  

---

### 📝 **Resposta:**  

**Análise Jurídica:**  
[Forneça a resposta fundamentada com base no contexto.]  

**Legislação Aplicável:**  
[Listar as leis, artigos ou regulamentos citados no contexto que sejam relevantes para a resposta.] 
"""

def query_rag(query_text, model_name: str):
    # chroma_dir = f"{CHROMA_PATH}_{model_name}"
    # chroma_dir = CHROMA_PATH
    embedding_function = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    print("\n\n Results:", results, "\n\n")
    print("\n\n Results:", results[0][1], "\n\n")
    
    if len(results) == 0 or results[0][1] < 0.75:
        return "Unable to find matching results."
    
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _ in results]

    print("\n\n\n")
    
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text

def generate_data_store(model_name: str, chunk_size, chunk_overlap):
    documents = load_documents()
    chunks = split_text(documents,chunk_size, chunk_overlap)
    chroma_dir = save_to_chroma(chunks, model_name)
    return chroma_dir

# Ejecutar
model_name = "text-embedding-ada-002"
generate_data_store(model_name, chunk_size=600, chunk_overlap=200)

query = "Inconstitucionalidade do Reajuste Automático pelos Estados"
formatted_response, response_text = query_rag(query, model_name)
print(response_text)
