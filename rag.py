from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

CHROMA_PATH = "chromadb"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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


model_name = "text-embedding-ada-002"

query = "Inconstitucionalidade do Reajuste Automático pelos Estados"
formatted_response, response_text = query_rag(query, model_name)
print(response_text)
