from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

CHROMA_PATH = "chromadb"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#PROMPT_TEMPLATE = """
#Você é um assistente jurídico altamente especializado.  
#Responda estritamente com base no seguinte contexto extraído dos documentos:  
#
#### **Contexto Fornecido:**  
#{context}  
#
#---
#
#Agora, responda à seguinte pergunta de forma objetiva, clara e fundamentada:  
#
#**🔹 Pergunta:** {question}  
#
#### **Instruções:**  
#1. **Baseie-se exclusivamente no contexto fornecido.**  
#2. **Se a resposta não estiver no contexto, informe isso claramente.**  
#3. **Se houver múltiplos pontos relevantes, estruture a resposta em tópicos.**  
#4. **Identifique e destaque as leis, artigos ou regulamentos mencionados no contexto.**  
#5. **Use uma linguagem formal e precisa, como um parecer jurídico.**  
#
#---
#
#### 📝 **Resposta:**  
#
#**Análise Jurídica:**  
#[Forneça a resposta fundamentada com base no contexto.]  
#
#**Legislação Aplicável:**  
#[Listar as leis, artigos ou regulamentos citados no contexto que sejam relevantes para a resposta.] 
#"""

#PROMPT_TEMPLATE = """" 
#Contexto: {context}  
#
#Tarefa: Você é um especialista jurídico em direito administrativo e constitucional. Sua missão é analisar a consulta do usuário e responder com base exclusivamente nas informações fornecidas no contexto.  
#
#Passos:  
#1. **Classificação da Consulta:** Determine se a pergunta do usuário se encaixa em uma das seguintes categorias:  
#   - **Fato**
#   - **Pedido**   
#   - **Argumento**
#
#2. **Resposta Estruturada:**  
#   - **Classificação da Query:** [Indicar se é Fato, Pedido ou Argumento]  
#   - **Análise Jurídica:** Explicação da fundamentação legal aplicável.  
#   - **Precedentes e Normas:** Citação de leis, jurisprudências ou princípios constitucionais relevantes.  
#   - **Conclusão:** Resumo da resposta de forma clara e objetiva.  
#
#Se não houver informações suficientes no contexto para responder à pergunta, diga explicitamente:  
#*"Não há informações suficientes no contexto para responder a esta pergunta."*
#"""

def prompt(description):
    PROMPT_TEMPLATE = f'''
        Contexto:  
        O documento fornecido está classificado sob o seguinte tema:  

        "{description}"   

        Classificação:  
        Classifique o documento em uma das seguintes categorias:  

        - **Fato** 
        - **Pedido**  
        - **Argumento**  

        Consulta:  
        "{{context}}"  

        Explicação da Paragrafo:  
        Explique o paragrafo fornecida se relaciona com o tema do documento e por que é relevante para a questão jurídica em discussão.  
    '''
    return PROMPT_TEMPLATE

description = '''
A) POSSIBILIDADE DE REAJUSTE DE VENCIMENTO DAS CARREIRAS DO GRUPO DE ATIVIDADES DE EDUCAÇÃO BÁSICA DO PODER EXECUTIVO, PREVISTO PELO ARTIGO 3º DA LEI 21.710/2015 DO ESTADO DE MINAS GERAIS, COM BASE NAS ATUALIZAÇÕES DO PISO SALARIAL NACIONAL DOS PROFISSIONAIS DA EDUCAÇÃO BÁSICA (LEI FEDERAL 11.738/2008); B) ABRANGÊNCIA DAS ALTERAÇÕES EFETUADAS NO PROJETO DE REAJUSTE SALARIAL, PELA ASSEMBLEIA LEGISLATIVA E C) PERIODICIDADE A SER CONSIDERADA NAS ATUALIZAÇÕES.

Descrição:

Recurso extraordinário em que se discute, à luz dos artigos 1º, 2º, 18, 25, 37, X e XIII, 61, § 1º, II, a e c, e 63, I, da Constituição Federal, a constitucionalidade do artigo 3º da Lei 21.710/2015 do Estado de Minas Gerais, que previu o reajuste de vencimento das carreiras do Grupo de Atividades de Educação Básica do Poder Executivo mediante lei específica, observando-se as atualizações do piso salarial nacional dos profissionais da educação básica (Lei federal 11.738/2008), bem como a abrangência das alterações efetuadas pela Assembleia Legislativa no projeto encaminhado pelo Chefe do Poder Executivo, considerando-se a alegação de aumento de despesa não reconhecido na origem, e a definição de qual seria a periodicidade das atualizações a ser considerada.

Tese:

Assentada a constitucionalidade do piso salarial profissional nacional para os profissionais do magistério público da educação básica e sua forma de atualização, é infraconstitucional, a ela se aplicando os efeitos da ausência de repercussão geral, a controvérsia relativa aos reajustes de vencimento dos servidores do Grupo de Atividades de Educação Básica, com fundamento na Lei 21.710/2015 do Estado de Minas Gerais.
'''

query = '''
Cuida-se de ação ordinária por meio da qual a parte autora
pleiteia que seja o réu, ora recorrente, condenado a lhe pagar os valores de
reajuste do piso nacional da educação, referente aos meses de janeiro,
fevereiro e março de 2016, tomando como base de cálculo o mês em que, de
fato, o reajuste foi concedido (abril de 2016), tudo com a correção monetária
e juros devidos, pois afirma que, não obstante a previsão expressa da Lei
Estadual 21.710/15, no sentido de que o servidor da educação deve ter
reajuste salarial em seu vencimento todo mês de janeiro, o réu atrasou o
cumprimento do disposto na norma supracitada  
'''

PROMPT_TEMPLATE = prompt(description)
print(PROMPT_TEMPLATE)

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
    response_text = model.predict(prompt)
    sources = [doc.metadata.get("source", None) for doc, _ in results]
    #print(sources)
    print("\n\n\n")
    
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response, response_text
    return response_text


model_name = "text-embedding-ada-002"

formatted_response, response_text = query_rag(query, model_name)
print(formatted_response)
