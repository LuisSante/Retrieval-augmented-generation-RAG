PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

print(PROMPT_TEMPLATE)

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