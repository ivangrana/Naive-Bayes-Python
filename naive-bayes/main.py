#Autor: Ivan Grana
import random,pandas
import matplotlib.pyplot as plt
from bayes import train
elm = 14

def importar_csv(tabela):
  """
    Metodo para importar a tabela de dados utilizando o Pandas
                                                                """
  tabela = pandas.read_csv(tabela,skipinitialspace= True)
  return tabela
    
def contar_elementos_freq(tabela,coluna):
    """Conta os elementos mais frequentes de cada coluna"""
    end = tabela[coluna].value_counts()
    return end
    
def mais_frequentes(tab):
 print(tab,"\n","----------------"*3)
 print("a pele mais frequente é:",contar_elementos_freq(tab,'pele').idxmax())
 print("a cor mais frequente é:",contar_elementos_freq(tab,'cor').idxmax())
 print("o tamanho mais frequente é:",contar_elementos_freq(tab,'tamanho').idxmax())
 print("a carne mais frequente é:",contar_elementos_freq(tab,'carne').idxmax())

def calculo_probabilidade(tab,classe_a,classe_b,coluna,j,num_colunas):
  contador_iguais = 0
  for k in range(0,num_colunas):
    if(tab[coluna][k] == classe_a and tab['y'][k] == classe_b):
        contador_iguais += 1
  return (contador_iguais)/(contar_elementos_freq(tab,coluna)[j])

def suavizacao(tab,classe_a,classe_b,coluna,j,num_colunas):
  contador_iguais = 0
  for k in range(0,num_colunas):
    if(tab[coluna][k] == classe_a and tab['y'][k] == classe_b):
        contador_iguais += 1
  return (contador_iguais+1)/(contar_elementos_freq(tab,coluna)[j] + 1)

def caracterizacao(prob_seguro,prob_perigoso):
  if(prob_perigoso > prob_seguro):
   return 'perigoso'
  else:
   return 'seguro'

def prob_classe(tab,classe):
  total = 0
  for j in range(0,elm):
    if tab['y'][j] != 'nan': 
      total += 1
  cont = 0 
  for k in range(0,elm):
    if tab['y'][k] == classe:
      cont += 1
  return cont/total
  

tab = importar_csv('datase01.csv')

headers = tab.columns.to_list()
headers.pop()

print(tab)

final_seguro = calculo_probabilidade(tab,'dura','seguro','carne',0,14)
final_seguro *= calculo_probabilidade(tab,'verde','seguro','cor',1,14)
final_seguro *= calculo_probabilidade(tab,'liso','seguro','pele',1,14)*calculo_probabilidade(tab,'grande','seguro','tamanho',0,14)
print("prob de 14 de ser seguro:",final_seguro)

final_perigoso = calculo_probabilidade(tab,'dura','perigoso','carne',0,14)*calculo_probabilidade(tab,'verde','perigoso','cor',1,14)*calculo_probabilidade(tab,'liso','perigoso','pele',1,14)*calculo_probabilidade(tab,'grande','perigoso','tamanho',0,14)
print("prob de 14 de ser perigoso:",final_perigoso)

print("O animal da linha 14 é ->",caracterizacao(final_seguro,final_perigoso),"\n",2*"--------------------")

tab.xs(14)["y"] = caracterizacao(final_seguro,final_perigoso)
elm+=1

#Atualizando os dados...
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
final_seguro = suavizacao(train,'macia','seguro','carne',0,14)*suavizacao(train,'liso','seguro','pele',1,14)*suavizacao(train,'grande','seguro','tamanho',0,14)*(1/10)
print("prob de 15 de ser seguro:",final_seguro)

final_perigoso = suavizacao(train,'macia','perigoso','carne',0,14)*suavizacao(train,'liso','perigoso','pele',1,14)*suavizacao(train,'grande','perigoso','tamanho',0,14)*(1/6)
print("prob de 15 de ser perigoso:",final_perigoso)

print("O animal da linha 15 é ->",caracterizacao(final_seguro,final_perigoso),"\n",2*"--------------------")

tab.xs(15)["y"] = caracterizacao(final_seguro,final_perigoso)
elm+=1

#Atualizando os dados...
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
final_seguro = calculo_probabilidade(tab,'macia','seguro','carne',0,16)*calculo_probabilidade(tab,'peludo','seguro','pele',1,16)*calculo_probabilidade(tab,'grande','seguro','tamanho',0,16)*calculo_probabilidade(tab,'marrom','seguro','cor',2,16)
print("prob de 16 de ser seguro:",final_seguro)

final_perigoso = calculo_probabilidade(tab,'macia','perigoso','carne',0,16)*calculo_probabilidade(tab,'peludo','perigoso','pele',1,16)*calculo_probabilidade(tab,'grande','perigoso','tamanho',0,17)*suavizacao(tab,'marrom','perigoso','cor',2,16)
print("prob de 16 de ser perigoso:",final_perigoso)

print("O animal da linha 16 é ->",caracterizacao(final_seguro,final_perigoso),"\n",2*"--------------------")

tab.xs(16)["y"] = caracterizacao(final_seguro,final_perigoso)
elm+=1
#Atualizando os dados...
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
final_seguro = calculo_probabilidade(tab,'macia','seguro','carne',0,17)*calculo_probabilidade(tab,'peludo','seguro','pele',1,17)*calculo_probabilidade(tab,'grande','seguro','tamanho',0,17)*calculo_probabilidade(tab,'vermelho','seguro','cor',0,17)
print("prob de 17 de ser seguro:",final_seguro)

final_perigoso = calculo_probabilidade(tab,'macia','perigoso','carne',0,17)*calculo_probabilidade(tab,'peludo','perigoso','pele',1,17)*calculo_probabilidade(tab,'grande','perigoso','tamanho',0,17)*calculo_probabilidade(tab,'vermelho','perigoso','cor',0,17)
print("prob de 17 de ser perigoso:",final_perigoso)

print("O animal da linha 17 é ->",caracterizacao(final_seguro,final_perigoso),"\n",2*"--------------------")

tab.xs(17)["y"] = caracterizacao(final_seguro,final_perigoso)
print("Tabela após a caracterização:\n","---------"*3)
print(tab,"\n")
