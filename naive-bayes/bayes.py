import pandas 
from math import log

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

def probabilidade_classe(tab,classe):
  """Calculo da probabilidade a priori da classe"""
  total = 14
  for j in range(0,14):
    cont = 0 
  for k in range(0,14):
    if tab['y'][k] == classe:
      cont += 1
  return cont/total


train = importar_csv("train.csv")