"""
Script com funções úteis para facilitar as diferentes abordagens do desafio.
"""
import numpy as np
import pandas as pd



def log_retorno(df_previsto:pd.DataFrame,df_historico:pd.DataFrame)->pd.DataFrame:
    """
    Função para calcular o log-retorno entre dois Dataframes defasados 20 dias.
    Inputs:
        df_previsto: Dataframe com os valores das ações previstas.
        df_historico: Dataframe com os valores históricos das ações.
    Output:
        df_log_retorno: Dataframe com o log-retorno calculado.
    """
    
    df_final = pd.DataFrame(index=df_previsto.index, columns=df_previsto.columns)
    
    
    for stock in df_previsto.columns:
    
        t = zip(df_previsto[stock],df_historico[stock])
        list_aux = []

        for i,j in t:
            list_aux.append(round(np.log(i/j),6))

        df_final[stock] = list_aux
    
    return df_final


def calc_score(df_real:pd.DataFrame,df_previsto:pd.DataFrame)-> float:
    """
    Calcula o score das predições.
    Inputs:
        df_real: Dataframe com o log-retorno dos dados reais.
        df_previsto: Dataframe com o log-retorno dos dados previstos.
    Output:
        score: Ponduação do modelo.
    """

    # Erro
    E = df_real - df_previsto

    # Variância
    S = []
    c = 1/(E.shape[0]-1)

    for col in E.columns:
        S.append(c*sum((E[col] - np.mean(E[col]))**2))

    # Erro quadrático Médio
    EQM = []
    c = 1/(E.shape[0])

    for col in E.columns:
        EQM.append(c*sum(E[col]**2))

    # Score
    score = sum((np.array(EQM) + np.array(S))**(0.5))
    
    return score
