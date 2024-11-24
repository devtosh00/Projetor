# python

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm
import matplotlib.pyplot as plt

# ================================
# ======= FUNÇÕES PRINCIPAIS =====
# ================================

def calcular_elasticidade_pib(pib_percent_change, revenue_percent_change):
    """
    Calcula a elasticidade do PIB com base nos dados históricos.

    Parâmetros:
    - pib_percent_change (array): Crescimento percentual do PIB histórico.
    - revenue_percent_change (array): Crescimento percentual da receita histórica.

    Retorna:
    - float: Elasticidade do PIB (coeficiente beta da regressão linear).
    """
    if len(pib_percent_change) == 0 or len(revenue_percent_change) == 0:
        raise ValueError("Os arrays de PIB e Receita não podem estar vazios.")
    slope, intercept, r_value, p_value, std_err = linregress(pib_percent_change, revenue_percent_change)
    return slope

def calcular_elasticidade_preco_inflacao(inflacao_percent_change, preco_percent_change):
    """
    Calcula a elasticidade do preço do produto em relação à inflação com base nos dados históricos.

    Parâmetros:
    - inflacao_percent_change (array): Inflação percentual histórica.
    - preco_percent_change (array): Crescimento percentual do preço do produto histórico.

    Retorna:
    - float: Elasticidade do preço em relação à inflação (coeficiente beta da regressão linear).
    """
    if len(inflacao_percent_change) == 0 or len(preco_percent_change) == 0:
        raise ValueError("Os arrays de Inflação e Preço não podem estar vazios.")
    slope, intercept, r_value, p_value, std_err = linregress(inflacao_percent_change, preco_percent_change)
    return slope

def projetar_quantidade_vendida(quantidade_vendida_inicial, elasticidade_pib, crescimento_pib_futuro):
    """
    Projeta a quantidade vendida com base no crescimento do PIB e elasticidade.

    Parâmetros:
    - quantidade_vendida_inicial (float): Quantidade vendida no ano inicial.
    - elasticidade_pib (float): Elasticidade do PIB calculada.
    - crescimento_pib_futuro (array): Crescimento do PIB projetado para os anos futuros (%).

    Retorna:
    - array: Quantidade vendida projetada para os anos futuros.
    """
    return quantidade_vendida_inicial * (1 + elasticidade_pib * crescimento_pib_futuro / 100)

def projetar_preco(preco_inicial, inflacao_futura, elasticidade_preco_inflacao):
    """
    Projeta o preço com base na inflação futura e na elasticidade.

    Parâmetros:
    - preco_inicial (float): Preço no ano inicial.
    - inflacao_futura (array): Inflação projetada para os anos futuros (%).
    - elasticidade_preco_inflacao (float): Elasticidade do preço em relação à inflação.

    Retorna:
    - array: Preço projetado para os anos futuros.
    """
    preco_percent_change = inflacao_futura * elasticidade_preco_inflacao
    # Aplicar variação percentual ano a ano
    preco_projetado = [preco_inicial]
    for pct_change in preco_percent_change:
        novo_preco = preco_projetado[-1] * (1 + pct_change / 100)
        preco_projetado.append(novo_preco)
    # Remover o preço inicial para alinhar com os anos futuros
    return np.array(preco_projetado[1:])

def projetar_opex(opex_inicial, inflacao_futura):
    """
    Projeta o OPEX com base na inflação futura.

    Parâmetros:
    - opex_inicial (float): OPEX no ano inicial.
    - inflacao_futura (array): Inflação projetada para os anos futuros (%).

    Retorna:
    - array: OPEX projetado para os anos futuros.
    """
    opex_projetado = [opex_inicial]
    for pct_change in inflacao_futura:
        novo_opex = opex_projetado[-1] * (1 + pct_change / 100)
        opex_projetado.append(novo_opex)
    return np.array(opex_projetado[1:])

def projetar_capex(capex_inicial, inflacao_futura):
    """
    Projeta o CAPEX com base na inflação futura.

    Parâmetros:
    - capex_inicial (float): CAPEX no ano inicial.
    - inflacao_futura (array): Inflação projetada para os anos futuros (%).

    Retorna:
    - array: CAPEX projetado para os anos futuros.
    """
    capex_projetado = [capex_inicial]
    for pct_change in inflacao_futura:
        novo_capex = capex_projetado[-1] * (1 + pct_change / 100)
        capex_projetado.append(novo_capex)
    return np.array(capex_projetado[1:])

def calcular_receita_projetada(quantidade_vendida_projetada, preco_projetado):
    """
    Calcula a receita projetada multiplicando a quantidade vendida projetada pelo preço projetado.

    Parâmetros:
    - quantidade_vendida_projetada (array): Quantidade vendida projetada para os anos futuros.
    - preco_projetado (array): Preço projetado para os anos futuros.

    Retorna:
    - array: Receita projetada para os anos futuros.
    """
    return quantidade_vendida_projetada * preco_projetado

def calcular_fcl(ebitda, opex, capex, variacao_capital_giro):
    """
    Calcula o Fluxo de Caixa Livre.

    Parâmetros:
    - ebitda (array): EBITDA projetado.
    - opex (array): Despesas operacionais projetadas.
    - capex (array): Gastos de capital projetados.
    - variacao_capital_giro (array): Variação do capital de giro projetada.

    Retorna:
    - array: Fluxo de Caixa Livre (FCL) projetado.
    """
    return ebitda - opex - capex - variacao_capital_giro

def calcular_projecoes_receita(historico, projecao):
    """
    Calcula as projeções de receita com base nos dados históricos e premissas futuras.

    Parâmetros:
    - historico (dict): Dados históricos contendo 'pib_percent_change', 'revenue_percent_change',
                        'preco_percent_change', 'inflacao_percent_change', 'quantidade_vendida_inicial', 'preco_inicial'.
    - projecao (dict): Premissas futuras contendo 'anos', 'crescimento_pib_futuro', 'inflacao_futura'.

    Retorna:
    - DataFrame: Projeções de receita e custos para os anos futuros.
    """
    # Extrair dados históricos
    pib_percent_change = historico['pib_percent_change']
    revenue_percent_change = historico['revenue_percent_change']
    preco_percent_change_historico = historico['preco_percent_change']
    inflacao_percent_change_historico = historico['inflacao_percent_change']
    quantidade_vendida_inicial = historico['quantidade_vendida_inicial']
    preco_inicial = historico['preco_inicial']

    # Verificar se os arrays históricos têm o mesmo tamanho
    len_hist = len(pib_percent_change)
    if not (len(historico['revenue_percent_change']) == len_hist and
            len(historico['preco_percent_change']) == len_hist and
            len(historico['inflacao_percent_change']) == len_hist):
        raise ValueError("Todos os arrays históricos devem ter o mesmo comprimento.")

    # Extrair premissas futuras
    anos = projecao['anos']
    crescimento_pib_futuro = projecao['crescimento_pib_futuro']
    inflacao_futura = projecao['inflacao_futura']

    # Calcular a elasticidade do PIB
    try:
        elasticidade_pib = calcular_elasticidade_pib(pib_percent_change, revenue_percent_change)
        print(f"Elasticidade do PIB: {elasticidade_pib:.4f}")
    except ValueError as e:
        print(f"Erro ao calcular elasticidade do PIB: {e}")
        return

    # Calcular a elasticidade do preço em relação à inflação
    try:
        elasticidade_preco_inflacao = calcular_elasticidade_preco_inflacao(inflacao_percent_change_historico, preco_percent_change_historico)
        print(f"Elasticidade do Preço em Relação à Inflação: {elasticidade_preco_inflacao:.4f}")
    except ValueError as e:
        print(f"Erro ao calcular elasticidade do preço: {e}")
        return

    # Projetar a quantidade vendida
    quantidade_vendida_projetada = projetar_quantidade_vendida(
        quantidade_vendida_inicial, 
        elasticidade_pib, 
        crescimento_pib_futuro
    )
    print(f"Quantidade vendida projetada: {quantidade_vendida_projetada}")

    # Projetar o preço com base na inflação e elasticidade
    preco_projetado = projetar_preco(preco_inicial, inflacao_futura, elasticidade_preco_inflacao)
    print(f"Preço projetado: {preco_projetado}")

    # Calcular a receita projetada
    receita_projetada = calcular_receita_projetada(quantidade_vendida_projetada, preco_projetado)
    print(f"Receita projetada: {receita_projetada}")

    # Projetar OPEX e CAPEX ajustados pela inflação
    opex_inicial = 30000  # Valor inicial de OPEX
    capex_inicial = 50000  # Valor inicial de CAPEX
    opex_projetado = projetar_opex(opex_inicial, inflacao_futura)
    capex_projetado = projetar_capex(capex_inicial, inflacao_futura)
    print(f"OPEX projetado: {opex_projetado}")
    print(f"CAPEX projetado: {capex_projetado}")

    # Consolidar os resultados em um DataFrame
    df_projecoes = pd.DataFrame({
        'Ano': anos,
        'Crescimento PIB (%)': crescimento_pib_futuro,
        'Inflação (%)': inflacao_futura,
        'Quantidade Vendida Projetada': quantidade_vendida_projetada,
        'Preço Projetado': preco_projetado,
        'Receita Projetada': receita_projetada,
        'OPEX Projetado': opex_projetado,
        'CAPEX Projetado': capex_projetado
    })

    return df_projecoes

def monte_carlo_dcf(historico, projecao, n_simulations=1000, discount_rate=0.1):
    """
    Realiza a simulação de Monte Carlo para o Valuation DCF.

    Parâmetros:
    - historico (dict): Dados históricos para projeção.
    - projecao (dict): Premissas futuras para projeção.
    - n_simulations (int): Número de simulações a serem realizadas.
    - discount_rate (float): Taxa de desconto anual (%).

    Retorna:
    - dict: Estatísticas descritivas dos VPLs.
    - array: Valores Presente Líquido (VPL) resultantes das simulações.
    """
    # Calcular projeções de receita e custos
    df_projecoes = calcular_projecoes_receita(historico, projecao)
    if df_projecoes is None:
        print("Falha ao calcular as projeções. Verifique os dados históricos.")
        return

    anos = df_projecoes['Ano'].values
    receita = df_projecoes['Receita Projetada'].values
    opex = df_projecoes['OPEX Projetado'].values
    capex = df_projecoes['CAPEX Projetado'].values

    # Definir parâmetros fixos
    margem_ebitda = 0.25  # Margem EBITDA (25%)
    variacao_capital_giro_fixo = 10000  # Variação do Capital de Giro fixa anual

    # Arrays para armazenar os VPLs das simulações
    vpls = []

    for sim in range(n_simulations):
        # Introduzir incertezas nas premissas usando distribuições de probabilidade
        # Gerar uma margem EBITDA para cada ano
        margem_ebitda_sim = norm.rvs(loc=margem_ebitda, scale=0.02, size=len(anos))
        
        # Gerar OPEX e CAPEX simulados para cada ano
        opex_sim = norm.rvs(loc=opex, scale=2000, size=len(anos))  # OPEX com inflação já ajustada
        capex_sim = norm.rvs(loc=capex, scale=3000, size=len(anos))  # CAPEX com inflação já ajustada
        
        # Variação do Capital de Giro para cada ano
        variacao_capital_giro_sim = norm.rvs(loc=variacao_capital_giro_fixo, scale=1500, size=len(anos))
        
        # Calcular EBITDA para cada ano
        ebitda = receita * margem_ebitda_sim
        
        # Calcular Fluxo de Caixa Livre (FCL) para cada ano
        fcl = calcular_fcl(ebitda, opex_sim, capex_sim, variacao_capital_giro_sim)
        
        # Calcular VPL para a simulação atual
        vpl = np.sum(fcl / ((1 + discount_rate) ** np.arange(1, len(fcl)+1)))
        
        vpls.append(vpl)

    # Converter para NumPy array para facilitar a análise
    vpls = np.array(vpls)

    # Calcular estatísticas descritivas
    valuation_median = np.median(vpls)
    valuation_mean = np.mean(vpls)
    valuation_5th = np.percentile(vpls, 5)
    valuation_95th = np.percentile(vpls, 95)

    print(f"Valuation (Mediana): {valuation_median:.2f}")
    print(f"Valuation (Média): {valuation_mean:.2f}")
    print(f"Valuation (5º Percentil): {valuation_5th:.2f}")
    print(f"Valuation (95º Percentil): {valuation_95th:.2f}")

    # Plotar a distribuição dos VPLs
    plt.figure(figsize=(10,6))
    plt.hist(vpls, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(valuation_median, color='red', linestyle='dashed', linewidth=1, label='Mediana')
    plt.axvline(valuation_5th, color='green', linestyle='dashed', linewidth=1, label='5º Percentil')
    plt.axvline(valuation_95th, color='orange', linestyle='dashed', linewidth=1, label='95º Percentil')
    plt.title('Distribuição do Valor Presente Líquido (VPL) - Monte Carlo DCF')
    plt.xlabel('VPL')
    plt.ylabel('Frequência')
    plt.legend()
    plt.show()

    # Consolidar estatísticas
    stats = {
        'Mediana': valuation_median,
        'Média': valuation_mean,
        '5 Percentil': valuation_5th,
        '95 Percentil': valuation_95th
    }

    return stats, vpls

# ================================
# ========= USO DO MODELO =========
# ================================

# Definir os dados históricos
historico_selecionado = {
    'pib_percent_change': np.array([0.5, -3.5, -3.3, 1.3, 1.8, 1.2, -3.3, 4.8, 3.01, 2.9, 3.1]),  # Dados de 2014 a 2024
    'revenue_percent_change': np.array([1.0, -1.5, -1.3, 0.8, 1.2, 0.9, -1.2, 2.3, 1.8, 1.5, 2.0]),  # Variação de receita histórica
    'preco_percent_change': np.array([1.4, -0.5, -0.3, 1.0, 1.6, 1.2, -0.8, 2.5, 2.0, 1.7, 2.2]),  # Variação do preço histórica
    'inflacao_percent_change': np.array([7.09, 8.3, 8.42, 3.56, 4.08, 3.98, 4.84, 10.67, 8.93, 4.63, 4.64]),  # Inflação histórica
    'quantidade_vendida_inicial': 1000,  # Quantidade vendida no ano inicial (2025)
    'preco_inicial': 10  # Preço inicial (2025)
}

# Definir as premissas futuras
projecao_futura = {
    'anos': np.array([2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]),
    # Crescimento do PIB projetado (%)
    'crescimento_pib_futuro': np.array([2.5, 2.7, 2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.0, 3.1, 3.2]),
    # Inflação projetada (%)
    'inflacao_futura': np.array([4.12, 3.7, 3.5, 3.5, 3.4, 3.3, 3.2, 3.1, 2.5, 2.1, 2.2])
}

# Realizar a Simulação de Monte Carlo DCF
stats, vpls = monte_carlo_dcf(
    historico_selecionado, 
    projecao_futura, 
    n_simulations=1000, 
    discount_rate=0.1
)

print(stats)