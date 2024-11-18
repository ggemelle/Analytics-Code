import mne
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os

# Função principal para processar os dados EEG
def processar_dados_eeg(diretorio, taxa_amostragem, saida):
    """
    Processa os dados brutos de EEG e realiza segmentação por participante, tempo e frequência.

    Args:
        diretorio (str): Caminho para a pasta contendo os arquivos de EEG.
        taxa_amostragem (int): Taxa de amostragem dos dados (em Hz).
        saida (str): Caminho para salvar os resultados processados.
    """
    arquivos = [f for f in os.listdir(diretorio) if f.endswith('.json')]

    for arquivo in arquivos:
        caminho_arquivo = os.path.join(diretorio, arquivo)
        participante_id = os.path.splitext(arquivo)[0]
        
        # Carregar dados JSON
        dados_eeg = carregar_dados_json(caminho_arquivo)
        
        # Aplicar filtros básicos (2-50 Hz)
        dados_filtrados = filtrar_dados(dados_eeg, taxa_amostragem)
        
        # Transformar para escala logarítmica
        dados_log = transformar_para_log(dados_filtrados)
        
        # Separar por banda de frequência
        dados_bandas = segmentar_por_banda(dados_log, taxa_amostragem)
        
        # Detectar picos e transições
        picos_transicoes = detectar_picos(dados_bandas)
        
        # Salvar resultados em CSV
        salvar_resultados(participante_id, dados_bandas, picos_transicoes, saida)

# Função para carregar os dados JSON
def carregar_dados_json(caminho):
    import json
    with open(caminho, 'r') as f:
        dados = json.load(f)
    # Organize os dados para retornar um formato compatível com numpy
    return np.array(dados['dadosEEG'])  # Substitua pelo campo correto

# Função para filtrar dados (2-50 Hz)
def filtrar_dados(dados, taxa_amostragem):
    info = mne.create_info(ch_names=['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'O1', 'O2'], sfreq=taxa_amostragem, ch_types='eeg')
    raw = mne.io.RawArray(dados, info)
    raw.filter(l_freq=2, h_freq=50)
    return raw.get_data()

# Função para transformar dados para escala logarítmica
def transformar_para_log(dados):
    return np.log1p(np.abs(dados))

# Função para segmentar dados por banda de frequência
def segmentar_por_banda(dados, taxa_amostragem):
    bandas = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 50)
    }
    dados_bandas = {}
    for banda, (low, high) in bandas.items():
        dados_bandas[banda] = mne.filter.filter_data(dados, sfreq=taxa_amostragem, l_freq=low, h_freq=high)
    return dados_bandas

# Função para detectar picos e transições
def detectar_picos(dados_bandas):
    picos_transicoes = {}
    for banda, dados in dados_bandas.items():
        picos_transicoes[banda] = {}
        for i, canal in enumerate(dados):
            picos, _ = find_peaks(canal, height=np.mean(canal) + 2 * np.std(canal))
            picos_transicoes[banda][f'canal_{i+1}'] = picos
    return picos_transicoes

# Função para salvar resultados
def salvar_resultados(participante_id, dados_bandas, picos_transicoes, saida):
    for banda, dados in dados_bandas.items():
        df = pd.DataFrame(dados.T, columns=[f'Canal_{i+1}' for i in range(dados.shape[0])])
        df.to_csv(os.path.join(saida, f'{participante_id}_{banda}.csv'), index=False, sep=';')
    
    with open(os.path.join(saida, f'{participante_id}_picos_transicoes.json'), 'w') as f:
        import json
        json.dump(picos_transicoes, f, indent=4)

# Caminhos e parâmetros
diretorio_dados = "caminho/para/dados"
saida_dados = "caminho/para/saida"
taxa_amostragem = 250  # Exemplos: 250 Hz

# Executar processamento
processar_dados_eeg(diretorio_dados, taxa_amostragem, saida_dados)
