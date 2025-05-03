import streamlit as st
import pandas as pd
import numpy as np
import time
from plotly.subplots import make_subplots
from binance.client import Client as ClientSpot
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException
import plotly.graph_objs as go
from datetime import datetime
import pytz
import ta
import plotly.graph_objects as go
import os
import threading
import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 🔒 Defina sua senha aqui
SENHA_CORRETA = "senha123"  # Troque por sua senha

# Definindo as variáveis globais para API Key e Secret
api_key = None
api_secret = None

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        senha = st.text_input("Digite a senha para acessar o app:", type="password")
        if st.button("Entrar"):
            if senha == SENHA_CORRETA:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Senha incorreta. Tente novamente.")
        return False
    else:
        return True

# Se a senha estiver correta, o app é liberado
if check_password():
    # CONFIGURAÇÕES GERAIS
    st.set_page_config(layout="wide")
    st.title("📈 Trading App - Binance Spot & Futuros")

    # VARIÁVEL PARA CONTROLAR A ATIVAÇÃO DO ROBÔ
    robo_ativo = st.sidebar.checkbox("Ativar Robô de Trading", value=False)
    # --- VARIAVEIS GLOBAIS ---
    client_spot = None
    client_futures = None
    info_troca = None
    oco_ordens_ativas = {}
    ordens_abertas = {}
    SIDE_BUY = 'BUY'
    SIDE_SELL = 'SELL'

    # --- VARIAVEIS GLOBAIS ---
    # ... outras variáveis ...
    ordem_executada_recentemente = False
    tempo_inicio_cooldown = None
    tempo_cooldown_rapido = 300 # <---- AGORA O COOLDOWN É DE 300 SEGUNDOS (5 MINUTOS)
    # ...
    # CLIENTES DA BINANCE (com reconexão automática)
    client_spot = None
    client_futures = None

    # Parâmetros de retry
    MAX_RETRIES = 5
    RETRY_DELAY = 20  # segundos

    def criar_cliente_binance(api_key, api_secret, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
        for tentativa in range(max_retries):
            try:
                client_spot = Client(api_key, api_secret)
                # Testa uma chamada simples para verificar conexão
                client_spot.ping()
                return client_spot
            except Exception as e:
                st.warning(f"Tentativa {tentativa+1} falhou: {e}")
                time.sleep(delay)
        st.error("Não foi possível conectar ao cliente SPOT da Binance após várias tentativas.")
        return None

    def criar_cliente_futures(api_key, api_secret, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
        for tentativa in range(max_retries):
            try:
                client_futures = Client(api_key, api_secret)
                # Testa uma chamada simples para verificar conexão
                client_futures.futures_ping()
                return client_futures
            except Exception as e:
                st.warning(f"Tentativa {tentativa+1} falhou: {e}")
                time.sleep(delay)
        st.error("Não foi possível conectar ao cliente FUTURES da Binance após várias tentativas.")
        return None

    # Entrada do usuário
    api_key = st.sidebar.text_input("API Key", type="password")
    api_secret = st.sidebar.text_input("API Secret", type="password")

    if api_key and api_secret:
        client_spot = criar_cliente_binance(api_key, api_secret)
        client_futures = criar_cliente_futures(api_key, api_secret)

# DICIONÁRIO PARA ARMAZENAR AS ORDENS OCO ATIVAS
oco_ordens_ativas = {}

# FUNÇÕES PARA OBTER DADOS, CALCULAR INDICADORES, PLOTAR GRÁFICO, OBTER SALDO, ETC. (RESTAURANDO AS SUAS)
def obter_dados(par, intervalo, limite=100, mercado="Spot"):
    if not api_key or not api_secret:
        st.error("Por favor, insira suas chaves da API.")
        return pd.DataFrame()

    try:
        if mercado == "Spot" and client_spot:
            klines = client_spot.get_klines(symbol=par, interval=intervalo, limit=limite)
        elif mercado == "Futuros" and client_futures:
            klines = client_futures.futures_klines(symbol=par, interval=intervalo, limit=limite)
        else:
            st.warning(f"Cliente não inicializado para o mercado {mercado}.")
            return pd.DataFrame()

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except BinanceAPIException as e:
        st.error(f"Erro ao obter dados da Binance ({mercado}): {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    df = df.copy()

    # Médias Móveis Simples para a nova estratégia
    df['MM3'] = df['close'].rolling(window=3).mean()
    df['MM21'] = df['close'].rolling(window=21).mean()
    df['MM200'] = df['close'].rolling(window=200).mean() # Mantendo para o filtro opcional

    # Médias Móveis Exponenciais para a nova estratégia
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA400'] = df['close'].ewm(span=400, adjust=False).mean() # Mantendo para o filtro opcional

    # Bandas de Bollinger para a nova estratégia (usando 'ta')
    bb_ta = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['BB_Superior_ta'] = bb_ta.bollinger_hband()
    df['BB_Inferior_ta'] = bb_ta.bollinger_lband()

    # Mantenha os indicadores existentes se as flags `usar_...` forem True
    if usar_mm:
        df['MM9'] = df['close'].rolling(window=9).mean() # Pode ser redundante com EMA9, decidir qual usar na lógica
        pass # As outras MMs (MM21, MM200) já foram calculadas
    if usar_rsi:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    if usar_macd:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal_MACD'] = df['MACD'].ewm(span=9, adjust=False).mean()
    if usar_bb:
        df['MB'] = df['close'].rolling(window=20).mean()
        df['BB_Upper'] = df['MB'] + 2 * df['close'].rolling(window=20).std()
        df['BB_Lower'] = df['MB'] - 2 * df['close'].rolling(window=20).std()

    return df

def plotar_grafico(df, usar_mm, usar_rsi, usar_macd, usar_bb, exibir_mm200_grafico, exibir_ema400_grafico, exibir_medias_rapidas):
    rows = 1 + (1 if usar_rsi and 'RSI' in df.columns else 0) + (
        1 if usar_macd and 'MACD' in df.columns and 'Signal_MACD' in df.columns and 'Histograma_MACD' in df.columns else 0)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6] + [0.2] * (rows - 1))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Candlestick'
    ))
    if usar_mm and 'MM9' in df.columns and 'MM21' in df.columns and 'MM200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MM9'], mode='lines', name='MM9'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MM21'], mode='lines', name='MM21'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MM200'], mode='lines', name='MM200'))
    if usar_bb and 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Superior'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Inferior'))

    # Adicionar MM200 ao gráfico SE o checkbox 'exibir_mm200_grafico' estiver marcado (VISUAL)
    if exibir_mm200_grafico and 'MM200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MM200'], line=dict(color='orange'), name='MM200 (Visual)'))

    # Adicionar EMA400 ao gráfico SE o checkbox 'exibir_ema400_grafico' estiver marcado (VISUAL)
    if exibir_ema400_grafico and 'EMA400' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA400'], line=dict(color='brown'), name='EMA400 (Visual)'))

    # Adicionar Médias Rápidas ao gráfico SE o checkbox estiver marcado
    if exibir_medias_rapidas:
        if 'MM3' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MM3'], mode='lines', name='MM3', line=dict(color='purple')))
        if 'MM21' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MM21'], mode='lines', name='MM21 (Visual)', line=dict(color='blue')))
        if 'EMA9' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA9'], mode='lines', name='EMA9', line=dict(color='green')))

    fig.update_layout(
        title=f"Gráfico {symbol} - {interval}",
        xaxis_title='Data',
        yaxis_title='Preço',
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    return fig

def buscar_info_simbolo(client, symbol, mercado='spot'):
    if mercado == 'spot':
        return client.get_symbol_info(symbol)
    elif mercado == 'futuros':
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return s
    return None

def get_futures_symbol_info(client, symbol):
    try:
        exchange_info = client.futures_exchange_info()
        for s in exchange_info['symbols']:
            if s['symbol'] == symbol:
                return s
        return None
    except Exception as e:
        print(f"Erro ao buscar info do símbolo: {e}")
        return None

def obter_saldo():
    if not api_key or not api_secret:
        return "API não conectada."

    try:
        if tipo_mercado == "Spot" and client_spot:
            saldo = client_spot.get_asset_balance(asset="USDT")
            return f"USDT (Spot): {saldo['free'] if saldo else 'Erro ao obter saldo'}"
        elif tipo_mercado == "Futuros" and client_futures:
            conta_futuros = client_futures.futures_account_balance()
            usdt = next((x for x in conta_futuros if x['asset'] == 'USDT'), None)
            return f"USDT (Futuros): {usdt['balance'] if usdt else 'Erro ao obter saldo'}"
        else:
            return "Cliente não inicializado."
    except BinanceAPIException as e:
        return f"Erro ao obter saldo: {e}"

def salvar_ordem_simulada(symbol, side, price, quantity, stop_loss=None, take_profit=None, oco=False):
    log_path = f"ordens_simuladas_{symbol}.csv"
    now = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S")
    nova_ordem = {
        "timestamp": now,
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": quantity,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "oco": oco,
        "modo": "Simulado"
    }
    df_nova = pd.DataFrame([nova_ordem])

    if os.path.exists(log_path):
        df_existente = pd.read_csv(log_path)
        df_total = pd.concat([df_existente, df_nova], ignore_index=True)
    else:
        df_total = df_nova

    df_total.to_csv(log_path, index=False)

def salvar_ordem_real(ordem_entrada, symbol, mercado, stop_loss=None, take_profit=None, oco=False, sl_order_id=None, tp_order_id=None, preco_entrada_real=None, take_profit_real=None):
    log_path = f"ordens_reais_{symbol}_{mercado.lower()}.csv"
    now = datetime.now(pytz.timezone("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S")
    preco_execucao = float(ordem_entrada['fills'][0]['price']) if 'fills' in ordem_entrada and ordem_entrada['fills'] else preco_entrada_real if preco_entrada_real is not None else 0.0
    quantidade_executada = float(ordem_entrada['executedQty']) if 'executedQty' in ordem_entrada else float(ordem_entrada['origQty']) if 'origQty' in ordem_entrada else 0.0

    nova_ordem = {
        "timestamp": now,
        "symbol": symbol,
        "side": ordem_entrada['side'],
        "price": preco_execucao,
        "quantity": quantidade_executada,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "oco": oco,
        "modo": "Real",
        "orderId": ordem_entrada.get('orderId'),
        "stopLossOrderId": sl_order_id,
        "takeProfitOrderId": tp_order_id,
        "preco_entrada_real": preco_entrada_real,
        "take_profit_real": take_profit_real
    }
    df_nova = pd.DataFrame([nova_ordem])

    if os.path.exists(log_path):
        df_existente = pd.read_csv(log_path)
        df_total = pd.concat([df_existente, df_nova], ignore_index=True)
    else:
        df_total = df_nova

    df_total.to_csv(log_path, index=False)

def carregar_ordens(symbol):
    logs = []
    if os.path.exists(f"ordens_simuladas_{symbol}.csv"):
        df_sim = pd.read_csv(f"ordens_simuladas_{symbol}.csv")
        df_sim['modo'] = 'Simulado'
        logs.append(df_sim)
    if os.path.exists(f"ordens_reais_{symbol}_spot.csv"):
        df_real_spot = pd.read_csv(f"ordens_reais_{symbol}_spot.csv")
        df_real_spot['modo'] = 'Real (Spot)'
        logs.append(df_real_spot)
    if os.path.exists(f"ordens_reais_{symbol}_futuros.csv"):
        df_real_futures = pd.read_csv(f"ordens_reais_{symbol}_futuros.csv")
        df_real_futures['modo'] = 'Real (Futuros)'
        logs.append(df_real_futures)
    if logs:
        return pd.concat(logs, ignore_index=True).sort_values(by="timestamp", ascending=False)
    return pd.DataFrame()

def calcular_resultado_simulado(symbol):
    path = f"ordens_simuladas_{symbol}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df.sort_values(by="timestamp")
    df['cost'] = 0.0
    df['profit'] = 0.0
    position = 0
    entry_price = 0
    cumulative_profit = 0
    profits = []

    for index, row in df.iterrows():
        price = float(row['price'])
        quantity = float(row['quantity'])
        side = row['side']
        stop_loss = row.get('stop_loss', 0.03)  # usa 0.03 como valor padrão se a chave não existir
        take_profit = row.get('take_profit', 0.03) # usa 0.03 como valor padrão se a chave não existir

        if side == 'BUY' and position == 0:
            position = quantity
            entry_price = price
            df.loc[index, 'cost'] = price * quantity
        elif side == 'SELL' and position > 0:
            profit = (price - entry_price) * quantity
            df.loc[index, 'profit'] = profit
            cumulative_profit += profit
            position = 0
            entry_price = 0
        elif side == 'SELL' and position == 0: # Para operações short (futuros) - precisa ser melhor definido
            position = -quantity
            entry_price = price
            df.loc[index, 'cost'] = price * quantity
        elif side == 'BUY' and position < 0: # Para fechar operações short
            profit = (entry_price - price) * quantity
            cumulative_profit += profit
            position = 0
            entry_price = 0

        # Simulação de Stop Loss e Take Profit
        if position > 0 and stop_loss is not None and (price <= stop_loss or price >= take_profit):
            profit_sl_tp = (price - entry_price) * quantity
            df.loc[index, 'profit'] = profit_sl_tp
            cumulative_profit += profit_sl_tp
            position = 0
            entry_price = 0
        elif position < 0 and stop_loss is not None and (price >= stop_loss or price <= take_profit):
            profit_sl_tp = (entry_price - price) * quantity
            cumulative_profit += profit_sl_tp
            position = 0
            entry_price = 0

        profits.append(cumulative_profit)

    df['cumulative_profit'] = profits
    return df

# ----------- NOVAS SEÇÕES PARA ESTRATÉGIAS E GERENCIAMENTO DE RISCO -----------

# CONFIGURAÇÕES DE RISCO (MANTENDO AS SUAS)
niveis_risco = {
    "Suave": {"mm_rapida": 9, "mm_lenta": 21, "rsi_limites": (30, 70), "sl_tp_ratio": (0.5, 2)},
    "Moderado": {"mm_rapida": 12, "mm_lenta": 26, "rsi_limites": (35, 65), "sl_tp_ratio": (0.75, 1.5)},
    "Profissional": {"mm_rapida": 20, "mm_lenta": 50, "rsi_limites": (40, 60), "sl_tp_ratio": (1, 2)},
    "Agressivo": {"mm_rapida": 5, "mm_lenta": 15, "rsi_limites": (25, 75), "sl_tp_ratio": (1, 3)},
}

# INPUTS DO USUÁRIO (MANTENDO AS SUAS)
with st.sidebar:
    st.subheader("⚙️ Configurações")
    modo_operacao = st.selectbox("Modo de Operação", ["Simulado", "Real"])
    tipo_mercado = st.radio("Mercado", ["Spot", "Futuros"])

    st.subheader("📊 Gráfico")
    symbol = st.text_input("Símbolo (ex:BTCUSDT)", value="BTCUSDT")
    interval = st.selectbox("Intervalo do Gráfico", ["1m", "5m", "15m", "1h", "4h", "1d"], index=1)
    qtde_dados = st.slider("Quantidade de candles", min_value=50, max_value=1000, value=200)

    st.subheader("🧪 Indicadores")
    usar_mm = st.checkbox("Exibir Média Móvel (MM)")
    usar_rsi = st.checkbox("Exibir RSI")
    usar_macd = st.checkbox("Exibir MACD")
    usar_bb = st.checkbox("Exibir Bandas de Bollinger")
    exibir_mm200_grafico = st.checkbox("Exibir MM200")
    exibir_ema400_grafico = st.checkbox("Exibir EMA400")
    exibir_medias_rapidas = st.checkbox("Exibir Médias Rápidas (MM3, MM21, EMA9)")

    st.subheader("🤖 Estratégia")
    estrategia_ativa = st.selectbox("Selecione a Estratégia:",
                                    ["MM Cruzamento", "RSI (Toque nas Extremidades)", "MACD Cruzamento",
                                     "Bandas de Bollinger (Toque)", "Confluência Manual", "Cruzamento MM + BB Fluxo"])
    usar_confirmacao_rsi_compra = st.checkbox("Usar Confirmação RSI Compra (50-70)", value=False)
    usar_confirmacao_rsi_venda = st.sidebar.checkbox("Usar Confirmação RSI Venda (30-50)", value=False)  # <--- ADICIONE ESTA LINHA
    usar_confirmacao_volume = st.checkbox("Confirmar entrada com volume 20% acima do candle anterior", value=False)
    usar_filtro_mm200_global = st.checkbox("Usar Filtro MM200")
    usar_filtro_ema400_global = st.checkbox("Usar Filtro EMA400")
    st.subheader("➕ Confluência Manual")
    usar_confluencia_manual = st.checkbox("Ativar Confluência Manual", value=False)
    usar_confluencia_mm = st.checkbox("Usar MM na Confluência", value=False)
    usar_confluencia_rsi = st.checkbox("Usar RSI na Confluência", value=False)
    usar_confluencia_macd = st.checkbox("Usar MACD na Confluência", value=False)
    usar_confluencia_bb = st.checkbox("Usar BB na Confluência", value=False)

    st.subheader("⚙️ Configurações Futuros")
    alavancagem = st.slider("Alavancagem", min_value=1, max_value=125, value=1, step=1, key='alavancagem')  # Barra deslizante de 1x a 125x, começando em 1x, com passos de 1

    nivel_risco = st.selectbox("Nível de Risco", ["Suave", "Moderado", "Profissional", "Agressivo"])
    quantidade_trade_pct = st.number_input("Quantidade por Trade (%) da Banca", min_value=0.01, max_value=100.0,
                                           value=1.0, step=0.01)
    definir_sl_tp_manualmente = st.checkbox("Definir Stop Loss/Take Profit Manualmente")
    if definir_sl_tp_manualmente:
        sl_padrao = float(niveis_risco[nivel_risco]["sl_tp_ratio"][0])
        tp_padrao = float(niveis_risco[nivel_risco]["sl_tp_ratio"][1])
        stop_loss_manual = st.number_input("Stop Loss (%)", min_value=0.01, step=0.01, value=sl_padrao)
        take_profit_manual = st.number_input("Take Profit (%)", min_value=0.01, step=0.01, value=tp_padrao)
    else:
        stop_loss_manual = None
        take_profit_manual = None

    st.subheader("⚙️ Parâmetros da Estratégia")
    config_atual = niveis_risco[nivel_risco]
    if usar_mm:
        st.markdown(f"**Média Móvel:**")
        st.markdown(f"- Rápida: {config_atual['mm_rapida']}")
        st.markdown(f"- Lenta: {config_atual['mm_lenta']}")
    if usar_rsi:
        st.markdown(f"**RSI:**")
        st.markdown(f"- Limite Inferior: {config_atual['rsi_limites'][0]}")
        st.markdown(f"- Limite Superior: {config_atual['rsi_limites'][1]}")
    if usar_macd:
        st.markdown(f"**MACD:**")
        st.markdown("- EMA Rápida: 12")  # Valores padrão
        st.markdown("- EMA Lenta: 26")   # Valores padrão
        st.markdown("- Sinal EMA: 9")    # Valores padrão
    if usar_bb:
        st.markdown(f"**Bandas de Bollinger:**")
        st.markdown("- Período: 20")      # Valor padrão
        st.markdown("- Desvio Padrão: 2") # Valor padrão

# FUNÇÕES PARA GERAR SINAIS DE ESTRATÉGIA (MANTENDO AS SUAS)
def gerar_sinal_mm_cruzamento(df, rapida, lenta):
    if not 'MM_Rapida' in df.columns or not 'MM_Lenta' in df.columns or not 'close' in df.columns or len(df) < 3:
        return "Neutro"

    mm_rapida_atual = df['MM_Rapida'].iloc[-1]
    mm_lenta_atual = df['MM_Lenta'].iloc[-1]
    preco_atual = df['close'].iloc[-1]

    mm_rapida_anterior = df['MM_Rapida'].iloc[-2]
    mm_lenta_anterior = df['MM_Lenta'].iloc[-2]

    mm_rapida_penultima = df['MM_Rapida'].iloc[-3]
    mm_lenta_penultima = df['MM_Lenta'].iloc[-3]

    # --- Lógica de Cruzamento Persistente e Relação com o Preço ---

    # Sinal de Compra
    if (mm_rapida_anterior > mm_lenta_anterior and mm_rapida_penultima <= mm_lenta_penultima) and \
       (mm_rapida_atual > mm_lenta_atual) and \
       (preco_atual > mm_rapida_atual and preco_atual > mm_lenta_atual):
        return "Compra"

    # Sinal de Venda
    elif (mm_rapida_anterior < mm_lenta_anterior and mm_rapida_penultima >= mm_lenta_penultima) and \
         (mm_rapida_atual < mm_lenta_atual) and \
         (preco_atual < mm_rapida_atual and preco_atual < mm_lenta_atual):
        return "Venda"

    return "Neutro"

def gerar_sinal_rsi_extremidades(df, limite_inferior, limite_superior):
    if 'RSI' in df.columns:
        if df['RSI'].iloc[-1] < limite_inferior:
            return "Compra"
        elif df['RSI'].iloc[-1] > limite_superior:
            return "Venda"
    return "Neutro"

def verificar_divergencia_rsi(df):
    if not 'high' in df.columns or not 'low' in df.columns or not 'RSI' in df.columns or len(df) < 5:
        return "Neutro"

    # Divergência de Alta (Bullish)
    # Preço faz mínimas mais baixas, RSI faz mínimas mais altas
    if (df['low'].iloc[-2] < df['low'].iloc[-1]) and (df['RSI'].iloc[-2] > df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] < 50:
        return "Compra"

    # Divergência de Baixa (Bearish)
    # Preço faz máximas mais altas, RSI faz máximas mais baixas
    elif (df['high'].iloc[-2] > df['high'].iloc[-1]) and (df['RSI'].iloc[-2] < df['RSI'].iloc[-1]) and df['RSI'].iloc[-1] > 50:
        return "Venda"

    return "Neutro"

def gerar_sinal_macd_cruzamento(df):
    if not 'MACD' in df.columns or not 'Signal_MACD' in df.columns:
        return "Neutro"

    macd_atual = df['MACD'].iloc[-1]
    signal_atual = df['Signal_MACD'].iloc[-1]
    macd_anterior = df['MACD'].iloc[-2]
    signal_anterior = df['Signal_MACD'].iloc[-2]

    sinal = "Neutro"

    # Cruzamento da Linha MACD com a Linha de Sinal
    if macd_atual > signal_atual and macd_anterior <= signal_anterior:
        sinal = "Compra"
    elif macd_atual < signal_atual and macd_anterior >= signal_anterior:
        sinal = "Venda"

    # Considerar cruzamento com a Linha Zero (adicionando força ao sinal)
    if macd_atual > 0 and macd_anterior <= 0 and sinal == "Neutro":
        sinal = "Compra"
    elif macd_atual < 0 and macd_anterior >= 0 and sinal == "Neutro":
        sinal = "Venda"
    elif macd_atual > 0 and macd_anterior <= 0 and sinal == "Compra": # Reforçar compra
        sinal = "Compra"
    elif macd_atual < 0 and macd_anterior >= 0 and sinal == "Venda": # Reforçar venda
        sinal = "Venda"

    return sinal

def verificar_divergencia_macd(df):
    if not 'high' in df.columns or not 'low' in df.columns or not 'MACD' in df.columns or len(df) < 5:
        return "Neutro"

    # Divergência de Alta (Bullish Divergence)
    # Preço faz mínimas mais baixas, MACD faz mínimas mais altas
    if (df['low'].iloc[-2] < df['low'].iloc[-1]) and (df['MACD'].iloc[-2] > df['MACD'].iloc[-1]) and df['MACD'].iloc[-1] < 0:
        return "Compra"

    # Divergência de Baixa (Bearish Divergence)
    # Preço faz máximas mais altas, MACD faz máximas mais baixas
    elif (df['high'].iloc[-2] > df['high'].iloc[-1]) and (df['MACD'].iloc[-2] < df['MACD'].iloc[-1]) and df['MACD'].iloc[-1] > 0:
        return "Venda"

    return "Neutro"
def verificar_momentum_baixa_macd(df):
    if not 'MACD' in df.columns or not 'Histograma_MACD' in df.columns or len(df) < 2:
        return False  # Não há dados suficientes para verificar

    macd_atual = df['MACD'].iloc[-1]
    histograma_atual = df['Histograma_MACD'].iloc[-1]
    histograma_anterior = df['Histograma_MACD'].iloc[-2]

    if macd_atual < 0 and histograma_atual < 0 and histograma_atual < histograma_anterior:
        return True  # Condição de momentum de baixa confirmada
    else:
        return False
def verificar_momentum_alta_macd(df):
    if not 'MACD' in df.columns or not 'Histograma_MACD' in df.columns or len(df) < 2:
        return False  # Não há dados suficientes para verificar

    macd_atual = df['MACD'].iloc[-1]
    histograma_atual = df['Histograma_MACD'].iloc[-1]
    histograma_anterior = df['Histograma_MACD'].iloc[-2]

    if macd_atual > 0 and histograma_atual > 0 and histograma_atual > histograma_anterior:
        return True  # Condição de momentum de alta confirmada
    else:
        return False

def gerar_sinal_bb_toque(df):
    if not 'close' in df.columns or not 'BB_Lower' in df.columns or not 'BB_Upper' in df.columns:
        return "Neutro"

    preco_atual = df['close'].iloc[-1]
    preco_anterior = df['close'].iloc[-2] if len(df) >= 2 else None
    minima_anterior = df['low'].iloc[-2] if len(df) >= 2 else None
    maxima_anterior = df['high'].iloc[-2] if len(df) >= 2 else None
    banda_inferior = df['BB_Lower'].iloc[-1]
    banda_superior = df['BB_Upper'].iloc[-1]

    # --- Lógica de Múltiplos Toques ---
    historico_toques_inferior = sum(1 for close in df['close'].iloc[-4:-1] if close <= df['BB_Lower'].iloc[-4:-1].values[df['close'].iloc[-4:-1] == close].max())
    historico_toques_superior = sum(1 for close in df['close'].iloc[-4:-1] if close >= df['BB_Upper'].iloc[-4:-1].values[df['close'].iloc[-4:-1] == close].max())

    if historico_toques_inferior >= 2 and preco_atual <= banda_inferior:
        return "Neutro"  # Evitar múltiplos toques inferiores recentes
    if historico_toques_superior >= 2 and preco_atual >= banda_superior:
        return "Neutro"  # Evitar múltiplos toques superiores recentes

    sinal = "Neutro"

    # --- Lógica de Confirmação ---
    if preco_anterior is not None and minima_anterior is not None and maxima_anterior is not None:
        if preco_anterior <= banda_inferior and preco_atual > minima_anterior:
            sinal = "Compra"
        elif preco_anterior >= banda_superior and preco_atual < maxima_anterior:
            sinal = "Venda"

    # --- Lógica de Largura da Banda ---
    largura_atual = banda_superior - banda_inferior
    larguras_historicas = (df['BB_Upper'] - df['BB_Lower']).rolling(window=20).mean().iloc[-1]

    if sinal == "Compra" and larguras_historicas is not None and largura_atual < (larguras_historicas * 0.7):
        sinal = "Neutro" # Largura da banda muito estreita para compra
    if sinal == "Venda" and larguras_historicas is not None and largura_atual < (larguras_historicas * 0.7):
        sinal = "Neutro" # Largura da banda muito estreita para venda

    return sinal

def confirmar_sinal_com_volume_candle_anterior(df, sinal):
    if not usar_confirmacao_volume or len(df) < 2:
        return sinal  # Retorna o sinal original se a confirmação de volume não estiver ativada ou não houver candle anterior

    volume_atual = df['volume'].iloc[-1]
    volume_anterior = df['volume'].iloc[-2]

    if sinal == "Compra" and volume_atual < (volume_anterior * 1.20):
        return "Neutro"
    elif sinal == "Venda" and volume_atual < (volume_anterior * 1.20):
        return "Neutro"
    else:
        return sinal
def gerar_sinal_mm_bb_fluxo(df):
    sinal = "Neutro"
    if len(df) < 2:
        return sinal

    mm3_atual = df['MM3'].iloc[-1]
    mm3_anterior = df['MM3'].iloc[-2] if len(df) >= 2 else None
    ema9_atual = df['EMA9'].iloc[-1]
    ema9_anterior = df['EMA9'].iloc[-2] if len(df) >= 2 else None
    mm21_atual = df['MM21'].iloc[-1]
    mm21_anterior = df['MM21'].iloc[-2] if len(df) >= 2 else None
    close_atual = df['close'].iloc[-1]
    close_anterior = df['close'].iloc[-2] if len(df) >= 2 else None
    open_atual = df['open'].iloc[-1]
    open_anterior = df['open'].iloc[-2] if len(df) >= 2 else None
    bb_superior_atual = df['BB_Superior_ta'].iloc[-1]
    bb_superior_anterior = df['BB_Superior_ta'].iloc[-2] if len(df) >= 2 else None
    bb_inferior_atual = df['BB_Inferior_ta'].iloc[-1]
    bb_inferior_anterior = df['BB_Inferior_ta'].iloc[-2] if len(df) >= 2 else None
    largura_bb_atual = bb_superior_atual - bb_inferior_atual if bb_superior_atual is not None and bb_inferior_atual is not None else 0
    largura_bb_anterior = bb_superior_anterior - bb_inferior_anterior if bb_superior_anterior is not None and bb_inferior_anterior is not None else 0

    # Cruzamento sequencial para alta nos dois últimos candles
    cruzamento_alta_sequencial = (open_anterior is not None and mm3_anterior is not None and open_anterior <= mm3_anterior and close_atual > mm3_atual and
                                  open_anterior is not None and mm21_anterior is not None and open_anterior <= mm21_anterior and close_atual > mm21_atual and
                                  open_anterior is not None and ema9_anterior is not None and open_anterior <= ema9_anterior and close_atual > ema9_atual)

    # Cruzamento sequencial para baixa nos dois últimos candles
    cruzamento_baixa_sequencial = (open_anterior is not None and mm3_anterior is not None and open_anterior >= mm3_anterior and close_atual < mm3_atual and
                                   open_anterior is not None and mm21_anterior is not None and open_anterior >= mm21_anterior and close_atual < mm21_atual and
                                   open_anterior is not None and ema9_anterior is not None and open_anterior >= ema9_anterior and close_atual < ema9_anterior)

    banda_superior_alta_atual = bb_superior_atual > bb_superior_anterior if bb_superior_atual is not None and bb_superior_anterior is not None else False
    banda_inferior_baixa_atual = bb_inferior_atual < bb_inferior_anterior if bb_inferior_atual is not None and bb_inferior_anterior is not None else False
    expansao_bb_atual = largura_bb_atual > largura_bb_anterior

    medias_apontando_cima_atual = (ema9_atual > ema9_anterior if ema9_anterior is not None else False) and (mm3_atual > mm3_anterior if mm3_anterior is not None else False)  # Apenas MM3 e EMA9
    medias_apontando_baixo_atual = (ema9_atual < ema9_anterior if ema9_anterior is not None else False) and (mm3_atual < mm3_anterior if mm3_anterior is not None else False)  # Apenas MM3 e EMA9

    preco_acima_medias_atual = (close_atual > mm3_atual if mm3_atual is not None else False) and (close_atual > mm21_atual if mm21_atual is not None else False) and (close_atual > ema9_atual if ema9_atual is not None else False)
    preco_abaixo_medias_atual = (close_atual < mm3_atual if mm3_atual is not None else False) and (close_atual < mm21_atual if mm21_atual is not None else False) and (close_atual < ema9_atual if ema9_atual is not None else False)

    if cruzamento_alta_sequencial and banda_superior_alta_atual and expansao_bb_atual and medias_apontando_cima_atual and preco_acima_medias_atual:
        sinal = "Compra"
    elif cruzamento_baixa_sequencial and banda_inferior_baixa_atual and expansao_bb_atual and medias_apontando_baixo_atual and preco_abaixo_medias_atual:
        sinal = "Venda"

    return sinal

# FUNÇÃO PRINCIPAL PARA GERENCIAR AS ORDENS ABERTAS E AJUSTAR O STOP LOSS
def gerenciar_ordens_abertas(usar_trailing_stop, trailing_stop_lookback, trailing_stop_offset_pct=None ):
    global ordens_abertas
    while True:
        ordens_para_remover = []
        for order_id, detalhes in ordens_abertas.items():
            symbol = detalhes['symbol']
            preco_entrada = detalhes['preco_entrada']
            stop_loss_atual = detalhes['stop_loss']
            take_profit = detalhes['take_profit']
            tipo_mercado = detalhes['tipo_mercado']
            side = detalhes['side']
            sl_break_even = detalhes['sl_break_even']

            # Verificar se a ordem ainda está aberta
            try:
                if tipo_mercado == "Spot" and client_spot:
                    order_status = client_spot.get_order(symbol=symbol, orderId=order_id)
                    if order_status['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        ordens_para_remover.append(order_id)
                        st.info(f"Ordem {order_id} ({symbol}, {tipo_mercado}) fechada com status: {order_status['status']}. Removendo do monitoramento.")
                        continue
                elif tipo_mercado == "Futuros" and client_futures:
                    order_status = client_futures.futures_get_order(symbol=symbol, orderId=order_id)
                    if order_status['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        ordens_para_remover.append(order_id)
                        st.info(f"Ordem {order_id} ({symbol}, Futuros) fechada com status: {order_status['status']}. Removendo do monitoramento.")
                        continue
            except BinanceAPIException as e:
                st.error(f"Erro ao verificar status da ordem {order_id} ({symbol}, {tipo_mercado}): {e}")
                continue

            # Lógica para ajustar o Stop Loss para Break-Even (50% do TP atingido)
            if not sl_break_even and stop_loss_atual is not None and take_profit is not None:
                novo_sl_breakeven = ajustar_stop_loss_50_tp(symbol, preco_entrada, stop_loss_atual, take_profit, tipo_mercado, side)
                if novo_sl_breakeven is not None and novo_sl_breakeven != stop_loss_atual:
                    ordens_abertas[order_id]['stop_loss'] = novo_sl_breakeven
                    ordens_abertas[order_id]['sl_break_even'] = True # Marca como ajustado para break-even

            # Lógica para Trailing Stop Loss
            if usar_trailing_stop and detalhes.get('sl_break_even'): # Só ativa o trailing após o break-even
                novo_sl_trailing = ajustar_stop_loss_trailing_candles(
                    symbol,
                    preco_entrada,
                    ordens_abertas[order_id]['stop_loss'], # Usar o stop_loss atualizado
                    tipo_mercado,
                    side,
                    trailing_stop_lookback,
                    trailing_stop_offset_pct
                )
                if novo_sl_trailing is not None and novo_sl_trailing != ordens_abertas[order_id]['stop_loss']:
                    ordens_abertas[order_id]['stop_loss'] = novo_sl_trailing
                    st.info(f"Trailing Stop Loss da ordem {order_id} ({symbol}, {tipo_mercado}) ajustado para: {novo_sl_trailing:.4f}")

            time.sleep(5) # Verificar a cada 5 segundos

        # Remover ordens fechadas do dicionário
        for order_id_remover in ordens_para_remover:
            if order_id_remover in ordens_abertas:
                del ordens_abertas[order_id_remover]

        time.sleep(1) # Pausa entre as verificações

def ajustar_stop_loss_trailing_candles(symbol, preco_entrada, stop_loss_atual, tipo_mercado, side, lookback_period=20, offset_pct=0.1):
    """
    Ajusta o Stop Loss com base na máxima dos últimos 'lookback_period' candles.

    Args:
        symbol (str): O par de moedas (ex: BTCUSDT).
        preco_entrada (float): O preço de entrada da ordem.
        stop_loss_atual (float): O nível atual do Stop Loss.
        tipo_mercado (str, optional): 'Spot' ou 'Futuros'. Defaults to 'Futuros'.
        side (str, optional): 'BUY' ou 'SELL', para determinar a direção da operação. Defaults to 'BUY'.
        lookback_period (int, optional): Número de candles a serem considerados para a máxima/mínima. Defaults to 20.
        offset_pct (float, optional): Percentual de distância da máxima/mínima para o novo SL. Defaults to 0.1.

    Returns:
        float or None: O novo nível de Stop Loss se ajustado, ou None se não houver ajuste.
    """
    dados = obter_dados(symbol, '1m', lookback_period, tipo_mercado) # Obtém dados suficientes para o lookback
    if dados.empty:
        return stop_loss_atual

    preco_atual = dados['close'].iloc[-1]
    novo_stop_loss = stop_loss_atual

    if side == 'BUY':
        maxima_recente = dados['high'].max()
        novo_sl_candidato = maxima_recente * (1 - offset_pct / 100)
        if novo_sl_candidato > stop_loss_atual and novo_sl_candidato > preco_entrada:
            # Tenta cancelar a ordem SL antiga e criar uma nova
            try:
                if tipo_mercado == "Futuros" and client_futures:
                    orders = client_futures.futures_get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if float(order['stopPrice']) == stop_loss_atual and order['side'] == 'SELL' and order['type'] == 'STOP_MARKET' and order['reduceOnly'] == True), None)
                    if sl_order:
                        client_futures.futures_cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        original_quantity = float(sl_order['origQty'])
                        new_sl_order = client_futures.futures_create_order(symbol=symbol, side='SELL', type='STOP_MARKET', stopPrice=round(novo_sl_candidato, 4), quantity=original_quantity, reduceOnly=True)
                        st.info(f"Trailing SL (Compra/Futuros) ajustado para: {round(novo_sl_candidato, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_sl_candidato, 4)
                elif tipo_mercado == "Spot" and client_spot:
                    orders = client_spot.get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if order['stopPrice'] is not None and float(order['stopPrice']) == stop_loss_atual and order['side'] == 'SELL' and order['type'] in ['STOP_LOSS_LIMIT', 'STOP_LOSS']), None)
                    if sl_order:
                        client_spot.cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        original_quantity = float(sl_order['origQty'])
                        new_sl_order = client_spot.order_limit_stop(symbol=symbol, side='SELL', quantity=original_quantity, stopPrice=round(novo_sl_candidato, 4), price=round(novo_sl_candidato, 4))
                        st.info(f"Trailing SL (Compra/Spot) ajustado para: {round(novo_sl_candidato, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_sl_candidato, 4)
            except BinanceAPIException as e:
                st.error(f"Erro ao ajustar Trailing SL (Compra): {e}")
            return stop_loss_atual # Em caso de falha ao ajustar, mantém o SL atual

    elif side == 'SELL':
        minima_recente = dados['low'].min()
        novo_sl_candidato = minima_recente * (1 + offset_pct / 100)
        if novo_sl_candidato < stop_loss_atual and novo_sl_candidato < preco_entrada:
            # Tenta cancelar a ordem SL antiga e criar uma nova
            try:
                if tipo_mercado == "Futuros" and client_futures:
                    orders = client_futures.futures_get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if float(order['stopPrice']) == stop_loss_atual and order['side'] == 'BUY' and order['type'] == 'STOP_MARKET' and order['reduceOnly'] == True), None)
                    if sl_order:
                        client_futures.futures_cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        original_quantity = float(sl_order['origQty'])
                        new_sl_order = client_futures.futures_create_order(symbol=symbol, side='BUY', type='STOP_MARKET', stopPrice=round(novo_sl_candidato, 4), quantity=original_quantity, reduceOnly=True)
                        st.info(f"Trailing SL (Venda/Futuros) ajustado para: {round(novo_sl_candidato, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_sl_candidato, 4)
                elif tipo_mercado == "Spot" and client_spot:
                    orders = client_spot.get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if order['stopPrice'] is not None and float(order['stopPrice']) == stop_loss_atual and order['side'] == 'BUY' and order['type'] in ['STOP_LOSS_LIMIT', 'STOP_LOSS']), None)
                    if sl_order:
                        client_spot.cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        original_quantity = float(sl_order['origQty'])
                        new_sl_order = client_spot.order_limit_stop(symbol=symbol, side='BUY', quantity=original_quantity, stopPrice=round(novo_sl_candidato, 4), price=round(novo_sl_candidato, 4))
                        st.info(f"Trailing SL (Venda/Spot) ajustado para: {round(novo_sl_candidato, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_sl_candidato, 4)
            except BinanceAPIException as e:
                st.error(f"Erro ao ajustar Trailing SL (Venda): {e}")
            return stop_loss_atual # Em caso de falha ao ajustar, mantém o SL atual

    return novo_stop_loss

# FUNÇÃO PRINCIPAL PARA TOMAR DECISÕES DE TRADING (MANTENDO A SUA)
def tomar_decisao(df, estrategia, nivel_risco):
    config = niveis_risco[nivel_risco]
    sinal_final = "Neutro"
    sinais = []

    if estrategia == "MM Cruzamento" and usar_mm:
        df['MM_Rapida'] = df['close'].rolling(window=config['mm_rapida']).mean()
        df['MM_Lenta'] = df['close'].rolling(window=config['mm_lenta']).mean()
        sinais_mm = gerar_sinal_mm_cruzamento(df, config['mm_rapida'], config['mm_lenta'])

        preco_atual = df['close'].iloc[-1]
        mm200 = df['MM200'].iloc[-1]

        if usar_confluencia_mm:
            if sinais_mm == "Compra" and preco_atual > mm200:
                sinais.append("Compra")
            elif sinais_mm == "Venda" and preco_atual < mm200:
                sinais.append("Venda")
            else:
                sinais.append("Neutro")  # Ou talvez não adicionar nada à lista de sinais
        elif not usar_confluencia_manual:  # Se não estiver em confluência manual, usa o sinal individual com filtro
            if sinais_mm == "Compra" and preco_atual > mm200:
                sinal_final = "Compra"
            elif sinais_mm == "Venda" and preco_atual < mm200:
                sinal_final = "Venda"
            else:
                sinal_final = "Neutro"
    if estrategia == "RSI (Toque nas Extremidades)" and usar_rsi:
        sinal_extremidades = gerar_sinal_rsi_extremidades(df, config['rsi_limites'][0], config['rsi_limites'][1])
        sinal_divergencia = verificar_divergencia_rsi(df)
        sinais_rsi = sinal_divergencia if sinal_divergencia != "Neutro" else sinal_extremidades
        if usar_confluencia_rsi:
            sinais.append(sinais_rsi)
        elif not usar_confluencia_manual:
            sinal_final = sinais_rsi
            if estrategia == "RSI" and usar_rsi:
                rsi_atual = df['RSI'].iloc[-1]
                sinal_rsi = "Neutro"

                if rsi_atual < config['rsi_sobrevenda']:
                    sinal_rsi = "Compra"
                elif rsi_atual > config['rsi_sobrecompra']:
                    sinal_rsi = "Venda"

                sinal_final_rsi = "Neutro"

                if sinal_rsi == "Compra":
                    if usar_confirmacao_rsi_compra:  # Verifica se o checkbox está marcado para compra
                        if 50 < rsi_atual < 70:
                            sinal_final_rsi = "Compra"
                        else:
                            st.info(
                                f"Sinal de compra do RSI ({rsi_atual:.2f}), mas fora da zona de confirmação (50-70).")
                            sinal_final_rsi = "Neutro"  # Se fora da zona, o sinal é neutro
                    else:
                        sinal_final_rsi = "Compra"  # Se o checkbox não estiver marcado, compra no sinal de sobrevenda
                elif sinal_rsi == "Venda":
                    if usar_confirmacao_rsi_venda:  # Verifica se o checkbox está marcado para venda
                        if 30 < rsi_atual < 50:
                            sinal_final_rsi = "Venda"
                        else:
                            st.info(
                                f"Sinal de venda do RSI ({rsi_atual:.2f}), mas fora da zona de confirmação (30-50).")
                            sinal_final_rsi = "Neutro"  # Se fora da zona, o sinal é neutro
                    else:
                        sinal_final_rsi = "Venda"  # Se o checkbox não estiver marcado, vende no sinal de sobrecompra
                else:
                    sinal_final_rsi = "Neutro"

                if usar_confluencia_rsi:
                    if sinal_final_rsi != "Neutro":
                        sinais.append(sinal_final_rsi)
                elif not usar_confluencia_manual:
                    sinal_final = sinal_final_rsi
    if estrategia == "MACD Cruzamento" and usar_macd:
        sinal_cruzamento = gerar_sinal_macd_cruzamento(df)
        sinal_divergencia_macd = verificar_divergencia_macd(df)
        sinais_macd_base = sinal_cruzamento if sinal_cruzamento != "Neutro" else sinal_divergencia_macd
        sinal_final_macd = "Neutro"

        if sinais_macd_base == "Compra":
            if verificar_momentum_alta_macd(df):
                sinal_final_macd = "Compra"
            else:
                st.info("Sinal de compra do MACD, mas sem confirmação de momentum de alta.")
        elif sinais_macd_base == "Venda":
            if verificar_momentum_baixa_macd(df):
                sinal_final_macd = "Venda"
            else:
                st.info("Sinal de venda do MACD, mas sem confirmação de momentum de baixa.")
        else:
            sinal_final_macd = "Neutro"

        if usar_confluencia_macd:
            if sinal_final_macd != "Neutro":
                sinais.append(sinal_final_macd)
        elif not usar_confluencia_manual:
            sinal_final = sinal_final_macd
    if estrategia == "Bandas de Bollinger (Toque)" and usar_bb:
        sinais_bb = gerar_sinal_bb_toque(df)
        if usar_confluencia_bb:
            sinais.append(sinais_bb)
        elif not usar_confluencia_manual:
            sinal_final = sinais_bb

    if estrategia == "Confluência Manual" and sinais:
        compras = sinais.count("Compra")
        vendas = sinais.count("Venda")
        if compras >= 2:
            sinal_final = "Compra"
        elif vendas >= 2:
            sinal_final = "Venda"
        else:
            sinal_final = "Neutro"

            # Adicione a nova estratégia aqui com 'elif'
    elif estrategia == "Cruzamento MM + BB Fluxo":
            sinal_mm_bb_fluxo = gerar_sinal_mm_bb_fluxo(df)
            sinal_final = sinal_mm_bb_fluxo

    # --- Aplicar Filtros Globais ---
    mm200_atual = df['MM200'].iloc[-1] if 'MM200' in df.columns else None
    ema400_atual = df['EMA400'].iloc[-1] if 'EMA400' in df.columns else None
    preco_atual = df['close'].iloc[-1] if not df.empty else None

    if usar_filtro_mm200_global and mm200_atual is not None and preco_atual is not None:
            if sinal_final == "Compra" and not (preco_atual > mm200_atual):
                sinal_final = "Neutro"
            elif sinal_final == "Venda" and not (preco_atual < mm200_atual):
                sinal_final = "Neutro"

    if usar_filtro_ema400_global and ema400_atual is not None and preco_atual is not None:
            if sinal_final == "Compra" and not (preco_atual > ema400_atual):
                sinal_final = "Neutro"
            elif sinal_final == "Venda" and not (preco_atual < ema400_atual):
                sinal_final = "Neutro"

        # --- Aplicar Confirmação de Volume ---
    sinal_final = confirmar_sinal_com_volume_candle_anterior(df, sinal_final)
    return sinal_final

def calcular_sl_tp(preco_entrada, nivel_risco, manual_sl=None, manual_tp=None, tipo_ordem="COMPRA"):
    stop_loss_preco = None
    take_profit_preco = None

    try:
        if manual_sl is not None and manual_tp is not None:
            try:
                manual_sl_pct = float(manual_sl) / 100
                manual_tp_pct = float(manual_tp) / 100

                if tipo_ordem == "COMPRA":
                    stop_loss_preco = preco_entrada * (1 - manual_sl_pct)
                    take_profit_preco = preco_entrada * (1 + manual_tp_pct)
                elif tipo_ordem == "VENDA":
                    stop_loss_preco = preco_entrada * (1 + manual_sl_pct)
                    take_profit_preco = preco_entrada * (1 - manual_tp_pct)

                print(f"SL/TP Manual: SL Preço={stop_loss_preco:.8f} ({manual_sl}%), TP Preço={take_profit_preco:.8f} ({manual_tp}%)")
                st.info(f"SL/TP Manual: SL={stop_loss_preco:.4f} ({manual_sl}%), TP={take_profit_preco:.4f} ({manual_tp}%)")

            except ValueError:
                st.error("Erro ao converter valores manuais de SL/TP para número.")
                return None, None
        else:
            ratio_sl, ratio_tp = niveis_risco[nivel_risco]['sl_tp_ratio']
            stop_loss_pct = 0.01 * ratio_sl
            take_profit_pct = 0.01 * ratio_tp

            if tipo_ordem == "COMPRA":
                stop_loss_preco = preco_entrada * (1 - stop_loss_pct)
                take_profit_preco = preco_entrada * (1 + take_profit_pct)
            elif tipo_ordem == "VENDA":
                stop_loss_preco = preco_entrada * (1 + stop_loss_pct)
                take_profit_preco = preco_entrada * (1 - take_profit_pct)

            print(f"SL/TP Automático: SL Preço={stop_loss_preco:.8f} ({stop_loss_pct*100:.2f}%), TP Preço={take_profit_preco:.8f} ({take_profit_pct*100:.2f}%)")
            st.info(f"SL/TP Automático: SL={stop_loss_preco:.4f} ({stop_loss_pct*100:.2f}%), TP={take_profit_preco:.4f} ({take_profit_pct*100:.2f}%)")

    except KeyError as e:
        print(f"Erro: Nível de risco '{nivel_risco}' não encontrado em niveis_risco. Usando SL/TP manual, se fornecido.")
        if manual_sl is not None and manual_tp is not None:
            try:
                manual_sl_pct = float(manual_sl) / 100
                manual_tp_pct = float(manual_tp) / 100

                if tipo_ordem == "COMPRA":
                    stop_loss_preco = preco_entrada * (1 - manual_sl_pct)
                    take_profit_preco = preco_entrada * (1 + manual_tp_pct)
                elif tipo_ordem == "VENDA":
                    stop_loss_preco = preco_entrada * (1 + manual_sl_pct)
                    take_profit_preco = preco_entrada * (1 - manual_tp_pct)

                print(f"SL/TP Manual (após KeyError): SL Preço={stop_loss_preco:.8f} ({manual_sl}%), TP Preço={take_profit_preco:.8f} ({manual_tp}%)")
                st.info(f"SL/TP Manual (após KeyError): SL={stop_loss_preco:.4f} ({manual_sl}%), TP={take_profit_preco:.4f} ({manual_tp}%)")
            except ValueError:
                print("Erro ao converter valores manuais de SL/TP para número após KeyError.")
                return None, None
        else:
            print(f"Erro: Nível de risco '{nivel_risco}' não encontrado e SL/TP manual não fornecido.")
            return None, None
    except Exception as e:
        print(f"Erro inesperado ao calcular SL/TP: {e}")
        return None, None

    return stop_loss_preco, take_profit_preco

# FUNÇÃO PARA AJUSTAR A QUANTIDADE PARA OS PADRÕES DA BINANCE
def ajustar_quantidade(client, symbol, quantity, mercado="Spot"):
    info_symbol = buscar_info_simbolo(client, symbol, mercado)
    if info_symbol and 'filters' in info_symbol:
        lot_filter = next((f for f in info_symbol['filters'] if f['filterType'] == 'LOT_SIZE'), None)
        if lot_filter:
            step_size = float(lot_filter['stepSize'])
            precision = int(round(-math.log10(step_size), 0))
            return round(quantity, precision)
    return quantity

# FUNÇÃO PARA EXECUTAR A ORDEM (SIMULADA OU REAL) COM SL/TP E OCO (MODIFICADA PARA ARMAZENAR INFO OCO)
# FUNÇÃO PARA EXECUTAR A ORDEM (SIMULADA OU REAL) COM SL/TP E OCO
def executar_ordem(symbol, side, quantity, stop_loss_pct=None, take_profit_pct=None, oco=False):

    """
    Executa uma ordem de compra ou venda no mercado Spot ou Futuros da Binance,
    com opções para Stop Loss e Take Profit percentuais e OCO.

    Args:
        symbol (str): O par de moedas (ex: BTCUSDT).side (str): 'BUY' ou 'SELL'.
        quantity (float): A quantidade a ser negociada.
        stop_loss_pct (float, optional): Percentual do preço de entrada para Stop Loss (ex: 3.0 para 3%). Defaults to None.
        take_profit_pct (float, optional): Percentual do preço de entrada para Take Profit (ex: 5.0 para 5%). Defaults to None.
        oco (bool, optional): Se True, tenta criar uma ordem OCO (se suportado pela exchange). Defaults to False.
    """
    """
        Executa uma ordem de compra ou venda no mercado Spot ou Futuros da Binance,
        com opções para Stop Loss e Take Profit percentuais e OCO.
        Adiciona uma verificação rápida para evitar ordens repetidas.
        """
    global ordens_abertas, pode_abrir_nova_ordem, ultimo_posicionamento_tempo, tempo_limite_nova_ordem, ordem_executada_recentemente, tempo_inicio_cooldown, tempo_cooldown_rapido

    # Verifica se uma ordem foi executada recentemente e ainda estamos no cooldown rápido
    if ordem_executada_recentemente and tempo_inicio_cooldown:
        tempo_decorrido_cooldown = time.time() - tempo_inicio_cooldown
        if tempo_decorrido_cooldown < tempo_cooldown_rapido:
            st.info(
                f"Cooldown rápido ativo. Ignorando novo sinal para {symbol} por mais {tempo_cooldown_rapido - tempo_decorrido_cooldown:.1f} segundos.")
            return None
        else:
            ordem_executada_recentemente = False  # Desativa o cooldown rápido
    dados = obter_dados(symbol, '1m', 1, tipo_mercado)
    preco_atual = dados['close'].iloc[-1] if not dados.empty else 0

    if preco_atual <= 0:
        st.warning(f"Não foi possível obter o preço atual para {symbol}.")
        return None

    adjusted_quantity = quantity
    if tipo_mercado == "Spot" and client_spot:
        adjusted_quantity = ajustar_quantidade(client_spot, symbol, quantity, "Spot")
    elif tipo_mercado == "Futuros" and client_futures:
        adjusted_quantity = ajustar_quantidade(client_futures, symbol, quantity, "Futuros")

    st.info(f"Tentando executar ordem {side} para {symbol} com quantidade: {adjusted_quantity:.8f}, SL%: {stop_loss_pct}, TP%: {take_profit_pct}, OCO: {oco}")

    if modo_operacao == "Simulado":
        sl_preco_simulado = preco_atual * (1 - stop_loss_pct / 100) if side == 'BUY' and stop_loss_pct else (preco_atual * (1 + stop_loss_pct / 100) if side == 'SELL' and stop_loss_pct else None)
        tp_preco_simulado = preco_atual * (1 + take_profit_pct / 100) if side == 'BUY' and take_profit_pct else (preco_atual * (1 - take_profit_pct / 100) if side == 'SELL' and take_profit_pct else None)

        salvar_ordem_simulada(symbol, side, preco_atual, adjusted_quantity, sl_preco_simulado, tp_preco_simulado, oco)
        mensagem_simulada = f"💡 Ordem {side} simulada com sucesso a {preco_atual:.2f}, Qtd: {adjusted_quantity:.8f}"
        if stop_loss_pct:
            mensagem_simulada += f", SL%: {stop_loss_pct:.2f}% ({sl_preco_simulado:.4f})"
        if take_profit_pct:
            mensagem_simulada += f", TP%: {take_profit_pct:.2f}% ({tp_preco_simulado:.4f})"
        if oco and stop_loss_pct and take_profit_pct:
            mensagem_simulada += " (OCO)"
        st.success(mensagem_simulada)
        return {"status": "SIMULADO"}

    else:  # Operação Real
        try:
            order_entrada = None
            sl_preco_real = None
            tp_preco_real = None
            sl_order_id = None
            tp_order_id = None
            preco_entrada_real = None

            if tipo_mercado == "Spot" and client_spot:
                order_entrada = client_spot.order_market(symbol=symbol, side=side, quantity=adjusted_quantity)
                st.info(f"Resposta da API (Entrada Spot): {order_entrada}")
                if order_entrada and order_entrada.get('fills'):
                    preco_entrada_real = float(order_entrada['fills'][0]['price'])
                elif order_entrada:
                    preco_entrada_real = preco_atual

                    # AS LINHAS ABAIXO DEVEM VIR DEPOIS DA DECLARAÇÃO GLOBAL NO TOPO DA FUNÇÃO
                    if order_entrada and order_entrada.get('status') == 'FILLED':
                        ordem_executada_recentemente = True
                        tempo_inicio_cooldown = time.time()

                if order_entrada and stop_loss_pct and take_profit_pct and preco_entrada_real is not None:
                    side_oposto = SIDE_SELL if side == SIDE_BUY else SIDE_BUY

                    sl_preco_real = preco_entrada_real * (1 - stop_loss_pct / 100) if side == 'BUY' else (preco_entrada_real * (1 + stop_loss_pct / 100) if side == 'SELL' else None)
                    tp_preco_real = preco_entrada_real * (1 + take_profit_pct / 100) if side == 'BUY' else (preco_entrada_real * (1 - take_profit_pct / 100) if side == 'SELL' else None)

                    # Ordem de Stop Loss
                    if sl_preco_real is not None:
                        try:
                            sl_order = client_spot.order_limit_stop(symbol=symbol, side=side_oposto, quantity=adjusted_quantity, stopPrice=sl_preco_real, price=sl_preco_real)
                            sl_order_id = sl_order.get('orderId')
                            st.info(f"Resposta da API (SL Spot): {sl_order}")
                        except BinanceAPIException as e:
                            st.error(f"Erro ao criar SL (Spot): {e}")
                    # Ordem de Take Profit
                    if tp_preco_real is not None:
                        try:
                            tp_order = client_spot.order_limit(symbol=symbol, side=side_oposto, quantity=adjusted_quantity, price=tp_preco_real)
                            tp_order_id = tp_order.get('orderId')
                            st.info(f"Resposta da API (TP Spot): {tp_order}")
                        except BinanceAPIException as e:
                            st.error(f"Erro ao criar TP (Spot): {e}")

                    if oco and sl_order_id and tp_order_id:
                        oco_ordens_ativas[f"{symbol}_spot_{order_entrada.get('orderId')}"] = {'sl_id': sl_order_id, 'tp_id': tp_order_id, 'mercado': 'Spot', 'preco_entrada': preco_entrada_real, 'take_profit': tp_preco_real}
                        st.success(f"🚀 Ordem {side} (Spot) REAL executada com OCO! Entrada ID: {order_entrada.get('orderId')}, SL%: {stop_loss_pct:.2f}% ({sl_preco_real:.4f}), TP%: {take_profit_pct:.2f}% ({tp_preco_real:.4f})")
                    elif sl_order_id and tp_order_id:
                        st.success(f"🚀 Ordem {side} (Spot) REAL executada com SL%: {stop_loss_pct:.2f}% ({sl_preco_real:.4f}), TP%: {take_profit_pct:.2f}% ({tp_preco_real:.4f})")
                    else:
                        st.success(f"🚀 Ordem {side} (Spot) REAL executada! Order ID: {order_entrada.get('orderId')}, Qtd: {adjusted_quantity:.8f} (Falha ao criar SL/TP)")
                else:
                    st.success(f"🚀 Ordem {side} (Spot) REAL executada! Order ID: {order_entrada.get('orderId')}, Qtd: {adjusted_quantity:.8f}")

                salvar_ordem_real(order_entrada, symbol, "Spot", sl_preco_real, tp_preco_real, oco, sl_order_id, tp_order_id, preco_entrada_real=preco_entrada_real, take_profit_real=tp_preco_real)
                ordens_abertas[order_entrada.get('orderId')] = {
                    'symbol': symbol,
                    'preco_entrada': preco_entrada_real,
                    'stop_loss': sl_preco_real,
                    'take_profit': tp_preco_real,
                    'tipo_mercado': tipo_mercado,
                    'side': side,
                    'sl_break_even': False
                }
                return {"status": "EXECUTADO_SPOT", "order_id": order_entrada.get('orderId'), "sl_id": sl_order_id, "tp_id": tp_order_id, "preco_entrada": preco_entrada_real, "take_profit": tp_preco_real}

            elif tipo_mercado == "Futuros" and client_futures:
                order_entrada = client_futures.futures_create_order(symbol=symbol, side=side, type='MARKET', quantity=adjusted_quantity)
                st.info(f"Resposta da API (Entrada Futuros): {order_entrada}")
                preco_entrada_real = float(order_entrada['avgFillPrice']) if 'avgFillPrice' in order_entrada else preco_atual

                # AS LINHAS ABAIXO DEVEM VIR DEPOIS DA DECLARAÇÃO GLOBAL NO TOPO DA FUNÇÃO
                if order_entrada and order_entrada.get('updateType') == 'ORDER_TRADE_UPDATE' and order_entrada.get(
                        'orderStatus') == 'FILLED':
                    ordem_executada_recentemente = True
                    tempo_inicio_cooldown = time.time()

                if order_entrada and stop_loss_pct and take_profit_pct and preco_entrada_real is not None:
                    side_oposto = SIDE_SELL if side == SIDE_BUY else SIDE_BUY

                    sl_preco_real = preco_entrada_real * (1 - stop_loss_pct / 100) if side == 'BUY' else (preco_entrada_real * (1 + stop_loss_pct / 100) if side == 'SELL' else None)
                    tp_preco_real = preco_entrada_real * (1 + take_profit_pct / 100) if side == 'BUY' else (preco_entrada_real * (1 - take_profit_pct / 100) if side == 'SELL' else None)

                    exchange_info = client_futures.futures_exchange_info()
                    symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
                    price_precision = int(symbol_info['pricePrecision']) if symbol_info else 4
                    sl_rounded = round(sl_preco_real, price_precision) if sl_preco_real is not None else None
                    tp_rounded = round(tp_preco_real, price_precision) if tp_preco_real is not None else None

                    # Ordem de Stop Loss
                    if sl_rounded is not None:
                        try:
                            sl_order = client_futures.futures_create_order(symbol=symbol, side=side_oposto, type='STOP_MARKET', stopPrice=sl_rounded, quantity=adjusted_quantity, reduceOnly=True)
                            sl_order_id = sl_order.get('orderId')
                            st.info(f"Resposta da API (SL Futuros): {sl_order}")
                        except BinanceAPIException as e:
                            st.error(f"Erro ao criar SL (Futuros): {e}")

                    # Ordem de Take Profit
                    if tp_rounded is not None:
                        try:
                            tp_order = client_futures.futures_create_order(symbol=symbol, side=side_oposto, type='LIMIT', price=tp_rounded, quantity=adjusted_quantity, reduceOnly=True, timeInForce='GTC')
                            tp_order_id = tp_order.get('orderId')
                            st.info(f"Resposta da API (TP Futuros): {tp_order}")
                        except BinanceAPIException as e:
                            st.error(f"Erro ao criar TP (Futuros): {e}")

                    if oco and sl_order_id and tp_order_id:
                        oco_ordens_ativas[f"{symbol}_futures_{order_entrada.get('orderId')}"] = {'sl_id': sl_order_id, 'tp_id': tp_order_id, 'mercado': 'Futuros', 'preco_entrada': preco_entrada_real, 'take_profit': tp_preco_real}
                        st.success(f"🚀 Ordem {side} (Futuros) REAL executada com OCO! Entrada ID: {order_entrada.get('orderId')}, SL%: {stop_loss_pct:.2f}% ({sl_rounded:.4f}), TP%: {take_profit_pct:.2f}% ({tp_rounded:.4f})")
                    elif sl_order_id and tp_order_id:
                        st.success(f"🚀 Ordem {side} (Futuros) REAL executada com SL%: {stop_loss_pct:.2f}% ({sl_rounded:.4f}), TP%: {take_profit_pct:.2f}% ({tp_rounded:.4f})")
                    else:
                        st.success(f"🚀 Ordem {side} (Futuros) REAL executada! Order ID: {order_entrada.get('orderId')}, Qtd: {adjusted_quantity:.8f} (Falha ao criar SL/TP)")
                else:
                    st.success(f"🚀 Ordem {side} (Futuros) REAL executada! Order ID: {order_entrada.get('orderId')}, Qtd: {adjusted_quantity:.8f}")

                salvar_ordem_real(order_entrada, symbol, "Futuros", sl_preco_real, tp_preco_real, oco, sl_order_id, tp_order_id, preco_entrada_real=preco_entrada_real, take_profit_real=tp_preco_real)
                ordens_abertas[order_entrada.get('orderId')] = {
                    'symbol': symbol,
                    'preco_entrada': preco_entrada_real,
                    'stop_loss': sl_preco_real,
                    'take_profit': tp_preco_real,
                    'tipo_mercado': tipo_mercado,
                    'side': side,
                    'sl_break_even': False
                }
                return {"status": "EXECUTADO_FUTUROS", "order_id": order_entrada.get('orderId'), "sl_id": sl_order_id, "tp_id": tp_order_id, "preco_entrada": preco_entrada_real, "take_profit": tp_preco_real}

            st.error("Cliente da API não inicializado para o mercado selecionado.")
            return {"status": "ERRO", "mensagem": "Cliente da API não inicializado."}

        except BinanceAPIException as e:
            st.error(f"Erro ao executar ordem ({tipo_mercado}): {e}")
            return {"status": "ERRO", "mensagem": str(e)}

def ajustar_stop_loss_50_tp(symbol, preco_entrada, stop_loss_atual, take_profit, tipo_mercado='Futuros', side='BUY'):
    """
    Ajusta o Stop Loss para o preço de entrada (ou ligeiramente acima/abaixo) quando o preço atinge 50% do Take Profit.

    Args:
        symbol (str): O par de moedas (ex: BTCUSDT).
        preco_entrada (float): O preço de entrada da ordem.
        stop_loss_atual (float): O nível atual do Stop Loss.
        take_profit (float): O preço do Take Profit definido.
        tipo_mercado (str, optional): 'Spot' ou 'Futuros'. Defaults to 'Futuros'.
        side (str, optional): 'BUY' ou 'SELL', para determinar a direção da operação. Defaults to 'BUY'.

    Returns:
        float or None: O novo nível de Stop Loss se ajustado, ou None se não houver ajuste.
    """
    dados = obter_dados(symbol, '1m', 1, tipo_mercado)
    if dados.empty:
        return stop_loss_atual

    preco_atual = dados['close'].iloc[-1]
    novo_stop_loss = stop_loss_atual

    if side == 'BUY' and preco_entrada < take_profit:
        distancia_tp = take_profit - preco_entrada
        ponto_50_tp = preco_entrada + 0.5 * distancia_tp
        if preco_atual >= ponto_50_tp and stop_loss_atual < preco_entrada:
            # Mover o Stop Loss para o preço de entrada (break-even) ou ligeiramente acima
            novo_stop_loss = preco_entrada  # Você pode adicionar uma pequena margem aqui (ex: + 0.0001)
            try:
                if tipo_mercado == "Futuros" and client_futures:
                    orders = client_futures.futures_get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if float(order['stopPrice']) == stop_loss_atual and order['side'] == 'SELL' and order['type'] == 'STOP_MARKET' and order['reduceOnly'] == True), None)
                    if sl_order:
                        client_futures.futures_cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        # Recalcular a quantidade exata da ordem original
                        original_quantity = float(sl_order['origQty'])
                        # Criar nova ordem SL no break-even
                        new_sl_order = client_futures.futures_create_order(symbol=symbol, side='SELL', type='STOP_MARKET', stopPrice=round(novo_stop_loss, 4), quantity=original_quantity, reduceOnly=True)
                        st.info(f"SL ajustado (Compra/50% TP/Futuros) para: {round(novo_stop_loss, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_stop_loss, 4)
                elif tipo_mercado == "Spot" and client_spot:
                    orders = client_spot.get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if order['stopPrice'] is not None and float(order['stopPrice']) == stop_loss_atual and order['side'] == 'SELL' and order['type'] in ['STOP_LOSS_LIMIT', 'STOP_LOSS']), None)
                    if sl_order:
                        client_spot.cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        # Recalcular a quantidade exata da ordem original
                        original_quantity = float(sl_order['origQty'])
                        # Criar nova ordem SL no break-even
                        new_sl_order = client_spot.order_limit_stop(symbol=symbol, side='SELL', quantity=original_quantity, stopPrice=round(novo_stop_loss, 4), price=round(novo_stop_loss, 4))
                        st.info(f"SL ajustado (Compra/50% TP/Spot) para: {round(novo_stop_loss, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_stop_loss, 4)
            except BinanceAPIException as e:
                st.error(f"Erro ao ajustar SL (Compra/50% TP): {e}")

    elif side == 'SELL' and preco_entrada > take_profit:
        distancia_tp = preco_entrada - take_profit
        ponto_50_tp = preco_entrada - 0.5 * distancia_tp
        if preco_atual <= ponto_50_tp and stop_loss_atual > preco_entrada:
            # Mover o Stop Loss para o preço de entrada (break-even) ou ligeiramente abaixo
            novo_stop_loss = preco_entrada  # Você pode adicionar uma pequena margem aqui (ex: - 0.0001)
            try:
                if tipo_mercado == "Futuros" and client_futures:
                    orders = client_futures.futures_get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if float(order['stopPrice']) == stop_loss_atual and order['side'] == 'BUY' and order['type'] == 'STOP_MARKET' and order['reduceOnly'] == True), None)
                    if sl_order:
                        client_futures.futures_cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        # Recalcular a quantidade exata da ordem original
                        original_quantity = float(sl_order['origQty'])
                        # Criar nova ordem SL no break-even
                        new_sl_order = client_futures.futures_create_order(symbol=symbol, side='BUY', type='STOP_MARKET', stopPrice=round(novo_stop_loss, 4), quantity=original_quantity, reduceOnly=True)
                        st.info(f"SL ajustado (Venda/50% TP/Futuros) para: {round(novo_stop_loss, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_stop_loss, 4)
                elif tipo_mercado == "Spot" and client_spot:
                    orders = client_spot.get_open_orders(symbol=symbol)
                    sl_order = next((order for order in orders if order['stopPrice'] is not None and float(order['stopPrice']) == stop_loss_atual and order['side'] == 'BUY' and order['type'] in ['STOP_LOSS_LIMIT', 'STOP_LOSS']), None)
                    if sl_order:
                        client_spot.cancel_order(symbol=symbol, orderId=sl_order['orderId'])
                        # Recalcular a quantidade exata da ordem original
                        original_quantity = float(sl_order['origQty'])
                        # Criar nova ordem SL no break-even
                        new_sl_order = client_spot.order_limit_stop(symbol=symbol, side='BUY', quantity=original_quantity, stopPrice=round(novo_stop_loss, 4), price=round(novo_stop_loss, 4))
                        st.info(f"SL ajustado (Venda/50% TP/Spot) para: {round(novo_stop_loss, 4)}. Nova ordem SL: {new_sl_order}")
                        return round(novo_stop_loss, 4)
            except BinanceAPIException as e:
                st.error(f"Erro ao ajustar SL (Venda/50% TP): {e}")

    return novo_stop_loss

def monitorar_oco_ativamente():
    while True:
        oco_keys_para_remover = []
        for key, order_info in oco_ordens_ativas.items():
            symbol_oco = key.split('_')[0]
            mercado_oco = key.split('_')[1]
            entry_order_id = key.split('_')[2]
            sl_order_id = order_info['sl_id']
            tp_order_id = order_info['tp_id']

            try:
                if mercado_oco == 'Spot' and client_spot:
                    sl_status = client_spot.get_order(symbol=symbol_oco, orderId=sl_order_id)['status']
                    tp_status = client_spot.get_order(symbol=symbol_oco, orderId=tp_order_id)['status']
                    if sl_status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'] or tp_status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        oco_keys_para_remover.append(key)
                        st.info(f"OCO para ordem de entrada {entry_order_id} ({symbol_oco}, Spot) finalizado. SL Status: {sl_status}, TP Status: {tp_status}")
                elif mercado_oco == 'Futuros' and client_futures:
                    sl_status = client_futures.futures_get_order(symbol=symbol_oco, orderId=sl_order_id)['status']
                    tp_status = client_futures.futures_get_order(symbol=symbol_oco, orderId=tp_order_id)['status']
                    if sl_status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'] or tp_status in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                        oco_keys_para_remover.append(key)
                        st.info(f"OCO para ordem de entrada {entry_order_id} ({symbol_oco}, Futuros) finalizado. SL Status: {sl_status}, TP Status: {tp_status}")
            except BinanceAPIException as e:
                st.error(f"Erro ao verificar status OCO ({symbol_oco}, {mercado_oco}): {e}")

        for key_remover in oco_keys_para_remover:
            if key_remover in oco_ordens_ativas:
                del oco_ordens_ativas[key_remover]

        time.sleep(5)

# ----------- EXECUÇÃO PRINCIPAL (MANTENDO A SUA) -----------

saldo = obter_saldo()
st.sidebar.markdown(f"💰 **Saldo disponível:**{saldo}")

df = obter_dados(symbol, interval, qtde_dados, tipo_mercado)

col_grafico, col_acoes = st.columns([3, 1])

if not df.empty:
    df_indicadores = calcular_indicadores(df.copy())
    with col_grafico:
        st.plotly_chart(plotar_grafico(df_indicadores, usar_mm, usar_rsi, usar_macd, usar_bb, exibir_mm200_grafico, exibir_ema400_grafico, exibir_medias_rapidas), use_container_width=True)

    with col_acoes:
        st.subheader("🛠️ Ações")
        preco_atual = df['close'].iloc[-1]

        # Calcular quantidade (MANTENDO A SUA)
        try:
            saldo_disponivel = float(saldo.split(":")[1].strip())
            quantidade_trade = (saldo_disponivel * (quantidade_trade_pct / 100)) / preco_atual
            info_symbol = None
            if tipo_mercado == "Spot" and client_spot:
                info_symbol = client_spot.get_symbol_info(symbol)
            elif tipo_mercado == "Futuros" and client_futures:
                info_symbol = get_futures_symbol_info(client_futures, symbol)

            if info_symbol and 'filters' in info_symbol:
                lot_filter = next((f for f in info_symbol['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                if lot_filter:
                    step_size = float(lot_filter['stepSize'])
                    quantidade_trade = (quantidade_trade // step_size) * step_size
                else:
                    quantidade_trade = round(quantidade_trade, 8)
            else:
                quantidade_trade = round(quantidade_trade, 8)

            quantidade_trade = max(quantidade_trade, 0.01)
        except Exception as e:
            st.warning(f"Erro ao calcular a quantidade do trade: {e}")
            quantidade_trade = 0.01

        oco_ativo_manual = st.checkbox("Usar OCO (Manual)", value=False)

        if st.button("🔼 Comprar (Manual)"):
            if quantidade_trade > 0:
                sl_pct = stop_loss_manual if definir_sl_tp_manualmente and stop_loss_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][0]
                tp_pct = take_profit_manual if definir_sl_tp_manualmente and take_profit_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][1]
                executar_ordem(symbol, SIDE_BUY, quantidade_trade, sl_pct, tp_pct, oco_ativo_manual)
            else:
                st.warning("Quantidade de trade inválida.")
        if st.button("🔽 Vender (Manual)"):
            if quantidade_trade > 0:
                sl_pct = stop_loss_manual if definir_sl_tp_manualmente and stop_loss_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][0]
                tp_pct = take_profit_manual if definir_sl_tp_manualmente and take_profit_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][1]
                executar_ordem(symbol, SIDE_SELL, quantidade_trade, sl_pct, tp_pct, oco_ativo_manual)
            else:
                st.warning("Quantidade de trade inválida.")

        st.markdown("---")
        st.subheader("🤖 Estratégia (Automático)")
        sinal = tomar_decisao(df_indicadores.copy(), estrategia_ativa, nivel_risco)
        st.info(f"Sinal atual da estratégia ({estrategia_ativa}, {nivel_risco}): **{sinal}**")

        oco_ativo_auto = st.checkbox("Usar OCO (Automático)", value=True)

        if robo_ativo:
            if sinal == "Compra":
                st.info("🤖 Oportunidade de COMPRA detectada!")  # Nova linha de informação
                if quantidade_trade > 0:
                    # APLICA A MESMA LÓGICA DA EXECUÇÃO MANUAL
                    sl_pct = stop_loss_manual if definir_sl_tp_manualmente and stop_loss_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][0]
                    tp_pct = take_profit_manual if definir_sl_tp_manualmente and take_profit_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][1]
                    st.info(f"📈 Usando SL: {sl_pct}%, TP: {tp_pct}% (Automático)")
                    st.info("🚀 Executando ordem de COMPRA...")  # Mensagem antes da execução
                    executar_ordem(symbol, SIDE_BUY, quantidade_trade, sl_pct, tp_pct, oco_ativo_auto)
                else:
                    st.warning("Quantidade de trade inválida.")
            elif sinal == "Venda":
                st.info("🤖 Oportunidade de VENDA detectada!")  # Nova linha de informação
                if quantidade_trade > 0:
                    # APLICA A MESMA LÓGICA DA EXECUÇÃO MANUAL
                    sl_pct = stop_loss_manual if definir_sl_tp_manualmente and stop_loss_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][0]
                    tp_pct = take_profit_manual if definir_sl_tp_manualmente and take_profit_manual else niveis_risco[nivel_risco]['sl_tp_ratio'][1]
                    st.info(f"📉 Usando SL: {sl_pct}%, TP: {tp_pct}% (Automático)")
                    st.info("📉 Executando ordem de VENDA...")  # Mensagem antes da execução
                    executar_ordem(symbol, SIDE_SELL, quantidade_trade, sl_pct, tp_pct, oco_ativo_auto)
                else:
                    st.warning("Quantidade de trade inválida.")
            elif sinal == "Neutro":
                st.info("😴 Nenhum sinal de trading detectado.")
        else:
            st.info("Robô de trading desativado. Ative para execução automática.")

        st.markdown("---")
        if oco_ordens_ativas:
            st.subheader("OCO Ativas")
            oco_df = pd.DataFrame.from_dict(oco_ordens_ativas, orient='index')
            st.dataframe(oco_df)
            if st.button("🛑 Cancelar Todas as OCO"):
                for key in list(oco_ordens_ativas.keys()):
                    symbol_oco = key.split('_')[0]
                    mercado_oco = key.split('_')[1].capitalize()
                    sl_id_oco = oco_ordens_ativas[key]['sl_id']
                    tp_id_oco = oco_ordens_ativas[key]['tp_id']
                    if mercado_oco == 'Spot' and client_spot:
                        try:
                            client_spot.cancel_order(symbol=symbol_oco, orderId=sl_id_oco)
                            client_spot.cancel_order(symbol=symbol_oco, orderId=tp_id_oco)
                            st.info(f"✅ (Spot) Ordens OCO para {symbol_oco} canceladas (SL ID: {sl_id_oco}, TP ID: {tp_id_oco}).")
                            del oco_ordens_ativas[key]
                        except BinanceAPIException as e:
                            st.error(f"Erro ao cancelar ordens OCO (Spot) para {symbol_oco}: {e}")
                    elif mercado_oco == 'Futuros' and client_futures:
                        try:
                            client_futures.futures_cancel_order(symbol=symbol_oco, orderId=sl_id_oco)
                            client_futures.futures_cancel_order(symbol=symbol_oco, orderId=tp_id_oco)
                            st.info(f"✅ (Futuros) Ordens OCO para {symbol_oco} canceladas (SL ID: {sl_id_oco}, TP ID: {tp_id_oco}).")
                            del oco_ordens_ativas[key]
                        except BinanceAPIException as e:
                            st.error(f"Erro ao cancelar ordens OCO (Futuros) para {symbol_oco}: {e}")
        else:
            st.info("Nenhuma ordem OCO ativa.")

st.markdown("---")
st.subheader("📜 Histórico de Ordens")
df_ordens = carregar_ordens(symbol)
if not df_ordens.empty:
    st.dataframe(df_ordens)
else:
    st.info("Nenhuma ordem encontrada ainda.")

st.markdown("---")
st.subheader("📈 Performance da Simulação")
df_performance_simulada = calcular_resultado_simulado(symbol)
if not df_performance_simulada.empty:
    st.metric("Lucro/Prejuízo Total (Simulado)", f"${df_performance_simulada['cumulative_profit'].iloc[-1]:.2f}")
    st.line_chart(df_performance_simulada.set_index("timestamp")[["cumulative_profit"]])
else:
    st.info("Nenhuma ordem simulada para exibir performance.")

st.markdown("---")
st.caption("App de trading com integração Binance (Spot & Futuros) desenvolvido com Streamlit.")























