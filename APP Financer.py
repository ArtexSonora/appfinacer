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
from scipy.signal import argrelextrema
from tensorflow import keras
from tensorflow.keras import layers
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import collections
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

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

def ativar_robo():
    st.session_state['robo_ativo'] = True  # Adicionei 4 espaços de indentação aqui
    st.write("Botão de ativar robô foi clicado!") # E aqui também

st.button("Ativar Robô de Trader", on_click=ativar_robo)

if st.session_state.get('robo_ativo'):
    st.success("Robô de negociação ATIVADO!")
else:
    st.warning("Robô de negociação DESATIVADO.")
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

def main():
    st.title("Trading App com Estratégias")

# --- API Binance ---
api_key = st.sidebar.text_input("API Key Binance", type="password")
api_secret = st.sidebar.text_input("Secret Key Binance", type="password")

client = Client(api_key, api_secret) if api_key and api_secret else None

# Função para criar o modelo
# Função para Criar o Modelo LSTM com Múltiplas Saídas

if 'modelo_treinado_ia' not in st.session_state:
    try:
        st.info("Carregando o modelo de IA...")
        st.session_state['modelo_treinado_ia'] = load_model(
            '/root/appfinacer/modelo_ia_multi_output.h5', compile=False)
        model = st.session_state['modelo_treinado_ia']
        model.compile(optimizer=Adam(), loss=MeanSquaredError())
        st.success("Modelo de IA carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de IA: {e}")
        st.error(e)

modelo_treinado_ia = st.session_state.get('modelo_treinado_ia')
def criar_modelo_lstm_multi_output(num_features, num_outputs):
    modelo = keras.Sequential([
        layers.LSTM(100, activation='relu', input_shape=(None, num_features), return_sequences=True),
        layers.LSTM(100, activation='relu'),
        layers.Dense(num_outputs)  # Camada de saída com múltiplos neurônios
    ])
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])  # MSE para regressão, MAE como métrica
    return modelo

# Upload de dados
st.subheader("Carregar CSV para IA")
uploaded_file = st.file_uploader("CSV com dados", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    features_list = st.multiselect("Selecionar colunas de entrada", df.columns)
    outputs_list = st.multiselect("Selecionar colunas de saída (Open Time, Open, High, Low, Close, ...)", df.columns)

    if features_list and outputs_list:
        X = df[features_list].values
        y = df[outputs_list].values  # Agora y tem múltiplas colunas

        # Análise das Colunas de Saída
        st.subheader("Análise das Colunas de Saída")
        for i, output_name in enumerate(outputs_list):
            st.write(f"**{output_name}:**")
            st.write("  Tipo de dados:", y[:, i].dtype)
            st.write("  Valores únicos:", np.unique(y[:, i]))
            st.write("  Contagem de valores:", pd.Series(y[:, i]).value_counts())

        # Escala dos dados (Importante para LSTMs)
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)

        # Criar Sequências para LSTM
        def create_sequences(X, y, time_steps=10):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps=10)

        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

        num_features = len(features_list)
        num_outputs = len(outputs_list)
        modelo = criar_modelo_lstm_multi_output(num_features, num_outputs)

        # Treinamento
        if st.checkbox("Treinar Modelo"):
            historico = modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)
            st.success("Modelo treinado com sucesso!")

            # Exibir gráfico de evolução da perda
            st.line_chart(historico.history['loss'])

        # Avaliação
        if st.checkbox("Avaliar Modelo"):
            y_pred_scaled = modelo.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            y_true = scaler_y.inverse_transform(y_test)

            st.subheader("Métricas de Avaliação por Saída:")
            for i, output_name in enumerate(outputs_list):
                mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                st.write(f"**{output_name}:**")
                st.write(f"  MAE (escala original): {mae:.4f}")
                st.write(f"  R²: {r2:.4f}")

            # Validação Cruzada com TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            all_mae_scores = []
            all_r2_scores = []

            for train_index, val_index in tscv.split(X_seq):
                X_train_cv, X_val_cv = X_seq[train_index], X_seq[val_index]
                y_train_cv, y_val_cv = y_seq[train_index], y_seq[val_index]

                modelo_cv = criar_modelo_lstm_multi_output(num_features, num_outputs)

                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                historico_cv = modelo_cv.fit(X_train_cv, y_train_cv, epochs=100, batch_size=32,
                                            validation_data=(X_val_cv, y_val_cv),
                                            callbacks=[early_stopping],
                                            verbose=0)

                y_pred_val_scaled = modelo_cv.predict(X_val_cv)
                y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled)
                y_true_val = scaler_y.inverse_transform(y_val_cv)

                mae_cv = []
                r2_cv = []
                for i in range(num_outputs):
                    mae_cv.append(mean_absolute_error(y_true_val[:, i], y_pred_val[:, i]))
                    r2_cv.append(r2_score(y_true_val[:, i], y_pred_val[:, i]))

                all_mae_scores.append(mae_cv)
                all_r2_scores.append(r2_cv)

            avg_mae = np.mean(all_mae_scores, axis=0)
            avg_r2 = np.mean(all_r2_scores, axis=0)

            st.subheader("Média MAE Validação Cruzada por Saída:")
            for i, output_name in enumerate(outputs_list):
                st.write(f"**{output_name}:**")
                st.write(f"  Média MAE: {avg_mae[i]:.4f}")
                st.write(f"  Média R²: {avg_r2[i]:.4f}")

        # Salvar Modelo
        if st.checkbox("Salvar Modelo"):
            nome_modelo = st.text_input("Nome do Arquivo", "modelo_ia_multi_output.h5")
            if st.button("Salvar"):
                modelo.save(nome_modelo)
                st.success(f"Modelo salvo como {nome_modelo}")

# --- Obter Dados da Binance ---
st.subheader("Obter Dados Históricos da Binance")
symbol = st.text_input("Par Ex: BTCUSDT", "BTCUSDT").upper()
interval = st.selectbox("Intervalo", ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"], index=6)
start_date = st.text_input("Data Início (YYYY-MM-DD)", "2023-01-01")
end_date = st.text_input("Data Fim (opcional)", "")

if st.button("Buscar Dados da Binance"):
    if client:
        try:
            klines = client.get_historical_klines(symbol, interval, start_date, end_date)
            if klines:
                df_binance = pd.DataFrame(klines, columns=[
                    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close Time', 'Quote Asset Volume', 'Number of Trades',
                    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
                ])
                df_binance['Open Time'] = pd.to_datetime(df_binance['Open Time'], unit='ms')
                df_binance['Close Time'] = pd.to_datetime(df_binance['Close Time'], unit='ms')

                for col in ['Open', 'High', 'Low', 'Close', 'Volume',
                            'Quote Asset Volume', 'Taker Buy Base Asset Volume',
                            'Taker Buy Quote Asset Volume']:
                    df_binance[col] = pd.to_numeric(df_binance[col])

                st.dataframe(df_binance)

                csv = io.StringIO()
                df_binance.to_csv(csv, index=False)
                st.download_button(
                    label="Baixar CSV",
                    data=csv.getvalue(),
                    file_name=f"{symbol}_binance_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("Nenhum dado retornado.")
        except Exception as e:
            st.error(f"Erro ao buscar dados: {e}")
    else:
        st.warning("API Key e Secret inválidas ou ausentes.")


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

        # *** VERIFICAÇÃO DAS COLUNAS 'high' e 'low' AQUI DENTRO DA FUNÇÃO obter_dados ***
        if 'high' not in df.columns or 'low' not in df.columns:
            st.error(f"Erro: As colunas 'high' e 'low' não foram encontradas nos dados obtidos da Binance ({mercado}).")
            return pd.DataFrame() # Retorna um DataFrame vazio para indicar erro

        return df
    except BinanceAPIException as e:
        st.error(f"Erro ao obter dados da Binance ({mercado}): {e}")
        return pd.DataFrame()

def calcular_niveis_fibonacci(preco_inicial, preco_final):
    """Calcula os níveis de retração e extensão de Fibonacci."""
    if preco_inicial > preco_final:
        # Tendência de baixa: preco_inicial é o topo, preco_final é o fundo
        fibo_0 = preco_final
        fibo_100 = preco_inicial
    else:
        # Tendência de alta: preco_inicial é o fundo, preco_final é o topo
        fibo_0 = preco_inicial
        fibo_100 = preco_final

    diferenca = fibo_100 - fibo_0
    niveis = {
        "0.0%": fibo_100,
        "23.6%": fibo_100 - 0.236 * diferenca,
        "38.2%": fibo_100 - 0.382 * diferenca,
        "50.0%": fibo_100 - 0.500 * diferenca,
        "61.8%": fibo_100 - 0.618 * diferenca,
        "78.6%": fibo_100 - 0.786 * diferenca,
        "100.0%": fibo_0,
        "127.2%": fibo_0 - 0.272 * diferenca if preco_inicial > preco_final else fibo_100 + 0.272 * diferenca,
        "161.8%": fibo_0 - 0.618 * diferenca if preco_inicial > preco_final else fibo_100 + 0.618 * diferenca,
    }
    return niveis
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

      # ... (seu código para seleção de par, timeframe, nível de risco) ...

    df = obter_dados(par_selecionado, intervalo_selecionado)
    if not df.empty:
        # Calcular indicadores necessários para as estratégias
        df['MM200'] = df['close'].rolling(window=200).mean()
        df['EMA400'] = df['close'].ewm(span=400, adjust=False).mean()
        df['RSI'] = ta.rsi(df['close'], window=config_risco['rsi_periodo'])
        macd = ta.macd(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['MACD'] = macd['macd']
        df['Signal_MACD'] = macd['macd_signal']
        df['Histograma_MACD'] = macd['macd_diff']
        bb = ta.bbands(df['close'], window=config_risco['bb_periodo'], std=config_risco['bb_desvio'])
        df['BB_Superior'] = bb['bb_h']
        df['BB_Inferior'] = bb['bb_l']
        df['BB_Media'] = bb['bb_mavg']

        sinal = tomar_decisao(df.copy(), estrategia_selecionada, nivel_risco_selecionado)
        st.subheader(f"Sinal de Trading ({estrategia_selecionada}): {sinal}")

        if estrategia_selecionada == "Fibonacci":
            st.sidebar.subheader("⚙️ Configuração Fibonacci")
            if not df.empty:
                timestamps = df.index.tolist()
                ponto_inicial_fibonacci = st.sidebar.selectbox("Selecionar Ponto Inicial", timestamps)
                ponto_final_fibonacci = st.sidebar.selectbox("Selecionar Ponto Final", timestamps)
                tendencia_fibonacci = st.sidebar.selectbox("Tendência (para sinal)", ["Alta", "Baixa"])

                preco_inicial_fibonacci = df.loc[ponto_inicial_fibonacci]['close']
                preco_final_fibonacci = df.loc[ponto_final_fibonacci]['close']

                niveis_fibonacci = calcular_niveis_fibonacci(preco_inicial_fibonacci, preco_final_fibonacci)
                niveis_fibonacci_plot = niveis_fibonacci  # Para plotagem

                sinal_fibonacci = gerar_sinal_fibonacci(df['close'].iloc[-1], niveis_fibonacci, tendencia_fibonacci)
                st.subheader(f"Sinal Fibonacci: {sinal_fibonacci}")

                # Chamando tomar_decisao para a estratégia Fibonacci
                sinal_decisao = tomar_decisao(df.copy(), estrategia_selecionada, nivel_risco_selecionado,
                                              sinal_fibonacci=sinal_fibonacci)
                st.subheader(f"Sinal de Trading ({estrategia_selecionada}): {sinal_decisao}")

            else:
                st.sidebar.warning("Dados insuficientes para Fibonacci.")
        elif estrategia_selecionada != "Neutro":
            df_indicadores = calcular_indicadores(df.copy())
            sinal_decisao = tomar_decisao(df_indicadores.copy(), estrategia_selecionada, nivel_risco_selecionado)
            st.subheader(f"Sinal de Trading ({estrategia_selecionada}): {sinal_decisao}")
        else:
            sinal_decisao = "Neutro"
            st.subheader("Nenhuma estratégia selecionada.")

if __name__ == "__main__":
    main()


def plotar_grafico(df, usar_mm, usar_rsi, usar_macd, usar_bb, exibir_mm200_grafico, exibir_ema400_grafico, exibir_medias_rapidas, niveis_fibonacci=None):
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

    # Adicionar Níveis de Fibonacci ao gráfico SE niveis_fibonacci for fornecido
    if niveis_fibonacci:
        for nivel, preco in niveis_fibonacci.items():
            fig.add_trace(go.Scatter(
                x=df.index,
                y=[preco] * len(df.index),
                mode='lines',
                name=f"Fib {nivel}",
                line=dict(color='gray', dash='dash')
            ))

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

# CONFIGURAÇÕES DE RISCO
niveis_risco = {
    "Suave": {"mm_rapida": 9, "mm_lenta": 21, "rsi_limites": (30, 70), "sl_tp_ratio": (0.5, 2)},
    "Moderado": {"mm_rapida": 12, "mm_lenta": 26, "rsi_limites": (35, 65), "sl_tp_ratio": (0.75, 1.5)},
    "Profissional": {"mm_rapida": 20, "mm_lenta": 50, "rsi_limites": (40, 60), "sl_tp_ratio": (1, 2)},
    "Agressivo": {"mm_rapida": 5, "mm_lenta": 15, "rsi_limites": (25, 75), "sl_tp_ratio": (1, 3)},
    "Lara": {
        "lara_ordem_picos_fundos": 5,
        "lara_tolerancia_altura_dt": 0.005,
        "lara_tolerancia_distancia_dt": 10,
        "lara_tolerancia_altura_hocoi": 0.02,
        "lara_tolerancia_distancia_hocoi": 20,
        "lara_tolerancia_angulo_cw": 0.1
    }
}

# INPUTS DO USUÁRIO
with st.sidebar:
    st.subheader("⚙️ Configurações")
    modo_operacao = st.selectbox("Modo de Operação", ["Simulado", "Real"])
    tipo_mercado = st.radio("Mercado", ["Spot", "Futuros"])

    usar_ia = st.checkbox("Ativar Estratégia de Rede Neural", value=False)
    usar_confluencia_ia = False  # Inicializa como False


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
    lista_estrategias = ["Neutro", "MM Cruzamento", "RSI (Toque nas Extremidades)", "MACD Cruzamento",
                         "Bandas de Bollinger (Toque)", "Lara Reversao", "Confluência Manual",
                         "Cruzamento MM + BB Fluxo", "Fibonacci", "Rede Neural"]
    estrategia_ativa = st.selectbox("Selecione a Estratégia:", lista_estrategias)

    usar_confirmacao_rsi_compra = st.checkbox("Usar Confirmação RSI Compra (50-70)", value=False)
    usar_confirmacao_rsi_venda = st.sidebar.checkbox("Usar Confirmação RSI Venda (30-50)", value=False)
    usar_confirmacao_volume = st.checkbox("Confirmar entrada com volume 20% acima do candle anterior", value=False)
    usar_filtro_mm200_global = st.checkbox("Usar Filtro MM200")
    usar_filtro_ema400_global = st.checkbox("Usar Filtro EMA400")
    st.subheader("➕ Confluência Manual")
    usar_confluencia_manual = st.checkbox("Ativar Confluência Manual", value=False)
    usar_confluencia_mm = st.checkbox("Usar MM na Confluência", value=False)
    usar_confluencia_rsi = st.checkbox("Usar RSI na Confluência", value=False)
    usar_confluencia_macd = st.checkbox("Usar MACD na Confluência", value=False)
    usar_confluencia_bb = st.checkbox("Usar BB na Confluência", value=False)
    usar_confluencia_lara = st.checkbox("Usar Padrões Lara na Confluência",
                                        value=False)  # Checkbox de confluência para Lara

    st.subheader("⚙️ Configurações Futuros")
    alavancagem = st.slider("Alavancagem", min_value=1, max_value=125, value=1, step=1, key='alavancagem')

    nivel_risco = st.selectbox("Nível de Risco",
                               ["Suave", "Moderado", "Profissional", "Agressivo", "Lara"])  # Adicionando "Lara"
    quantidade_trade_pct = st.number_input("Quantidade por Trade (%) da Banca", min_value=0.01, max_value=100.0,
                                           value=1.0, step=0.01)
    definir_sl_tp_manualmente = st.checkbox("Definir Stop Loss/Take Profit Manualmente")
    if definir_sl_tp_manualmente:
        sl_padrao = float(
            niveis_risco[nivel_risco]["sl_tp_ratio"][0]) if nivel_risco in niveis_risco and "sl_tp_ratio" in \
                                                            niveis_risco[nivel_risco] else 1.0
        tp_padrao = float(
            niveis_risco[nivel_risco]["sl_tp_ratio"][1]) if nivel_risco in niveis_risco and "sl_tp_ratio" in \
                                                            niveis_risco[nivel_risco] else 2.0
        stop_loss_manual = st.number_input("Stop Loss (%)", min_value=0.01, step=0.01, value=sl_padrao)
        take_profit_manual = st.number_input("Take Profit (%)", min_value=0.01, step=0.01, value=tp_padrao)
    else:
        stop_loss_manual = None
        take_profit_manual = None

    st.subheader("⚙️ Parâmetros da Estratégia")
    config_atual = niveis_risco[nivel_risco]
    if usar_mm:
        st.markdown(f"**Média Móvel:**")
        st.markdown(f"- Rápida: {config_atual.get('mm_rapida', '-')}")
        st.markdown(f"- Lenta: {config_atual.get('mm_lenta', '-')}")
    if usar_rsi:
        st.markdown(f"**RSI:**")
        st.markdown(f"- Limite Inferior: {config_atual.get('rsi_limites', ('-', '-'))[0]}")
        st.markdown(f"- Limite Superior: {config_atual.get('rsi_limites', ('-', '-'))[1]}")
    if usar_macd:
        st.markdown(f"**MACD:**")
        st.markdown("- EMA Rápida: 12")
        st.markdown("- EMA Lenta: 26")
        st.markdown("- Sinal EMA: 9")
    if usar_bb:
        st.markdown(f"**Bandas de Bollinger:**")
        st.markdown("- Período: 20")
        st.markdown("- Desvio Padrão: 2")
    if estrategia_ativa == "Lara Reversao":
        st.markdown("**Padrões de Reversão de Lara:**")
        st.markdown("- Ordem para picos/fundos: 5 (padrão)")  # Informando o parâmetro
        st.markdown("- Tolerância de altura (Dobro Topo/Fundo): 0.5% (padrão)")
        st.markdown("- Tolerância de distância (Dobro Topo/Fundo): 10 candles (padrão)")
        st.markdown("- Tolerância de altura (Ombro/Cabeça): 2% (padrão)")
        st.markdown("- Tolerância de distância (Ombros): 20 candles (padrão)")
        st.markdown("- Tolerância de altura (Triplo Topo/Fundo): 0.5% (padrão)")
        st.markdown("- Tolerância de distância (Triplo Topo/Fundo): 20 candles (padrão)")
        st.markdown("- Tolerância de ângulo (Cunha): 0.1 (padrão)")

        if usar_ia:
            st.subheader("Configurações da Rede Neural")
            df_ia_local = st.session_state.get('df_ia')
            if df_ia_local is not None and 'columns' in dir(df_ia_local):
                colunas_disponiveis_ia = df_ia_local.columns
                colunas_features_ia = st.multiselect("Features (Entradas) para a IA", colunas_disponiveis_ia)
                coluna_alvo_ia = st.selectbox("Alvo (Saída) para a IA", colunas_disponiveis_ia)
                sua_saida_eh_classificacao_ia = st.checkbox("Prever Classificação (Alta/Baixa)?", value=True)
                usar_confluencia_ia = st.checkbox("Usar IA em Confluência com Outras Estratégias", value=True)
            else:
                st.warning("Por favor, carregue os dados para a IA na seção correspondente.")
        else:
            usar_confluencia_ia = False

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

def aplicar_confirmacao_rsi(sinal, df, usar_confirmacao_compra, usar_confirmacao_venda):
    if 'RSI' not in df.columns:
        return sinal  # RSI não disponível, retorna o sinal original

    rsi_atual = df['RSI'].iloc[-1]
    sinal_final = sinal

    if sinal == "Compra":
        if usar_confirmacao_compra:
            if not (50 < rsi_atual < 70):
                st.info(f"Sinal de compra do RSI ({rsi_atual:.2f}), mas fora da zona de confirmação (50-70).")
                sinal_final = "Neutro"
    elif sinal == "Venda":
        if usar_confirmacao_venda:
            if not (30 < rsi_atual < 50):
                st.info(f"Sinal de venda do RSI ({rsi_atual:.2f}), mas fora da zona de confirmação (30-50).")
                sinal_final = "Neutro"

    return sinal_final
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

def gerar_sinal_fibonacci(preco_atual, niveis_fibonacci, tendencia):
    """Gera um sinal de negociação baseado nos níveis de Fibonacci e na tendência."""
    tolerancia = 0.005 * (max(niveis_fibonacci.values()) - min(niveis_fibonacci.values())) # 0.5% de tolerância

    sinal = "Neutro"

    if tendencia == "Alta":
        # Procurar por sinais de compra perto dos níveis de retração (abaixo de 100%)
        if (niveis_fibonacci["38.2%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["38.2%"] * (1 + tolerancia)):
            sinal = "Compra (Retração 38.2%)"
        elif (niveis_fibonacci["50.0%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["50.0%"] * (1 + tolerancia)):
            sinal = "Compra (Retração 50.0%)"
        elif (niveis_fibonacci["61.8%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["61.8%"] * (1 + tolerancia)):
            sinal = "Compra (Retração 61.8%)"
        elif (niveis_fibonacci["78.6%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["78.6%"] * (1 + tolerancia)):
            sinal = "Compra (Retração 78.6%)"

    elif tendencia == "Baixa":
        # Procurar por sinais de venda perto dos níveis de retração (acima de 0%)
        if (niveis_fibonacci["23.6%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["23.6%"] * (1 + tolerancia)):
            sinal = "Venda (Retração 23.6%)"
        elif (niveis_fibonacci["38.2%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["38.2%"] * (1 + tolerancia)):
            sinal = "Venda (Retração 38.2%)"
        elif (niveis_fibonacci["50.0%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["50.0%"] * (1 + tolerancia)):
            sinal = "Venda (Retração 50.0%)"
        elif (niveis_fibonacci["61.8%"] * (1 - tolerancia) < preco_atual < niveis_fibonacci["61.8%"] * (1 + tolerancia)):
            sinal = "Venda (Retração 61.8%)"

    return sinal

def identificar_picos_e_fundos(df, ordem=5):
    """Identifica picos e fundos locais em uma série temporal."""
    if len(df) < ordem * 2 + 1:
        return np.array([]), np.array([])
    picos_idx = argrelextrema(df['high'].values, np.greater, order=ordem)[0]
    fundos_idx = argrelextrema(df['low'].values, np.less, order=ordem)[0]
    return picos_idx, fundos_idx

def verificar_dobro_topo(df, picos_idx, tolerancia_altura=0.005, tolerancia_distancia=10):
    """Verifica a formação de Dobro Topo."""
    if len(picos_idx) < 2:
        return None
    for i in range(len(picos_idx) - 1):
        pico1_idx = picos_idx[i]
        pico2_idx = picos_idx[i+1]
        if abs(df['high'].iloc[pico1_idx] - df['high'].iloc[pico2_idx]) < tolerancia_altura * df['high'].iloc[pico1_idx] and \
           abs(pico1_idx - pico2_idx) > 1 and abs(pico1_idx - pico2_idx) < tolerancia_distancia:
            # Encontrar o vale entre os picos
            vale_start = min(pico1_idx, pico2_idx)
            vale_end = max(pico1_idx, pico2_idx)
            if vale_end - vale_start > 1:
                vale_idx = df['low'].iloc[vale_start+1:vale_end].idxmin()
                return pico1_idx, pico2_idx, df.index.get_loc(vale_idx)
    return None

def verificar_dobro_fundo(df, fundos_idx, tolerancia_altura=0.005, tolerancia_distancia=10):
    """Verifica a formação de Dobro Fundo."""
    if len(fundos_idx) < 2:
        return None
    for i in range(len(fundos_idx) - 1):
        fundo1_idx = fundos_idx[i]
        fundo2_idx = fundos_idx[i+1]
        if abs(df['low'].iloc[fundo1_idx] - df['low'].iloc[fundo2_idx]) < tolerancia_altura * df['low'].iloc[fundo1_idx] and \
           abs(fundo1_idx - fundo2_idx) > 1 and abs(fundo1_idx - fundo2_idx) < tolerancia_distancia:
            # Encontrar o pico entre os fundos
            pico_start = min(fundo1_idx, fundo2_idx)
            pico_end = max(fundo1_idx, fundo2_idx)
            if pico_end - pico_start > 1:
                pico_idx = df['high'].iloc[pico_start+1:pico_end].idxmax()
                return fundo1_idx, fundo2_idx, df.index.get_loc(pico_idx)
    return None

def verificar_cabeca_ombros(df, picos_idx, tolerancia_altura_ombro_cabeca=0.02, tolerancia_distancia_ombros=20):
    """Verifica a formação de Cabeça e Ombros."""
    if len(picos_idx) < 3:
        return None
    for i in range(1, len(picos_idx) - 1):
        om1_idx = picos_idx[i-1]
        cabeca_idx = picos_idx[i]
        om2_idx = picos_idx[i+1]
        if om1_idx < cabeca_idx < om2_idx and \
           abs(df['high'].iloc[om1_idx] - df['high'].iloc[om2_idx]) < tolerancia_altura_ombro_cabeca * df['high'].iloc[cabeca_idx] and \
           df['high'].iloc[cabeca_idx] > df['high'].iloc[om1_idx] and df['high'].iloc[cabeca_idx] > df['high'].iloc[om2_idx] and \
           abs(om1_idx - om2_idx) < tolerancia_distancia_ombros:
            # Encontrar a linha de pescoço (suporte)
            pescoco1_start = min(om1_idx, cabeca_idx)
            pescoco1_end = max(om1_idx, cabeca_idx)
            if pescoco1_end - pescoco1_start > 0:
                pescoco1_idx = df['low'].iloc[pescoco1_start:pescoco1_end].idxmax() # Máxima entre ombro1 e cabeça

            pescoco2_start = min(cabeca_idx, om2_idx)
            pescoco2_end = max(cabeca_idx, om2_idx)
            if pescoco2_end - pescoco2_start > 0:
                pescoco2_idx = df['low'].iloc[pescoco2_start:pescoco2_end].idxmax() # Máxima entre cabeça e ombro2

            return om1_idx, cabeca_idx, om2_idx, df.index.get_loc(pescoco1_idx), df.index.get_loc(pescoco2_idx)
    return None

def verificar_cabeca_ombros_invertido(df, fundos_idx, tolerancia_altura_ombro_cabeca=0.02, tolerancia_distancia_ombros=20):
    """Verifica a formação de Cabeça e Ombros Invertido."""
    if len(fundos_idx) < 3:
        return None
    for i in range(1, len(fundos_idx) - 1):
        om1_idx = fundos_idx[i-1]
        cabeca_idx = fundos_idx[i]
        om2_idx = fundos_idx[i+1]
        if om1_idx < cabeca_idx < om2_idx and \
           abs(df['low'].iloc[om1_idx] - df['low'].iloc[om2_idx]) < tolerancia_altura_ombro_cabeca * df['low'].iloc[cabeca_idx] and \
           df['low'].iloc[cabeca_idx] < df['low'].iloc[om1_idx] and df['low'].iloc[cabeca_idx] < df['low'].iloc[om2_idx] and \
           abs(om1_idx - om2_idx) < tolerancia_distancia_ombros:
            # Encontrar a linha de pescoço (resistência)
            pescoco1_start = min(om1_idx, cabeca_idx)
            pescoco1_end = max(om1_idx, cabeca_idx)
            if pescoco1_end - pescoco1_start > 0:
                pescoco1_idx = df['high'].iloc[pescoco1_start:pescoco1_end].idxmax() # Máxima entre ombro1 e cabeça

            pescoco2_start = min(cabeca_idx, om2_idx)
            pescoco2_end = max(cabeca_idx, om2_idx)
            if pescoco2_end - pescoco2_start > 0:
                pescoco2_idx = df['high'].iloc[pescoco2_start:pescoco2_end].idxmax() # Máxima entre cabeça e ombro2

            return om1_idx, cabeca_idx, om2_idx, df.index.get_loc(pescoco1_idx), df.index.get_loc(pescoco2_idx)
    return None

def verificar_triplo_topo(df, picos_idx, tolerancia_altura=0.005, tolerancia_distancia=20):
    """Verifica a formação de Triplo Topo."""
    if len(picos_idx) < 3:
        return None
    for i in range(len(picos_idx) - 2):
        p1_idx = picos_idx[i]
        p2_idx = picos_idx[i+1]
        p3_idx = picos_idx[i+2]
        if abs(df['high'].iloc[p1_idx] - df['high'].iloc[p2_idx]) < tolerancia_altura * df['high'].iloc[p1_idx] and \
           abs(df['high'].iloc[p1_idx] - df['high'].iloc[p3_idx]) < tolerancia_altura * df['high'].iloc[p1_idx] and \
           p1_idx < p2_idx < p3_idx and \
           abs(p1_idx - p3_idx) < tolerancia_distancia:
            # Encontrar os vales entre os picos
            v1_start = min(p1_idx, p2_idx)
            v1_end = max(p1_idx, p2_idx)
            v2_start = min(p2_idx, p3_idx)
            v2_end = max(p2_idx, p3_idx)
            if v1_end - v1_start > 0 and v2_end - v2_start > 0:
                v1_idx = df['low'].iloc[v1_start+1:v1_end].idxmin()
                v2_idx = df['low'].iloc[v2_start+1:v2_end].idxmin()
                return p1_idx, p2_idx, p3_idx, df.index.get_loc(v1_idx), df.index.get_loc(v2_idx)
    return None

def verificar_triplo_fundo(df, fundos_idx, tolerancia_altura=0.005, tolerancia_distancia=20):
    """Verifica a formação de Triplo Fundo."""
    if len(fundos_idx) < 3:
        return None
    for i in range(len(fundos_idx) - 2):
        f1_idx = fundos_idx[i]
        f2_idx = fundos_idx[i+1]
        f3_idx = fundos_idx[i+2]
        if abs(df['low'].iloc[f1_idx] - df['low'].iloc[f2_idx]) < tolerancia_altura * df['low'].iloc[f1_idx] and \
           abs(df['low'].iloc[f1_idx] - df['low'].iloc[f3_idx]) < tolerancia_altura * df['low'].iloc[f1_idx] and \
           f1_idx < f2_idx < f3_idx and \
           abs(f1_idx - f3_idx) < tolerancia_distancia:
            # Encontrar os picos entre os fundos
            p1_start = min(f1_idx, f2_idx)
            p1_end = max(f1_idx, f2_idx)
            p2_start = min(f2_idx, f3_idx)
            p2_end = max(f2_idx, f3_idx)
            if p1_end - p1_start > 0 and p2_end - p2_start > 0:
                p1_idx = df['high'].iloc[p1_start+1:p1_end].idxmax()
                p2_idx = df['high'].iloc[p2_start+1:p2_end].idxmax()
                return f1_idx, f2_idx, f3_idx, df.index.get_loc(p1_idx), df.index.get_loc(p2_idx)
    return None

def verificar_falling_wedge(df, ordem=5, tolerancia_angulo=0.1):
    """Verifica a formação de Cunha de Queda."""
    picos_idx, _ = identificar_picos_e_fundos(df, ordem=ordem)
    if len(picos_idx) < 2:
        return None
    picos_idx = sorted(picos_idx[-5:]) # Analisar os últimos picos
    if len(picos_idx) < 2:
        return None

    pico1_idx = picos_idx[0]
    pico2_idx = picos_idx[-1]

    if pico1_idx < pico2_idx:
        # Calcular inclinação da linha de resistência
        delta_t = pico2_idx - pico1_idx
        delta_p = df['high'].iloc[pico2_idx] - df['high'].iloc[pico1_idx]
        inclinacao_resistencia = delta_p / delta_t

        # Encontrar fundos correspondentes
        _, fundos_idx = identificar_picos_e_fundos(df, ordem=ordem)
        fundos_idx = sorted([f for f in fundos_idx if f > pico1_idx and f < pico2_idx])
        if len(fundos_idx) >= 2:
            fundo1_idx = min(fundos_idx)
            fundo2_idx = max(fundos_idx)
            if fundo1_idx < fundo2_idx:
                delta_t_f = fundo2_idx - fundo1_idx
                delta_p_f = df['low'].iloc[fundo2_idx] - df['low'].iloc[fundo1_idx]
                inclinacao_suporte = delta_p_f / delta_t_f

                # Verificar se ambas as linhas são descendentes e convergentes
                if inclinacao_resistencia < 0 and inclinacao_suporte < 0 and inclinacao_resistencia > inclinacao_suporte and abs(inclinacao_resistencia - inclinacao_suporte) < tolerancia_angulo:
                    return pico1_idx, pico2_idx, fundo1_idx, fundo2_idx
    return None

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
def tomar_decisao(df, estrategia, nivel_risco, usar_ia, usar_confluencia_ia, sinal_fibonacci=None):
    config = niveis_risco[nivel_risco]
    sinal_final = "Neutro"
    sinais = []

    ordem_picos_fundos = config.get('lara_ordem_picos_fundos', 5)

    print("Chamando identificar_picos_e_fundos...")
    picos_idx, fundos_idx = identificar_picos_e_fundos(df, ordem_picos_fundos)
    print(f"picos_idx após chamada: {picos_idx}")
    print(f"fundos_idx após chamada: {fundos_idx}")

    if estrategia == "MM Cruzamento" and usar_mm:
        df['MM_Rapida'] = df['close'].rolling(window=config['mm_rapida']).mean()
        df['MM_Lenta'] = df['close'].rolling(window=config['mm_lenta']).mean()
        sinais_mm = gerar_sinal_mm_cruzamento(df, config['mm_rapida'], config['mm_lenta'])

        if usar_confluencia_mm:
            if sinais_mm == "Compra":
                sinais.append("Compra")
            elif sinais_mm == "Venda":
                sinais.append("Venda")
            else:
                sinais.append("Neutro")  # Ou talvez não adicionar nada à lista de sinais
        elif not usar_confluencia_manual:  # Se não estiver em confluência manual, usa o sinal individual
            sinal_final = sinais_mm
    elif estrategia == "RSI (Toque nas Extremidades)" and usar_rsi:
        sinal_extremidades = gerar_sinal_rsi_extremidades(df, config['rsi_limites'][0], config['rsi_limites'][1])
        sinal_divergencia = verificar_divergencia_rsi(df)
        sinais_rsi_base = sinal_divergencia if sinal_divergencia != "Neutro" else sinal_extremidades
        sinal_final = sinais_rsi_base  # Inicializa o sinal final com o sinal base
        if usar_confluencia_rsi:
            sinais.append(sinal_final)
        elif not usar_confluencia_manual:
            pass  # O sinal final já está definido

    elif estrategia == "RSI" and usar_rsi:
        rsi_atual = df['RSI'].iloc[-1]
        sinal_rsi_base = "Neutro"
        if rsi_atual < config['rsi_sobrevenda']:
            sinal_rsi_base = "Compra"
        elif rsi_atual > config['rsi_sobrecompra']:
            sinal_rsi_base = "Venda"
        sinal_final = aplicar_confirmacao_rsi(sinal_rsi_base, df, usar_confirmacao_rsi_compra,
                                              usar_confirmacao_rsi_venda)
        if usar_confluencia_rsi:
            if sinal_final != "Neutro":
                sinais.append(sinal_final)
        elif not usar_confluencia_manual:
            pass  # O sinal final já está definido
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

    elif estrategia == "Lara Reversao":
        sinal_lara = "Neutro"
        tolerancia_altura_dt = config.get('lara_tolerancia_altura_dt', 0.005)
        tolerancia_distancia_dt = config.get('lara_tolerancia_distancia_dt', 10)
        tolerancia_altura_hocoi = config.get('lara_tolerancia_altura_hocoi', 0.02)
        tolerancia_distancia_hocoi = config.get('lara_tolerancia_distancia_hocoi', 20)
        tolerancia_angulo_cw = config.get('lara_tolerancia_angulo_cw', 0.1)

        # Verificar padrões em ordem de prioridade e confirmação

        dt_padrao = verificar_dobro_topo(df, picos_idx, tolerancia_altura_dt, tolerancia_distancia_dt)
        if dt_padrao and df['close'].iloc[-1] < df['low'].iloc[dt_padrao[2]]:
            sinal_lara = "Venda"
            st.info(f"Dobro Topo detectado. Rompimento em {df.index[dt_padrao[2]]}")

        elif df_padrao := verificar_dobro_fundo(df, fundos_idx, tolerancia_altura_dt, tolerancia_distancia_dt):
            if df['close'].iloc[-1] > df['high'].iloc[df_padrao[2]]:
                sinal_lara = "Compra"
                st.info(f"Dobro Fundo detectado. Rompimento em {df.index[df_padrao[2]]}")

        elif hs_padrao := verificar_cabeca_ombros(df, picos_idx, tolerancia_altura_hocoi, tolerancia_distancia_hocoi):
            pescoco = min(df['low'].iloc[hs_padrao[3]], df['low'].iloc[hs_padrao[4]])
            if df['close'].iloc[-1] < pescoco:
                sinal_lara = "Venda"
                st.info(f"Cabeça e Ombros detectado. Rompimento da linha de pescoço em {pescoco:.2f}")

        elif hsi_padrao := verificar_cabeca_ombros_invertido(df, fundos_idx, tolerancia_altura_hocoi,
                                                             tolerancia_distancia_hocoi):
            pescoco = max(df['high'].iloc[hsi_padrao[3]], df['high'].iloc[hsi_padrao[4]])
            if df['close'].iloc[-1] > pescoco:
                sinal_lara = "Compra"
                st.info(f"Cabeça e Ombros Invertido detectado. Rompimento da linha de pescoço em {pescoco:.2f}")

        elif tt_padrao := verificar_triplo_topo(df, picos_idx, tolerancia_altura_dt, tolerancia_distancia_dt):
            rompimento = max(df['low'].iloc[tt_padrao[3]], df['low'].iloc[tt_padrao[4]])
            if df['close'].iloc[-1] < rompimento:
                sinal_lara = "Venda"
                st.info(f"Triplo Topo detectado. Rompimento abaixo de {rompimento:.2f}")

        elif tf_padrao := verificar_triplo_fundo(df, fundos_idx, tolerancia_altura_dt, tolerancia_distancia_dt):
            rompimento = min(df['high'].iloc[tf_padrao[3]], df['high'].iloc[tf_padrao[4]])
            if df['close'].iloc[-1] > rompimento:
                sinal_lara = "Compra"
                st.info(f"Triplo Fundo detectado. Rompimento acima de {rompimento:.2f}")

        elif cw_padrao := verificar_falling_wedge(df, ordem_picos_fundos, tolerancia_angulo_cw):
            resistencia = df['high'].iloc[cw_padrao[1]]
            if df['close'].iloc[-1] > resistencia:
                sinal_lara = "Compra"
                st.info(f"Cunha de Queda detectada. Rompimento acima de {resistencia:.2f}")

        # Confluência
        if usar_confluencia_lara and sinal_lara != "Neutro":
            sinais.append(sinal_lara)
        elif not usar_confluencia_manual:
            sinal_final = sinal_lara

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
    elif estrategia == "Fibonacci" and sinal_fibonacci:
        sinal_final = sinal_fibonacci

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

     #FUNÇÃO DE REDE NEURAL

    sinal_ia = "Neutro (IA)"  # Inicializa o sinal da IA
    if usar_ia and 'modelo_treinado_ia' in locals() and 'colunas_features_ia' in locals() and colunas_features_ia and 'coluna_alvo_ia' in locals() and coluna_alvo_ia:
        try:
            features_para_ia_previsao = df[colunas_features_ia].iloc[[-1]].values
            previsao_ia = modelo_treinado_ia.predict(features_para_ia_previsao, verbose=0)[0][0]
            st.write(f"Previsão da Rede Neural: {previsao_ia:.4f}")

            if sua_saida_eh_classificacao_ia:
                if previsao_ia > 0.7:
                    sinal_ia = "Compra (IA)"
                elif previsao_ia < 0.3:
                    sinal_ia = "Venda (IA)"
            else:  # Regressão - adapte os limiares
                if previsao_ia > 0.01:
                    sinal_ia = "Compra (IA)"
                elif previsao_ia < -0.01:
                    sinal_ia = "Venda (IA)"

            if estrategia == "Rede Neural":
                sinal_final = f"{sinal_ia} (Principal)"
            else:
                sinais.append(sinal_ia)

        except Exception as e:
            st.error(f"Erro ao usar a Rede Neural para previsão: {e}")
            sinal_ia = "Neutro (IA)"

    # Lógica de Confluência
    sinais_ativos = [s for s in sinais if s != "Neutro" and "(IA)" not in s]
    sinais_ia_ativos = [s for s in sinais if "(IA)" in s and s != "Neutro (IA)"]

    if estrategia != "Rede Neural" and usar_ia and usar_confluencia_ia:
        if sinais_ativos and sinais_ia_ativos:
            # Lógica de confluência com a IA
            if any("Compra" in s for s in sinais_ativos) and any("Compra" in s for s in sinais_ia_ativos):
                sinal_final = "Compra (Confluência com IA)"
            elif any("Venda" in s for s in sinais_ativos) and any("Venda" in s for s in sinais_ia_ativos):
                sinal_final = "Venda (Confluência com IA)"
        elif not sinais_ativos and sinais_ia_ativos:
            sinal_final = f"{sinais_ia_ativos[-1]} (Autônoma)"
        elif sinais_ativos:
            sinal_final = sinais_ativos[-1]
    elif estrategia == "Rede Neural" and usar_ia:
        sinal_final = f"{sinal_ia} (Principal)"
    elif estrategia != "Rede Neural" and sinais_ativos and not usar_ia:
        sinal_final = sinais_ativos[-1]

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
        sinal = tomar_decisao(df_indicadores.copy(), estrategia_ativa, nivel_risco, usar_ia, usar_confluencia_ia)
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

# Atualização de página a cada N segundos (opcional)
tempo_atualizacao = 30  # Atualizar a cada 30 segundos
time.sleep(tempo_atualizacao)
st.rerun()
















