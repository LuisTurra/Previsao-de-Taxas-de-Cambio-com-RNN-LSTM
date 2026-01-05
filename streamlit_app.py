import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
import requests
import plotly.graph_objects as go
import plotly.io as pio
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

CACHE_FILE = 'data_cache.csv'

@st.cache_data
def get_exchange_data(from_currency, to_currency, interval='daily'):
    try:
        url = f'https://www.alphavantage.co/query?function=FX_{interval.upper()}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}'
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'Time Series FX (Daily)' in data:
            df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df['4. close'].to_csv(CACHE_FILE)  
            return df['4. close']
        else:
            raise ValueError("Erro na API.")
    except Exception as e:
        st.warning(f"API falhou ({e}). Usando cache local se disponível.")
        if os.path.exists(CACHE_FILE):
            df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            return df.iloc[:, 0]  
        else:
            st.error("Sem cache local. Obtenha dados primeiro.")
            return None

# Função pred
def predict_future(model, data, scaler, lookback=60, future_days=30):
    last_sequence = data[-lookback:].values.reshape(-1, 1)
    scaled_last = scaler.transform(last_sequence)
    
    predictions = []
    for _ in range(future_days):
        pred = model.predict(scaled_last.reshape(1, lookback, 1))
        predictions.append(pred[0, 0])
        scaled_last = np.append(scaled_last[1:], pred)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Interface
st.set_page_config(page_title="Previsão de Câmbio", page_icon="📈", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Sobre o Projeto")
    st.write("Desenvolvido por :Luis Henrique Turra Ramos")  
    st.write("Previsão de taxas de câmbio usando RNNs (LSTM/GRU/Conv1D LSTM/SimpleRNN/Bidirectional LSTM). Compara modelos para melhor desempenho.")
    st.write("Dados: Alpha Vantage API com cache local.")
    st.markdown("[GitHub Repo](https://github.com/SEU_USUARIO/previsao-taxas-cambio-lstm)")
    st.markdown("[Alpha Vantage](https://www.alphavantage.co)")

st.title("🪙 Previsão de Taxas de Câmbio com RNN")

col1, col2 = st.columns(2)
with col1:
    from_currency = st.selectbox("Moeda de Origem", ['USD', 'EUR', 'GBP'], help="Escolha a moeda base")
with col2:
    to_currency = st.selectbox("Moeda de Destino", ['BRL', 'EUR', 'JPY'], help="Escolha a moeda alvo")

future_days = st.slider("Dias para Prever", 1, 90, 30, help="Quantos dias no futuro prever")

if st.button("Carregar Dados e Prever", type="primary"):
    data = get_exchange_data(from_currency, to_currency)
    if data is not None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Carregar melhor modelo
        if os.path.exists('modelo_melhor.h5'):
            model = load_model('modelo_melhor.h5')
        else:
            st.error("Modelo não encontrado. Rode train_model.py.")
            
        
        predictions = predict_future(model, data, scaler, future_days=future_days)
        
        last_date = data.index[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)
        
        # Gráfico interativo
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[-180:], y=data[-180:], mode='lines', name='Histórico', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Previsão', line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f'Previsão {from_currency}/{to_currency} (até {future_days} dias)',
            xaxis_title='Data', yaxis_title='Taxa de Câmbio',
            template='plotly_dark', hovermode='x unified',
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas
        with st.expander("Detalhes Técnicos"):
            st.write(f"Última taxa real: {data[-1]:.4f}")
            st.write(f"Previsão para +{future_days} dias: {predictions[-1][0]:.4f}")
            st.write("Modelo usado: O melhor foi GRU baseado em MSE.")

# Seção para gráfico de comparação
with st.expander("Análise de Modelos: Comparação de Desempenho"):
    json_file = 'comparacao_modelos.json'
    png_file = 'comparacao_modelos.png'
    if os.path.exists(json_file):
        fig_comp = pio.read_json(json_file)
        st.plotly_chart(fig_comp, use_container_width=True)
    elif os.path.exists(png_file):
        st.image(png_file, caption='Comparação de MSE dos Modelos', use_column_width=True)
    else:
        st.warning("Gráfico de comparação não encontrado. Rode train_model.py para gerá-lo.")