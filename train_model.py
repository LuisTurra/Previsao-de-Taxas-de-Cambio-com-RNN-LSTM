import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Conv1D, MaxPooling1D # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
import requests 
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio

load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
if not API_KEY:
    raise ValueError("API Key não encontrada. Configure em .env.")

def get_exchange_data(from_currency='USD', to_currency='BRL', interval='daily'):
    url = f'https://www.alphavantage.co/query?function=FX_{interval.upper()}&from_symbol={from_currency}&to_symbol={to_currency}&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()
    
    if 'Time Series FX (Daily)' in data:
        df = pd.DataFrame.from_dict(data['Time Series FX (Daily)'], orient='index')
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df['4. close']
    else:
        raise ValueError("Erro ao obter dados.")

# Parâmetros
from_currency = 'USD'
to_currency = 'BRL'
data = get_exchange_data(from_currency, to_currency)
data.to_csv('data_cache.csv')  

if len(data) < 100:
    print("Aviso: Dataset pequeno (<100 pontos). Considere intervalo maior ou mais dados.")

# Pré-processamento
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Sequências
def create_sequences(data, lookback=60):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback), 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)

lookback = 60
X, y = create_sequences(scaled_data, lookback)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split train/test (80/20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Callbacks para otimização
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),  
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
]

# Função para criar modelos
def create_model(model_type='LSTM'):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
    elif model_type == 'GRU':
        model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(GRU(units=50))
    elif model_type == 'SimpleRNN':
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(SimpleRNN(units=50))
    elif model_type == 'Bidirectional_LSTM':
        model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(X.shape[1], 1)))
        model.add(Bidirectional(LSTM(units=50)))
    elif model_type == 'Conv1D_LSTM':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Modelos a comparar
model_types = ['LSTM', 'GRU', 'SimpleRNN', 'Bidirectional_LSTM', 'Conv1D_LSTM']
models = {typ: create_model(typ) for typ in model_types}
performances = {}

for name, model in models.items():
    print(f"Treinando {name}...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=callbacks)  # Epochs max aumentado
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    performances[name] = mse
    print(f"{name} MSE: {mse}")

# Selecionar melhor
best_model_name = min(performances, key=performances.get)
best_model = models[best_model_name]
best_model.save('modelo_melhor.h5')
print(f"Melhor modelo: {best_model_name} (MSE: {performances[best_model_name]}). Salvo como 'modelo_melhor.h5'.")

# Gráfico de comparação melhorado (ordenado, interativo)
perf_df = pd.DataFrame(list(performances.items()), columns=['Modelo', 'MSE']).sort_values('MSE')
fig = px.bar(perf_df, x='Modelo', y='MSE', title='Comparação de MSE dos Modelos (Menor é Melhor)',
             labels={'MSE': 'Mean Squared Error'}, color='MSE', color_continuous_scale='bluered')
fig.update_layout(xaxis_tickangle=-45, hovermode='x unified', template='plotly_dark')
fig_json = pio.to_json(fig)
with open('comparacao_modelos.json', 'w', encoding='utf-8') as f:
    f.write(fig_json)
fig.write_image('comparacao_modelos.png') 
print("Gráfico salvo como 'comparacao_modelos.json' e 'comparacao_modelos.png'.")