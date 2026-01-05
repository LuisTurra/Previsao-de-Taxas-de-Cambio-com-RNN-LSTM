# Previsão de Taxas de Câmbio com RNN (LSTM/GRU)

Projeto de Data Science para prever taxas de câmbio usando RNNs, comparando LSTM, GRU, Conv1D LSTM, SimpleRNN e Bidirectional LSTM para melhor desempenho.

## Configuração
1. Crie um `.env` com `ALPHA_VANTAGE_API_KEY="sua_key"` (não commite!).
2. Instale deps: `pip install -r requirements.txt`.
3. Rode `python train_model.py` para treinar/comparar modelos.
4. Rode `streamlit run app.py`.

## Hospedagem no Streamlit Cloud
- Conecte o repo.
- Adicione secret: ALPHA_VANTAGE_API_KEY = "sua_key".
- App em `app.py`.

## LIVE DEMO: https://luisturra-previsao-de-taxas-de-cambio-com--streamlit-app-93pqrb.streamlit.app/

## Segurança
- API Key é carregada via env var, não hardcode!

## Etapas
- Dados via API com fallback local.
- Comparação entre RNNs
- App com visual melhorado.