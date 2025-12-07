import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

API_URL = os.getenv("API_URL", "http://model-api:5000/predict")

st.title("ConversionRate - Predição")


best_model_name = None
best_model_name_path = "/app/models/best_model_name.txt"  

if os.path.exists(best_model_name_path):
    try:
        with open(best_model_name_path, "r") as f:
            best_model_name = f.read().strip()
    except Exception:
        best_model_name = None

st.write("Faça upload de um CSV para executar a predição.")

uploaded_file = st.file_uploader("Upload do arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Pré-visualização dos dados:")
    st.dataframe(df.head())

    if st.button("Gerar Predições"):
        data = {"data": df.to_dict(orient="records")}
        try:
            response = requests.post(API_URL, json=data)
            if response.status_code == 200:
                preds = response.json()["predictions"]
                df_result = df.copy()
                df_result["Predicted_Total_Conversion"] = preds

                st.write("Resultado da predição:")
                st.dataframe(df_result.head())

                # métricas de avaliação 
                if "Total_Conversion" in df_result.columns:
                    y_true = df_result["Total_Conversion"]
                    y_pred = df_result["Predicted_Total_Conversion"]

                    mse  = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae  = mean_absolute_error(y_true, y_pred)
                    r2   = r2_score(y_true, y_pred)

                    st.subheader("Métricas de Avaliação no CSV enviado")
                    if best_model_name:
                        st.write(f"*Modelo usado:* {best_model_name}")
                    st.write(f"*MSE* : {mse:.4f}")
                    st.write(f"*RMSE*: {rmse:.4f}")
                    st.write(f"*MAE* : {mae:.4f}")
                    st.write(f"*R²*  : {r2:.4f}")
                else:
                    st.info("Coluna 'Total_Conversion' não encontrada no CSV. Métricas não calculadas.")

                # baixar resultado
                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Baixar resultados em CSV",
                    data=csv_out,
                    file_name="predictions_conversion_rate.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erro ao chamar a API: {e}")