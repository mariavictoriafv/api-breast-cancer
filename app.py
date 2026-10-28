
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="API de Classificação de Câncer de Mama")

# Carregar o modelo salvo
artefato = joblib.load("modelo_breast.joblib")

modelo = artefato["modelo"]
colunas = artefato["colunas"]


class Dados(BaseModel):
    features: dict[str, float]


@app.post("/predict")
def prever(dados: Dados):

    # Organizar os dados na ordem esperada pelo modelo
    X_novo = pd.DataFrame(
        [[dados.features[coluna] for coluna in colunas]],
        columns=colunas
    )

    previsao = int(modelo.predict(X_novo)[0])
    probabilidades = modelo.predict_proba(X_novo)[0]

    return {
        "classe_prevista": previsao,
        "diagnostico": "maligno" if previsao == 1 else "benigno",
        "probabilidade_benigno": float(probabilidades[0]),
        "probabilidade_maligno": float(probabilidades[1])
    }
