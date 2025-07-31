# ✅ Modelos de Avaliação e Comparação - Cross Validation:
# Models
# * RandomForest
#    Ensemble de Árvores de Decisão (supervisionado)
#    Robusto, evita overfitting, lida bem com dados mistos
# *  LogisticRegression
#    Regressão linear para classificação binária
# 	Interpretação simples e rápida, baseline eficaz
# * XGBoos
#   Gradient Boosting com regularização	Alta performance,
#   controle fino do aprendizado

# 🛠️ Etapas:
# Pré-processamento com get_dummies()
# Aplicação de validação cruzada (StratifiedKFold)
# Avaliação com f1-score (porque o problema é desequilibrado, muitos No e pouco Yes ou 0/1)
# Utilização de melhores hiperparâmetros/recomendados

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
import plotly.express as px

warnings.filterwarnings("ignore")
#
# Base tratada........
df = pd.read_csv("Customer_Churn_Customer_Churn.csv")

# Remover espaços em branco nas colunas
df.columns = df.columns.str.strip()
# Variável Está com Valores Categoricos
# Conversão da variável alvo para binária --- na proxima
# df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


# Substituir vírgulas por pontos nas colunas numéricas e converter para float
df["MonthlyCharges"] = (
    df["MonthlyCharges"].astype(str).str.replace(",", ".").astype(float)
)
df["TotalCharges"] = (
    df["TotalCharges"]
    .astype(str)
    .str.replace(",", ".")
    .replace(" ", pd.NA)
    .astype(float)
)

# Remover linhas com valores ausentes
df.dropna(inplace=True)

# 1. Pré-processamento
df_encoded = pd.get_dummies(df.drop(columns=["customerID"]), drop_first=True)
df_encoded.columns
# Separar features e target
X = df_encoded.drop(
    columns="Churn_Yes"
)  # target binário(Yes/No) virou coluna com get_dummies
y = df_encoded["Churn_Yes"]

# 2. Validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1 = make_scorer(f1_score)

# 3. Modelos com hiperparâmetros recomendados
modelos = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",
        random_state=42,
    ),
    "LogisticRegression": LogisticRegression(
        solver="liblinear", C=1.0, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=2,  # importante para classes desbalanceadas
        random_state=42,
    ),
}

# 4. Avaliar modelos
for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=f1)
    print(
        f"{nome}: F1-score médio = {scores.mean():.4f} | Desvio padrão = {scores.std():.4f}"
    )

# RandomForest: F1-score médio = 0.6382 | Desvio padrão = 0.0091
# LogisticRegression: F1-score médio = 0.6291 | Desvio padrão = 0.0063
# XGBoost: F1-score médio = 0.6287 | Desvio padrão = 0.0121

# Dados dos modelos
dados_modelos = pd.DataFrame(
    {
        "Modelo": ["RandomForest", "LogisticRegression", "XGBoost"],
        "F1_score_medio": [0.6382, 0.6291, 0.6287],
        "Desvio_padrao": [0.0091, 0.0063, 0.0121],
    }
)

# Gráfico de barras (desvio padrão)
fig = px.bar(
    dados_modelos,
    x="Modelo",
    y="F1_score_medio",
    error_y="Desvio_padrao",
    title="Comparação de Modelos - F1-score Médio",
    labels={"F1_score_medio": "F1-score Médio"},
    text=dados_modelos["F1_score_medio"].round(4),
    color="Modelo",
)

fig.update_traces(textposition="outside")
fig.update_layout(yaxis=dict(title="F1-score", range=[0.60, 0.65]))
fig.show()


# 📊 F1 Score
# Combina precisão e recall.
# Ideal para problemas desequilibrados,
# como Churn, onde errar a classe minoritária
# (quem cancela) custa caro.
#
# Um F1-score de 0.63 indica desempenho intermediário —
# o modelo acerta boa parte dos casos, mas ainda há
# espaço para melhorar a detecção dos clientes que vão
# cancelar.
#
# 📌 Justificativa gerencial por modelo:
# 🔷 1. RandomForest – F1: 0.6382
# ✅ Melhor desempenho geral entre os três.
#
# ✅ É um modelo robusto, funciona bem com dados mistos e sem muitos ajustes.
#
# 📊 Boa escolha para implantar rapidamente com confiança e estabilidade.
#
# 🔄 Fácil de explicar para times de negócio usando importância de variáveis.
#
# Indicado para produção, a menos que haja limitações de processamento.
#
# 🔶 2. LogisticRegression – F1: 0.6291
# ⚖️ Quase o mesmo desempenho que RandomForest.
#
# ✅ Mais simples, mais rápido e mais interpretável.
#
# 📈 Ideal para cenários com pouca infraestrutura, ou onde a explicação clara do modelo é essencial.
#
# 🧠 Pode ser útil como modelo base de comparação, ou para análises exploratórias.
#
# 🔸 3. XGBoost – F1: 0.6287
# ⚙️ Modelo de alta performance, ótimo para tunar em problemas complexos.
#
# 🤏 Mais instável (maior desvio padrão) — precisa de ajustes finos (hyperparameter tuning).
#
# 💻 Recomendado quando se busca máximo desempenho e há recursos computacionais disponíveis.
#
# Por ora, seu desempenho não superou os outros — talvez por precisar de mais ajustes.
#
# ✅ Conclusão gerencial:
# Random Forest é o modelo com o melhor equilíbrio entre desempenho e estabilidade, sendo recomendado para uso imediato ou piloto.
# Logistic Regression é uma boa alternativa quando há foco em simplicidade, transparência e custo computacional baixo.
# XGBoost, embora poderoso, ainda não superou os outros — vale a pena explorar ajustes para ver se ele pode render mais.

# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Grade de hiperparâmetros
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10],
    "class_weight": ["balanced"],
}

# Modelo base
rf = RandomForestClassifier(random_state=42)

# Validação estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,  # usa todos os núcleos disponíveis
    verbose=1,
)

# Rodar busca
grid_search.fit(X, y)

# Resultados
print("Melhor F1-score:", grid_search.best_score_)
print("Melhores hiperparâmetros:", grid_search.best_params_)
# Melhor F1-score: 0.63824195126777
# Melhores hiperparâmetros: {'class_weight': 'balanced',
#                               'max_depth': 10,
#                       'min_samples_split': 10,
#                            'n_estimators': 200}
