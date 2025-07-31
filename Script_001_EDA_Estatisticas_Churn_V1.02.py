import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px

# Carregamento dos dados
df = pd.read_csv("Customer_Churn_Customer_Churn.csv", sep=",", index_col=False)

# Remover espaços em branco nas colunas
df.columns = df.columns.str.strip()

# Features (Características)/ Atributos.
df.info()
df.head()
print("Tamanho da Base:", df.shape)
# Tamanho das bases: (7043, 21)
# 🧠 Visão geral nos permite entender:
# A base tem volume suficiente para
# modelagem estatística e machine learning
# Possui variáveis diversas
# (categóricas, numéricas, binárias)
# que permitem análises exploratórias ricas

# A base
df.describe()

# SeniorCitizen	tenure
# count	7043.000000	7043.000000
# mean	0.162147	32.371149
# std	0.368612	24.559481
# min	0.000000	0.000000
# 25%	0.000000	9.000000
# 50%	0.000000	29.000000
# 75%	0.000000	55.000000
# max	1.000000	72.000000

# Variável de Interesse
df["Churn"].head()
# Ver quais valores únicos existem na coluna "Churn"
print(df["Churn"].unique())
# ['No' 'Yes']

df["Churn"].describe()
# count     7043
# unique       2
# top         No
# freq      5174
# Name: Churn, dtype: object
# 🔍 Insight direto:
# 5174 clientes (≈ 73.5%) permaneceram
# 1869 clientes (≈ 26.5%) cancelaram
# 💡 A base tem um desequilíbrio moderado,
# o que justifica o uso de F1-score e balanceamento no
# modelo de machine learning.

# Agrupar os dados pela coluna "Churn"
grupo = df.groupby("Churn")

# Contar quantas entradas existem em cada grupo
# Clientes por categoria de churn, via group by
print(grupo.size())
# Churn
# No     5174
# Yes    1869
# dtype: int64

# Obter a quantidade (número de registros) em cada grupo
quantidade = grupo.size()

# Calcular a porcentagem de cada grupo
percentual = (quantidade / len(df)) * 100

# Juntar quantidade e percentual em um único DataFrame
resultado = pd.DataFrame(
    {
        "Quantidade": quantidade,
        "Percentual (%)": percentual.round(2),  # Arredonda para 2 casas decimais
    }
)

print(resultado)
# Quantidade  Percentual (%)
# Churn
# No           5174           73.46
# Yes          1869           26.54


# Calcular a média das colunas numéricas para cada
# grupo de churn
# Médias das colunas numéricas por grupo
print(grupo.mean(numeric_only=True))
# SeniorCitizen     tenure
# Churn
# No          0.128721  37.569965
# Yes         0.254682  17.979133

# 💡 Insights:
# Idosos e clientes novos têm maior risco de churn.
# Ações de retenção devem focar nos primeiros meses de contrato e em usuários mais velhos.
# Agrupar os dados pela coluna "SeniorCitizen"
df["SeniorCitizen"].unique()
SeniorCitizenGroup = df.groupby("SeniorCitizen")
SeniorCitizenGroup.size()
# SeniorCitizen
# 0    5901
# 1    1142
# dtype: int64

# rereset_index
grupo = df.groupby("Churn").mean(numeric_only=True).reset_index()

# Gráfico 1: Proporção de clientes idosos por Churn
# 25% dos cancelamentos vêm de Idosos
# Ao passo que apenas 13% dos não churn são Idosos
# ✅ Interpretação Gerencial:
# Clientes idosos têm quase o dobro de chance de cancelar o serviço.
# Isso pode indicar dificuldade com tecnologia, sensibilidade a preço ou falta de suporte adequado.

fig1 = px.bar(
    grupo,
    x="Churn",
    y="SeniorCitizen",
    title="Proporção de Clientes Idosos por Churn",
    labels={"SeniorCitizen": "Proporção de Idosos"},
    text=grupo["SeniorCitizen"].round(2),
    color="Churn",
)
fig1.update_traces(textposition="outside")
fig1.update_layout(yaxis_tickformat=".0%")  # formato percentual
fig1.show()

# Tempo médio de permanência por Churn
fig2 = px.bar(
    grupo,
    x="Churn",
    y="tenure",
    title="Tempo Médio de Permanência por Churn",
    labels={"tenure": "Tempo médio (meses)"},
    text=grupo["tenure"].round(1),
    color="Churn",
)
fig2.update_traces(textposition="outside")
fig2.show()

# ver TotalCharges
df["TotalCharges"].describe()
df[df["TotalCharges"].isna()]
# 11 rows × 21 columns -- valores ausentes
# count     7032
# unique    6530
# top       20,2
# freq        11
# Name: TotalCharges, dtype: object

df["MonthlyCharges"].describe()
df[df["MonthlyCharges"].isna()]
# 0 rows × 21 columns -- valores ausentes
# count      7043
# unique     1585
# top       20,05
# freq         61
# Name: MonthlyCharges, dtype: object

# 📊 Visão Geral dos Dados
# A análise inicial do dataset nos revelou os seguintes pontos importantes:
# 	•	Total de registros: 7.043 clientes
# 	•	Variável alvo (Churn):
# 	◦	Clientes que não cancelaram: 5.174 (≈ 73.5%)
# 	◦	Clientes que cancelaram: 1.869 (≈ 26.5%)
# 	•	Colunas com problemas:
# 	◦	TotalCharges possui 11 valores ausentes
# 	◦	MonthlyCharges e TotalCharges vieram como texto.

# Variável Está com Valores Categoricos
# Conversão da variável alvo para binária
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})


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

# Copia dos dados Tratados
df2 = df.copy()
# Guardar Dados Tratados
df2.to_csv("Customer_Churn_Customer_Churn_Processadas.csv", index=False)
# Remover linhas com valores ausentes
df.dropna(inplace=True)

df2.head(5)
df2.info()
##########################
# Selecionar colunas categóricas (exceto 'customerID' que é ID única)
cat_cols = df2.select_dtypes(include="object").columns.drop("customerID")

# Para cada variável categórica, Agrupamos via groupby com Churn
for col in cat_cols:
    print(f"\n--- Distribuição da variável '{col}' por Churn ---")
    print(
        df.groupby(["Churn", col])
        .size()
        .groupby(level=0)
        .apply(lambda x: 100 * x / x.sum())
        .round(2)
        .unstack()
        .fillna(0)
    )
# --- Distribuição da variável 'gender' por Churn ---
# gender       Female   Male
# Churn Churn
# 0     0       49.27  50.73
# 1     1       50.24  49.76
# ➡️ Interpretação:
# Gênero não influencia o churn — homens e mulheres
# cancelam em proporções praticamente iguais.

# --- Distribuição da variável 'Partner' por Churn ---
# Partner         No    Yes
# Churn Churn
# 0     0      47.24  52.76
# 1     1      64.21  35.79
# Clientes sem parceiro têm maior probabilidade de
# cancelar. Talvez estejam mais propensos a cortar custos
# por viverem sozinhos.

# --- Distribuição da variável 'Dependents' por Churn ---
# Dependents      No    Yes
# Churn Churn
# 0     0      65.66  34.34
# 1     1      82.56  17.44
# Ter dependentes reduz o risco de churn, talvez por
# estabilidade financeira/familiar.

# --- Distribuição da variável 'PhoneService' por Churn ---
# PhoneService    No    Yes
# Churn Churn
# 0     0       9.88  90.12
# 1     1       9.10  90.90
# Clientes com contrato mensal são os que mais cancelam!
# Planos de longo prazo (anual ou bienal) reduzem churn drasticamente.
# --- Distribuição da variável 'MultipleLines' por Churn ---
# MultipleLines     No  No phone service    Yes
# Churn Churn
# 0     0        49.12              9.88  41.00
# 1     1        45.43              9.10  45.48
#
# --- Distribuição da variável 'InternetService' por Churn ---
# InternetService    DSL  Fiber optic     No
# Churn Churn
# 0     0          37.90        34.84  27.25
# 1     1          24.56        69.40   6.05
#
# --- Distribuição da variável 'OnlineSecurity' por Churn ---
# OnlineSecurity     No  No internet service    Yes
# Churn Churn
# 0     0         39.43                27.25  33.31
# 1     1         78.17                 6.05  15.78
#
# --- Distribuição da variável 'OnlineBackup' por Churn ---
# OnlineBackup     No  No internet service    Yes
# Churn Churn
# 0     0       35.91                27.25  36.84
# 1     1       65.97                 6.05  27.98
#
# --- Distribuição da variável 'DeviceProtection' por Churn ---
# DeviceProtection     No  No internet service    Yes
# Churn Churn
# 0     0           36.47                27.25  36.28
# 1     1           64.79                 6.05  29.16
#
# --- Distribuição da variável 'TechSupport' por Churn ---
# TechSupport     No  No internet service    Yes
# Churn Churn
# 0     0      39.24                27.25  33.51
# 1     1      77.37                 6.05  16.59
#
# --- Distribuição da variável 'StreamingTV' por Churn ---
# StreamingTV     No  No internet service    Yes
# Churn Churn
# 0     0      36.16                27.25  36.59
# 1     1      50.40                 6.05  43.55
#
# --- Distribuição da variável 'StreamingMovies' por Churn ---
# StreamingMovies     No  No internet service    Yes
# Churn Churn
# 0     0          35.70                27.25  37.05
# 1     1          50.19                 6.05  43.77
#
# --- Distribuição da variável 'Contract' por Churn ---
# Contract     Month-to-month  One year  Two year
# Churn Churn
# 0     0               43.00     25.30     31.71
# 1     1               88.55      8.88      2.57
#
# --- Distribuição da variável 'PaperlessBilling' por Churn ---
# PaperlessBilling     No    Yes
# Churn Churn
# 0     0           46.39  53.61
# 1     1           25.09  74.91
#
# --- Distribuição da variável 'PaymentMethod' por Churn ---
# PaymentMethod  Bank transfer (automatic)  Credit card (automatic)  \
# Churn Churn
# 0     0                            24.87                    24.97
# 1     1                            13.80                    12.41
#
# PaymentMethod  Electronic check  Mailed check
# Churn Churn
# 0     0                   25.06         25.10
# 1     1                   57.30         16.48

# 📊 RESUMO GERENCIAL:
# Variável	Ponto de Atenção / Ação
# Contrato mensal	- Incentivar upgrade para anual com bônus/desconto
# Sem parceiro	- Oferecer planos personalizados (ex: individual econômico)
# Sem dependentes	- Foco em clientes jovens ou sozinhos com ofertas práticas
# Pagamento manual	- Incentivar débito automático
# Sem segurança/suporte	- Oferecer serviço grátis no início do contrato
# Fibra óptica	- Avaliar qualidade ou precificação

# Churn por tipo de contrato, via grafico
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn por Tipo de Contrato")
plt.ylabel("Número de Clientes")
plt.xlabel("Tipo de Contrato")
plt.tight_layout()
plt.savefig("churn_por_contrato.png")
# plt.close()

# Churn por tempo de permanência (tenure) v2
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Churn por Tempo de Permanência")
plt.xlabel("Meses de Permanência")
plt.ylabel("Número de Clientes")
plt.tight_layout()
plt.savefig("churn_por_tenure.png")
plt.close()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px

# Codificação de variáveis categóricas
# Codificar dados categóricos em formato numérico
# Preparar os dados para análise estatística / modelagem
df_encoded = pd.get_dummies(df.drop(columns=["customerID"]), drop_first=True)

# Separação entre variáveis independentes e alvo
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Divisão em treino e teste
# ✅ 20% dos dados serão usados para o conjunto de teste, e os 80% restantes para treino.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modelo Random Forest
# Modelo de classificação supervisionada
# o algoritmo Random Forest é ideal para resolver
# problemas de classificação binária ou multiclasse
# com boa performance e interpretabilidade.
# A semente aleatória (random_state=42) garante que os
# resultados sejam consistentes e reprodutíveis.
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
# Relatório de Classificação:
#               precision    recall  f1-score   support
#
#           0       0.82      0.88      0.85      1027
#           1       0.59      0.47      0.53       380
#
#    accuracy                           0.77      1407
#   macro avg       0.70      0.68      0.69      1407
# weighted avg       0.76      0.77      0.76      1407
# Conclusão executiva:
# Embora o modelo tenha bom desempenho geral (77% de
# acurácia), ele ainda falha em prever
# corretamente os clientes que cancelam (classe mais
# crítica para o negócio).
# A baixa revocação para churn (47%) indica que ações de
# melhoria são necessárias
# no modelo.


# Dados do classification_report obtidos
metrics = {
    "Class": ["Classe 0", "Classe 1", "Macro Avg", "Weighted Avg"],
    "Precision": [0.82, 0.59, 0.70, 0.76],
    "Recall": [0.88, 0.47, 0.68, 0.77],
    "F1-Score": [0.85, 0.53, 0.69, 0.76],
}

df_metrics = pd.DataFrame(metrics)

# formato long (melt) para usar no px.bar
df_long = df_metrics.melt(id_vars="Class", var_name="Métrica", value_name="Valor")

# Gráfico de barras - plotly express
fig = px.bar(
    df_long,
    x="Class",
    y="Valor",
    color="Métrica",
    barmode="group",
    title="Comparação de Métricas por Classe (Precision, Recall, F1-Score)",
    labels={"Class": "Classe", "Valor": "Valor da Métrica"},
)

fig.update_layout(yaxis=dict(range=[0, 1]))  # Limitar o eixo Y de 0 a 1
fig.show()

# 📌 Interpretação visual:
# A Classe 0 tem desempenho bem superior em todas as
# métricas — o modelo prevê melhor quem não cancela.
# A Classe 1, que é o foco gerencial (prever quem cancela),
# tem métricas mais baixas, especialmente no recall.
# A macro média reflete a média geral entre as duas classes,
# enquanto a média ponderada (weighted)
# se aproxima mais da Classe 0, porque ela tem mais exemplos.

print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred))
# Matriz de Confusão:
# [[902 125]
# [200 180]]
# Previsto:
# Real: Não	902 (Verdadeiro Negativo)	125 (Falso Positivo)
# Real: Sim	200 (Falso Negativo)	180 (Verdadeiro Positivo)

# Matriz de confusão
z = [[902, 125], [200, 180]]

# Etiquetas
x_labels = ["Previsto: Não", "Previsto: Sim"]
y_labels = ["Real: Não", "Real: Sim"]

# Criar heatmap com Plotly
fig = ff.create_annotated_heatmap(
    z=z,
    x=x_labels,
    y=y_labels,
    colorscale="Blues",
    showscale=True,
    annotation_text=[[str(val) for val in row] for row in z],
)

fig.update_layout(
    title="Matriz de Confusão - Previsão de Churn",
    xaxis_title="Previsão do Modelo",
    yaxis_title="Valor Real",
    yaxis_autorange="reversed",
)

fig.show()

# 📌 Matriz Confusão gerencial:
# Métrica	Valor	Significado
# Verdadeiros Negativos (902)✅Clientes que realmente
# ficaram e o modelo previu corretamente
# Falsos Positivos (125)⚠️Clientes que ficaram,
# mas o modelo previu churn por engano — risco de
# gastar com retenção desnecessária
# Falsos Negativos (200)❌Clientes que cancelaram,
# mas o modelo não identificou — custo direto de perda real
# Verdadeiros Positivos (180)✅Clientes que cancelaram,
# e o modelo previu corretamente — útil para ações de
# retenção

# 🧠 Insight:
# O modelo acerta a maioria dos que não cancelam,
# mas ainda falha bastante em detectar os que realmente
# cancelam (200 erros).
# Do ponto de vista de retorno financeiro, os falsos
# negativos (clientes perdidos) são os mais críticos.

# 🗺️ Roadmap de Entregáveis
# ✅ Fase 1 – Diagnóstico e Exploração (Concluído)
# ✔️ Entendimento do problema de negócio
# ✔️ Limpeza e preparação dos dados
# ✔️ Análise exploratória (EDA)
# ✔️ Identificação de padrões de churn
# ✔️ Construção de modelo preditivo (Random Forest)

# 🔄 Fase 2 – Aprofundamento e Otimização (Próximos passos)
# 🔍 Testar outros modelos (Logistic Regression, XGBoost, LightGBM)
# ⚖️ Aplicar técnicas de balanceamento de classes (SMOTE, undersampling)
# 🧪 Validação cruzada e ajuste de hiperparâmetros
# 📊 Análise de importância das variáveis
# 🚀 Fase 3 – Ações e Implementação
# 📈 Geração de score de churn para cada cliente
# 🛠️ Integração com sistemas de CRM para alertas automáticos
# 🎯 Campanhas de retenção personalizadas com base no perfil de risco
# 📅 Fase 4 – Monitoramento e Evolução
# 📉 Acompanhamento contínuo da taxa de churn
# 🔁 Reentrenamento periódico do modelo
# 📥 Coleta de feedback das ações aplicadas

# 🧩 Ações Recomendadas com Base nos Insights
# 1. Foco nos Primeiros Meses
# Problema: Alta taxa de churn nos primeiros 6 meses
# Ação: Criar programa de onboarding com suporte técnico proativo e benefícios nos 3 primeiros meses
# Impacto estimado: Redução de até 15% no churn inicial
# 2. Incentivo à Fidelização
# Problema: Contratos mensais têm maior churn
# Ação: Oferecer descontos progressivos ou bônus para migração para contratos anuais/bienais
# Impacto estimado: Aumento de 10% na base de clientes fidelizados
# 3. Campanhas Baseadas em Score de Risco
# Problema: Falta de ação preventiva
# Ação: Usar o modelo preditivo para gerar alertas de churn e acionar campanhas personalizadas
# Impacto estimado: Redução de até 20% no churn entre clientes de alto risco
# 4. Monitoramento de Serviços Críticos
# Problema: Clientes sem suporte técnico ou backup têm maior propensão ao churn
# Ação: Oferecer pacotes com esses serviços incluídos como padrão ou em promoção
# Impacto estimado: Aumento da percepção de valor e retenção

# 📈 Métricas Recomendadas para Monitoramento Contínuo
# Para acompanhar a eficácia das ações e a evolução do churn, estas são as principais métricas a serem monitoradas:

# 🔹 Métricas de Negócio
# Taxa de Churn Mensal e Anual
# Tempo Médio de Permanência (Tenure)
# Receita Média por Cliente
# ROI das Campanhas de Retenção
# 🔹 Métricas do Modelo Preditivo
# Acurácia
# Precisão e Recall (especialmente para churn)
# F1-score

# 🔹 Métricas Operacionais
# Taxa de Aderência às Campanhas
# Tempo de Resposta às Ações de Retenção
# Volume de Clientes em Risco Alto vs. Ações Executadas
