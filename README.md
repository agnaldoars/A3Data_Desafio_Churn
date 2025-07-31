# 🔍 Projeto Churn Prediction - A3Data Desafio Técnico

Este projeto tem como objetivo analisar e prever o cancelamento de clientes (churn) de uma empresa de 
telecomunicações, com base em um desafio técnico proposto pela A3Data.

---

## 📁 Estrutura do Projeto

```
📦 churn-prediction-a3data
├── 📊 Script_001_EDA_Estatisticas_Churn_V1.02.py     # Estatísticas descritivas e gráficos de churn
├── 🤖 Script_001_EDA_Modelos_IA_Churn_V1.02.py       # Modelagem preditiva com RandomForest, XGBoost, etc.
├── 📈 Apresentacao_Churn_A3Data.pptx                 # Slides de apresentação gerencial com insights e roadmap
├── 📄 requirements.txt                               # Bibliotecas necessárias
└── 📝 README.md                                      # Este arquivo
```

---

## 🚀 Tecnologias Utilizadas

- Python 3.11
- Pandas, NumPy
- Scikit-Learn
- XGBoost, LightGBM
- Seaborn, Matplotlib, Plotly
- Python-PPTX (para geração de slides)

---

## 🧠 Modelos Avaliados, com cross validation

| Modelo              | Tipo                      | F1-score médio | Observação                            |
|---------------------|---------------------------|----------------|----------------------------------------|
| RandomForest        | Ensemble de Árvores       | 0.6382         | Melhor desempenho geral                |
| LogisticRegression  | Regressão Linear Binária  | 0.6291         | Interpretação simples                  |
| XGBoost             | Gradient Boosting         | 0.6287         | Potencial de ajuste fino com tuning    |

---

## 📌 Principais Insights

- **88,6% dos cancelamentos** vêm de clientes com **contrato mensal**
- Clientes **idosos** têm quase o **dobro de chance de churn**
- Ausência de **segurança online** e **pagamentos manuais** elevam o risco

## ✅ Hipóteses Levantadas com Base nos Dados de Churn
 - Clientes com contrato mensal estão mais propensos ao churn
 - 🔍 Justificativa: 88,6% dos cancelamentos vêm de contratos do tipo month-to-month, 
 - o que sugere menor compromisso e maior rotatividade.
 - 
 - Clientes idosos tendem a ser mais estáveis (menor churn)
 - 🔍 Justificativa: Apenas 25% dos churns são de idosos, enquanto 75% vêm de não idosos. 
 - Idosos possivelmente valorizam estabilidade e evitam mudanças frequentes de serviço.
 - 
 - Clientes sem parceiro ou dependentes têm maior chance de churn
 - 🔍 Justificativa: Entre os clientes que cancelaram, há maior proporção de pessoas solteiras ou sem dependentes, 
 - indicando um perfil com menor “vínculo familiar” ao serviço.

---

## 📈 Ações Recomendadas e Estimativas de Impacto
✓ Incentivar contrato anual → Redução esperada de até 30%
✓ Oferecer suporte gratuito no início → Redução esperada de até 15%
✓ Automatizar forma de pagamento → Redução esperada de até 10%

---

## ✅ Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/agnaldoars/A3Data_Desafio_Churn.git
```

2. Instale os requisitos:
```bash
pip install -r requirements.txt
```

3. Rode os scripts Python para análise ou modeling, VSCode com Jupyter:
```bash
python Script_001_EDA_Estatisticas_Churn_V1.02.py
python Script_001_EDA_Modelos_IA_Churn_V1.02.py
```

---

## 📬 Contato

Desenvolvido para o desafio técnico da A3Data por Agnaldo S. Rodrigues.

---
>>>>>>> 865837c (Fim Crurn Project A3Data)
