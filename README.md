Projeto de Mineração de Dados: Conversion Rate

Este projeto compara o desempenho de diferentes algoritmos de Machine Learning para problemas de regressão, utilizando um dataset que inclui métricas de marketing digital (impressões, cliques, gastos) e características demográficas dos usuários.

Esse projeto tem como objetivo avaliar e comparar a performance de 5 modelos de regressão distintos para identificar o melhor algoritmo para previsão da variável alvo, considerando diferentes métricas de avaliação.
---
Modelos Implementados
1. Ridge Regression 
Modelo linear com regularização L2


2. Decision Tree 
Árvore de decisão única

3. Random Forest 
Ensemble de múltiplas árvores (bagging)

4. MLPRegressor 
Rede Neural Artificial (Multi-Layer Perceptron)

5. Gradient Boosting 
Ensemble sequencial com boosting

---

Métricas de Avaliação
Cada modelo foi avaliado usando 4 métricas principais:

```python
Métrica	|        Descrição            |	Ideal
--------|-----------------------------|-------------------------------------------
MSE	    |   Mean Squared Error	      |  Quanto menor, melhor
RMSE    |	 Root Mean Squared Error  |	 Quanto menor, melhor
MAE     |	  Mean Absolute Error	  |  Quanto menor, melhor
R²	    | Coeficiente de Determinação |  Quanto mais próximo de 1, melhor
--------|-----------------------------|-------------------------------------------
```
Técnicas Aplicadas

Pré-processamento:

Padronização com StandardScaler() para modelos sensíveis à escala;

Codificação com OneHotEncoder() para variáveis categóricas;

Divisão dos dados: 70% treino, 30% teste.

---
Otimização:

Grid Search com validação cruzada (5 folds);

K-Fold Cross Validation para avaliação robusta;

Paralelização com n_jobs=-1 para eficiência.

---
Feature Engineering:

Análise de importância de features para cada modelo;

Seleção de hiperparâmetros ótimos.

---
Resultados Finais

Ranking dos Modelos (ordenado por RMSE):

```python
Posição |   	Modelo     |	 MSE    |  RMSE   |   	MAE   |  	R²   |
--------|------------------|------------|---------|-----------|----------|--------------
   1º	|  MLPRegressor    |	4.8214	| 2.1958  |  1.2538	  |  0.8421  |
   2º   | 	  Ridge        |	7.2819	| 2.6985  |  1.3862	  |  0.7616  |
--------|------------------|------------|---------|-----------|----------|--------------
```
