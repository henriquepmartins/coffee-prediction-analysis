# 📊 Próximos Passos: Modelo de Clusterização

## 📋 Contexto do Projeto

Este projeto visa prever e agrupar padrões de consumo de café baseado em características temporais, climáticas e de pagamento. Atualmente, temos:

### ✅ **O que já foi feito:**

1. **Análise Exploratória de Dados (EDA)**
   - Notebook: `notebooks/01_exploratory_data_analysis.ipynb`
   - Limpeza e preparação dos dados
   - Criação de features temporais e climáticas
   - Exportação para `data/coffee_cleaned.csv`

2. **Modelo de Classificação**
   - Notebook: `notebooks/02_model_training.ipynb`
   - Classificação binária: "Cafe Preto" vs "Leite e Doces"
   - XGBoost com otimização de hiperparâmetros
   - Accuracy: ~67% | Balanced Accuracy: ~58%
   - Features utilizadas: tempo, clima, período do dia, interações

---

## 🎯 Objetivo da Clusterização

Identificar **padrões de comportamento** nos clientes/compras sem usar labels pré-definidos. A clusterização deve revelar:

- **Grupos de clientes** com comportamentos similares
- **Padrões temporais** de consumo (horários, dias da semana)
- **Perfis de preferência** (café puro vs bebidas com leite)
- **Segmentação** para estratégias de marketing personalizadas

---

## 📁 Estrutura do Projeto

```
coffee-prediction/
├── data/
│   └── coffee_cleaned.csv          # Dataset limpo e processado
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb  # EDA completo
│   ├── 02_model_training.ipynb             # Classificação (concluído)
│   └── 03_clustering.ipynb                  # ⚠️ A CRIAR
└── coffee-venv/                    # Ambiente virtual Python
```

---

## 📊 Dados Disponíveis

### **Dataset:** `data/coffee_cleaned.csv`

**Colunas principais:**
- `datetime`: Data e hora da compra
- `coffee_name`: Nome do café (8 tipos)
- `hour_float_30min`: Hora em formato decimal (0-24)
- `day_of_week`: Dia da semana (0=Segunda, 6=Domingo)
- `month`: Mês (1-12)
- `weather`: Condição climática (sol, chuva, nublado, frio)
- `cash_type`: Tipo de pagamento

**Features já criadas (disponíveis no EDA):**
- `hour_sin`, `hour_cos`: Encoding cíclico da hora
- `is_weekend`: Binário (fim de semana)
- `is_morning`, `is_afternoon`, `is_evening`: Períodos do dia
- `morning_cold`, `evening_cold`, `weekend_morning`: Interações

**Total de registros:** ~3,638 transações

---

## 🔧 Tarefas para o Modelo de Clusterização

### **1. Preparação dos Dados**

```python
# Carregar dados
df = pd.read_csv("data/coffee_cleaned.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Decidir quais features usar para clusterização
# Opções:
# A) Features temporais + clima (sem target)
# B) Features + comportamento de compra (frequência, valor médio)
# C) Features + histórico de preferências
```

**Decisões necessárias:**
- [ ] Incluir ou excluir `coffee_name` como feature?
- [ ] Criar features agregadas por cliente? (se houver ID de cliente)
- [ ] Normalizar/escalar features numéricas
- [ ] Tratar features categóricas (OneHotEncoder ou LabelEncoder)

---

### **2. Escolha do Algoritmo**

**Opções recomendadas:**

#### **A) K-Means** (Mais simples)
- ✅ Rápido e interpretável
- ✅ Bom para dados numéricos
- ⚠️ Requer número de clusters pré-definido
- ⚠️ Sensível a outliers

#### **B) DBSCAN** (Densidade)
- ✅ Não precisa definir número de clusters
- ✅ Identifica outliers automaticamente
- ⚠️ Mais complexo de ajustar (eps, min_samples)

#### **C) Hierarchical Clustering** (Agrupamento hierárquico)
- ✅ Visualização com dendrograma
- ✅ Não precisa definir K inicialmente
- ⚠️ Computacionalmente caro para datasets grandes

#### **D) Gaussian Mixture Models (GMM)**
- ✅ Probabilístico (soft clustering)
- ✅ Lida bem com clusters de formas diferentes
- ⚠️ Mais complexo

**Recomendação inicial:** Começar com **K-Means** e depois testar **DBSCAN** se necessário.

---

### **3. Determinação do Número de Clusters**

**Métricas a usar:**

1. **Elbow Method** (Método do Cotovelo)
   - Plotar inércia vs número de clusters
   - Identificar o "cotovelo" no gráfico

2. **Silhouette Score**
   - Mede quão bem separados estão os clusters
   - Valores entre -1 e 1 (quanto maior, melhor)
   - Plotar silhouette score vs número de clusters

3. **Gap Statistic**
   - Compara inércia real vs inércia esperada
   - Mais robusto que Elbow Method

**Código exemplo:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Testar diferentes valores de K
k_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plotar resultados
```

---

### **4. Visualização dos Clusters**

**Gráficos essenciais:**

1. **PCA/T-SNE para redução de dimensionalidade**
   ```python
   from sklearn.decomposition import PCA
   from sklearn.manifold import TSNE
   
   # Reduzir para 2D para visualização
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_scaled)
   
   # Plotar clusters em 2D
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
   ```

2. **Análise de características por cluster**
   - Boxplots das features mais importantes por cluster
   - Heatmap de médias de features por cluster
   - Distribuição de `coffee_name` por cluster

3. **Análise temporal**
   - Distribuição de horários por cluster
   - Distribuição de dias da semana por cluster
   - Padrões de clima por cluster

---

### **5. Interpretação e Validação**

**Perguntas a responder:**

- [ ] Cada cluster representa um perfil distinto de cliente?
- [ ] Os clusters fazem sentido do ponto de vista de negócio?
- [ ] Há clusters que são claramente "Cafe Preto" vs "Leite e Doces"?
- [ ] Existem padrões temporais específicos por cluster?
- [ ] Os clusters são estáveis? (testar com diferentes seeds)

**Validação:**
- Comparar clusters com labels conhecidos (se disponível)
- Análise de features mais discriminantes por cluster
- Teste de estabilidade (rodar múltiplas vezes com diferentes seeds)

---

## 📝 Estrutura Sugerida do Notebook `03_clustering.ipynb`

### **Célula 1: Imports**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
```

### **Célula 2: Carregamento e Preparação**
- Carregar `coffee_cleaned.csv`
- Selecionar features para clusterização
- Escalar/normalizar dados

### **Célula 3: Análise Exploratória**
- Estatísticas descritivas
- Correlações entre features
- Visualizações iniciais

### **Célula 4: Determinação do Número de Clusters**
- Elbow Method
- Silhouette Analysis
- Decisão do K ótimo

### **Célula 5: Treinamento do Modelo**
- K-Means (ou outro algoritmo escolhido)
- Ajuste de hiperparâmetros

### **Célula 6: Visualização dos Clusters**
- PCA/T-SNE 2D
- Análise de características por cluster
- Gráficos de distribuição

### **Célula 7: Interpretação**
- Perfis de cada cluster
- Análise de negócio
- Insights e recomendações

---

## 🎯 Objetivos de Negócio

A clusterização deve ajudar a responder:

1. **Segmentação de Clientes**
   - Quais são os principais perfis de consumidores?
   - Como personalizar ofertas para cada segmento?

2. **Otimização de Operações**
   - Quais horários têm padrões similares?
   - Como preparar estoque baseado em clusters?

3. **Marketing**
   - Quais clusters respondem melhor a promoções?
   - Como criar campanhas segmentadas?

---

## 🔗 Referências Úteis

- **Scikit-learn Clustering:** https://scikit-learn.org/stable/modules/clustering.html
- **K-Means:** https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- **DBSCAN:** https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
- **Silhouette Analysis:** https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

---

## ⚠️ Pontos de Atenção

1. **Escalonamento:** Sempre escalar features numéricas antes de clusterizar (StandardScaler)
2. **Features Categóricas:** Decidir se usar OneHotEncoder ou outra estratégia
3. **Dimensionalidade:** Considerar PCA se houver muitas features
4. **Interpretabilidade:** Priorizar clusters que façam sentido de negócio
5. **Validação:** Testar estabilidade dos clusters com diferentes seeds

---

## 📌 Checklist para Iniciar

- [ ] Ler este documento completamente
- [ ] Revisar `notebooks/01_exploratory_data_analysis.ipynb` para entender os dados
- [ ] Revisar `notebooks/02_model_training.ipynb` para ver features criadas
- [ ] Carregar `data/coffee_cleaned.csv` e explorar estrutura
- [ ] Decidir quais features usar para clusterização
- [ ] Criar notebook `03_clustering.ipynb`
- [ ] Começar com K-Means e Elbow Method
- [ ] Validar resultados com Silhouette Score
- [ ] Visualizar e interpretar clusters

---

**Boa sorte com a clusterização! 🚀☕**

*Última atualização: Baseado no estado do projeto após conclusão do modelo de classificação.*

