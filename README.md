# análise de predição de café

## resumo

este projeto apresenta uma análise completa de dados de vendas de café utilizando técnicas de machine learning para predizer a categoria de café que um cliente provavelmente escolherá. foram desenvolvidos dois modelos distintos de classificação utilizando xgboost, cada um com abordagens e objetivos diferentes.

o **modelo 1** realiza classificação binária entre duas categorias amplas ("café preto" vs "leite e doces"), alcançando 68% de acurácia. é otimizado através de randomized search cv e utiliza técnicas de balanceamento de dados para lidar com classes desbalanceadas.

o **modelo 2** implementa uma arquitetura hierárquica em cascata para classificação multiclasse entre três categorias específicas (chocolate quente, café com leite, café preto), alcançando 61% de acurácia. utiliza dois modelos xgboost sequenciais que decompõem o problema complexo em decisões binárias mais simples, resultando em melhor performance para categorias minoritárias.

ambos os modelos utilizam características temporais (hora do dia com representação cíclica, dia da semana, mês), condições climáticas, tipo de pagamento e interações entre essas variáveis.

## modelos treinados

### modelo 1: classificação binária

**algoritmo**: xgboost classifier com randomized search cv

**objetivo**: classificar entre "café preto" e "leite e doces"

**função prática**: este modelo foi desenvolvido para predizer a categoria geral de café que um cliente provavelmente escolherá com base em características contextuais da venda. agrupa os tipos de café em duas categorias amplas: "café preto" (americano, espresso, cortado) e "leite e doces" (latte, cappuccino, chocolate quente, etc.). é útil para análises de padrões de consumo e planejamento de estoque em nível de categoria.

**como funciona**: o modelo recebe como entrada características temporais (hora do dia, dia da semana, mês), condições climáticas e tipo de pagamento. através de um pipeline de preprocessamento, as features categóricas são codificadas com one-hot encoding e as numéricas são normalizadas com standard scaler. o xgboost então aprende padrões complexos entre essas características e a escolha do cliente, utilizando uma estrutura de árvores de decisão otimizada via randomized search cv.

**processo de predição**:
1. recebe features de contexto (hora, clima, pagamento, etc.)
2. aplica transformações de preprocessamento
3. passa pelos modelos xgboost treinados
4. retorna probabilidade e classe predita (café preto ou leite e doces)

**características técnicas**:

- pré processamento: one-hot encoding (categóricas) e standard scaler (numéricas)
- balanceamento: resampling para lidar com desbalanceamento (classe minoritária aumentada para 67% da classe majoritária)
- otimização: randomized search cv com 15 iterações e validação cruzada de 5 folds
- métrica de otimização: balanced_accuracy para lidar com classes desbalanceadas

**métricas**:

- accuracy: 68%
- precision (café preto): 43%
- precision (leite e doces): 78%
- recall (café preto): 43%
- recall (leite e doces): 78%

**features principais** (14 features totais):

- hora do dia: representação cíclica usando seno e cosseno para capturar padrões temporais (hour_sin, hour_cos, hour_float_30min)
- período do dia: flags binárias para manhã (6h-11h), tarde (12h-17h) e noite (18h+)
- características temporais: dia da semana, mês, flag de fim de semana
- condições climáticas: variável categórica com one-hot encoding
- tipo de pagamento: variável categórica
- interações: combinações como "manhã + frio", "noite + frio", "fim de semana + manhã"

### modelo 2: classificação hierárquica

**algoritmo**: dois modelos xgboost em cascata

**objetivo**: classificar entre três categorias específicas (chocolate quente, café com leite, café preto)

**função prática**: este modelo foi desenvolvido para predições mais granulares, diferenciando três categorias distintas de café. é especialmente útil para sistemas de recomendação personalizados e análises detalhadas de preferências do cliente. a abordagem hierárquica permite que o modelo se concentre primeiro em identificar a categoria mais distinta (chocolate quente) antes de diferenciar entre os tipos de café com leite.

**arquitetura em cascata**:

1. **primeiro nível (modelo_choc)**: classifica se a venda é de "chocolate quente" ou não
   - se predito como chocolate quente → retorna "chocolate quente"
   - se predito como não-chocolate → passa para o segundo nível

2. **segundo nível (modelo_milk)**: classifica entre "café com leite" e "café preto" (apenas para casos não-chocolate)
   - recebe apenas os casos onde o primeiro modelo indicou "não-chocolate"
   - diferencia entre café com leite (latte, cappuccino, cortado, americano com leite) e café preto (americano, espresso)

**como funciona**: o modelo utiliza uma estratégia de decomposição hierárquica do problema multiclasse. em vez de treinar um único classificador para três classes, divide o problema em dois classificadores binários sequenciais. isso permite que cada modelo se especialize em uma decisão mais simples e focada, melhorando a capacidade de capturar padrões específicos de cada categoria.

**processo de predição**:
1. recebe features de contexto (hora, clima, histórico do cartão, etc.)
2. primeiro modelo (modelo_choc) calcula probabilidade de ser chocolate quente
3. se probabilidade alta de chocolate → retorna "chocolate quente"
4. caso contrário, segundo modelo (modelo_milk) calcula probabilidade de café com leite vs café preto
5. probabilidades finais são combinadas: p(chocolate), p(café com leite) = p(não-chocolate) × p(café com leite|não-chocolate), p(café preto) = p(não-chocolate) × p(café preto|não-chocolate)
6. retorna a categoria com maior probabilidade

**características técnicas**:

- balanceamento: sample weights calculados com `compute_sample_weight(class_weight='balanced')` para lidar com classes desbalanceadas
- parâmetros comuns aos dois modelos: max_depth=4, learning_rate=0.05, n_estimators=300, subsample=0.9, colsample_bytree=0.9
- features adicionais: inclui histórico de preferências por cartão (card_choc_rate, card_milk_rate, card_black_rate) calculado a partir do conjunto de treino
- métrica de avaliação: logloss para ambos os modelos

**métricas** (no conjunto de teste):

- accuracy: 61%
- precision (café preto): 47%
- precision (café com leite): 87%
- precision (chocolate quente): 33%
- recall (café preto): 66%
- recall (café com leite): 60%
- recall (chocolate quente): 62%

**vantagens da abordagem hierárquica**:

- simplifica o problema multiclasse em dois problemas binários mais simples
- melhor performance em categorias minoritárias (chocolate quente tem recall de 62% vs 43% no modelo binário para café preto)
- permite especialização: cada modelo foca em uma decisão específica
- mais interpretável: é possível entender qual nível da hierarquia está influenciando a decisão
- flexível: permite ajustar cada modelo independentemente conforme necessário

## visualizações

### análise exploratória

**vendas por horário**

![vendas por horário](images/vendas_por_horario.png)

gráfico de linha mostrando distribuição de vendas por categoria de café (café preto vs leite e doces) ao longo do dia. agrupa os tipos individuais em categorias para melhor visualização e identifica padrões de consumo e picos de demanda.

**horário de pico**

![horário de pico](images/horario_pico.png)

visualização hexbin mostrando concentração de vendas por horário. identifica períodos de maior movimento e ajuda a entender a distribuição temporal das vendas.

**cafés no horário de pico**

![cafés no pico](images/cafes_pico.png)

gráficos de barras comparando vendas por tipo nos horários de maior movimento. mostra preferências durante períodos de alta demanda, útil para planejamento de estoque.

### avaliação dos modelos

**matriz de confusão**

![matriz de confusão](images/matriz_confusao.png)

mostra performance do modelo na classificação. diagonal principal indica acertos, fora da diagonal mostra erros de classificação. útil para identificar quais categorias são mais difíceis de prever.

**top 10 features determinantes**

![features importantes](images/features_importantes.png)

gráfico de barras horizontal com features mais importantes ordenado por feature importance do xgboost. ajuda a entender quais fatores mais influenciam a escolha do café.

## estrutura do projeto

```
coffee-prediction-analysis/
├── data/
│   ├── coffee_cleaned.csv
│   └── model_trained_coffee.csv
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_recommendation_system.ipynb
└── images/
    ├── vendas_por_horario.png
    ├── horario_pico.png
    ├── cafes_pico.png
    ├── matriz_confusao.png
    └── features_importantes.png
```

## resultados e uso dos modelos

**modelo 1 (binário)**: alcança 68% de acurácia, sendo mais preciso para "leite e doces" (78% precision/recall) do que para "café preto" (43% precision/recall). este modelo é ideal para análises de alto nível e planejamento estratégico, onde a distinção entre categorias amplas é suficiente. sua simplicidade e velocidade o tornam adequado para aplicações em tempo real que requerem respostas rápidas.

**modelo 2 (hierárquico)**: alcança 61% de acurácia geral, mas oferece granularidade adicional ao diferenciar três categorias específicas. destaca-se especialmente no recall de categorias minoritárias (chocolate quente com 62% de recall vs 43% do modelo binário para café preto). este modelo é mais adequado para sistemas de recomendação personalizados, análises detalhadas de comportamento do cliente e aplicações onde a diferenciação entre tipos específicos de café é importante. a abordagem hierárquica permite melhor interpretabilidade e ajuste fino para categorias específicas.
