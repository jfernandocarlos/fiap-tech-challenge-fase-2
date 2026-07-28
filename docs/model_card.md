# Model Card — MovieLens NCF

> Formato baseado em [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).

---

## Informações gerais

| Campo | Valor |
|---|---|
| **Nome** | `movielens-ncf` (NCFModel) |
| **Tipo** | Neural Collaborative Filtering (embeddings + MLP, classificação binária) |
| **Framework** | PyTorch |
| **Versão** | 1.0.0 |
| **Data de treinamento** | 01/06/2026 |
| **Autor** | José Fernando Carlos — RM373126 |

---

## Descrição

A rede aprende um **embedding** para cada usuário e cada filme, concatena os
dois vetores e passa por uma MLP que estima a probabilidade do usuário **gostar**
do filme (`rating >= 4`). A saída é um logit; a sigmoid é aplicada na avaliação.

### Arquitetura

```
user_id → Embedding(610, 32) ┐
                             ├─ concat(64) → Linear(64) → BN → ReLU → Dropout(0.2)
movie_id → Embedding(3649,32)┘                → Linear(32) → BN → ReLU → Dropout(0.2)
                                              → Linear(1)  → logit → sigmoid → P(like)
```

Total de parâmetros: ~142,7k.

---

## Uso pretendido

- **Para quê:** ranquear filmes por probabilidade de afinidade e gerar
  recomendações personalizadas.
- **Usuários:** times de produto/recomendação e data science.
- **Fora do escopo:** decisões sensíveis (crédito, preço) ou qualquer uso que
  não seja recomendação de conteúdo.

---

## Dados de treinamento

| Campo | Valor |
|---|---|
| **Dataset** | MovieLens (`ratings.csv`, `movies.csv`) |
| **Interações (após filtro)** | 90.274 |
| **Usuários / Filmes** | 610 / 3.649 |
| **Filtro** | ≥ 5 interações por usuário e por filme |
| **Split** | 70% treino / 10% validação / 20% teste (aleatório, seed 42) |
| **Alvo** | `liked` = 1 se `rating >= 4` |

Distribuição do alvo: aproximadamente equilibrada (~50% positivos no teste).

---

## Métricas de performance

Conjunto de teste (~18k interações). Valores reais gerados pelo pipeline:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| DummyClassifier | 0.5035 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4965 |
| LogisticRegression | 0.7104 | 0.7041 | 0.7186 | 0.7113 | 0.7741 | 0.7565 |
| RandomForest | 0.7119 | 0.7003 | 0.7337 | 0.7166 | 0.7792 | 0.7626 |
| GradientBoosting | 0.7119 | 0.7035 | 0.7253 | 0.7142 | 0.7751 | 0.7486 |
| **NCF (PyTorch)** | **0.7129** | 0.7000 | **0.7380** | **0.7185** | **0.7833** | **0.7669** |

A NCF lidera em F1, ROC-AUC e PR-AUC, confirmando ganho de ranqueamento sobre
os baselines.

---

## Limitações

### Técnicas
- **Cold start:** usuários/filmes não vistos no treino não têm embedding e são
  descartados na avaliação. Em produção exigiriam fallback (ex.: popularidade).
- Dataset pequeno (~90k interações, 610 usuários) — generalização limitada.
- Não usa conteúdo do item (gêneros, sinopse) na rede; apenas IDs.

### De negócio
- Mede afinidade prevista, não satisfação real pós-recomendação.
- Limiar `rating >= 4` é uma simplificação do conceito de "gostar".

---

## Viés e fairness

- **Popularidade:** filmes muito populares dominam o sinal; o modelo pode
  reforçar bolhas de popularidade (efeito "rich get richer").
- **Usuários pouco ativos:** embeddings de usuários com poucas interações são
  menos confiáveis.
- **Recomendação:** monitorar diversidade/novidade das recomendações e cobertura
  do catálogo, não só acurácia.

---

## Cenários de falha

| Cenário | Impacto | Mitigação |
|---|---|---|
| Cold start (user/item novo) | Sem predição | Fallback por popularidade/conteúdo |
| Concept drift (gosto muda) | Degradação silenciosa | Monitoramento + retreino agendado |
| Distribuição de rating muda | Alvo desbalanceia | Revisar limiar `like_threshold` |

---

## Reprodutibilidade

```bash
git clone <repo> && cd movie-recommender
make install
cp .env.example .env
make download-data      # MovieLens via Kaggle
make repro              # pipeline determinístico (seed=42)
```

Modelo registrado no MLflow Model Registry como `movielens-ncf`, promovido a
`staging` e `production` via `make promote-staging` / `make promote-production`.
