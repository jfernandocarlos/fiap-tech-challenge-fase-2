# Arquitetura

## Visão geral

O projeto é um pipeline de ML reprodutível para recomendação de filmes,
organizado em camadas com responsabilidades únicas (SOLID) e orquestrado pelo
DVC.

```
┌─────────────────────────────────────────────────────────────┐
│                         params.yaml                           │
│      (hiperparâmetros rastreados pelo DVC: data/model/train)  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
data/raw/ratings.csv ──► [preprocess] ──► train/val/test.parquet
                                │
                                ▼
                          [feature_eng] ──► feat_*.parquet + mappings.json
                                │
                ┌───────────────┴───────────────┐
                ▼                                ▼
           [train] (NCF/PyTorch)          (baselines no evaluate)
           models/ncf.pt                        │
                │                                │
                └──────────► [evaluate] ◄────────┘
                                │
                                ▼
                metrics.json + docs/comparativo_*.md
                                │
                                ▼
                    MLflow (tracking + Registry)
```

## Camadas (`src/recsys`)

| Módulo | Responsabilidade |
|--------|------------------|
| `config.py` | `Settings` (infra via `.env`) + `load_params` (ML via `params.yaml`) |
| `data/loader.py` | Leitura e filtro das avaliações brutas |
| `data/splitting.py` | Estratégias de split (**Strategy**) |
| `features/engineering.py` | Índices contíguos + estatísticas de treino |
| `models/ncf.py` | Rede neural NCF (embeddings + MLP) |
| `models/baselines.py` | Baselines scikit-learn |
| `models/factory.py` | Construção dos modelos (**Factory**) |
| `models/training.py` | Loop de treino + early stopping |
| `evaluation/metrics.py` | Métricas e tabela comparativa |
| `tracking/mlflow_tracker.py` | Logging de runs no MLflow |
| `tracking/registry.py` | Promoção no Model Registry (aliases) |
| `pipeline/base.py` | Esqueleto dos stages (**Template Method**) |
| `pipeline/*.py` | Os 4 stages do DVC |

## Decisões de projeto

- **Formalização como classificação binária** (`liked`): permite comparar a
  rede com baselines tabulares usando métricas padronizadas (F1, ROC-AUC, etc.).
- **Embeddings em vez de one-hot:** dimensionalidade fixa e aprendizado de
  similaridade latente entre usuários/filmes.
- **Estatísticas calculadas só no treino:** evita vazamento de dados para os
  baselines.
- **`params.yaml` separado do `.env`:** parâmetros de ML versionados pelo DVC;
  infraestrutura e segredos no `.env`.
- **Saída em logits + `BCEWithLogitsLoss`:** mais estável numericamente que
  `Sigmoid + BCELoss`.

## Reprodutibilidade

- Seed única (`RANDOM_SEED=42`) aplicada a NumPy, PyTorch e splits.
- `poetry.lock` fixa versões exatas.
- DVC versiona dados/artefatos e garante `dvc repro` determinístico.
- Docker multi-stage isola o ambiente de execução.
