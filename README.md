# Movie Recommender — Tech Challenge Fase 2

Sistema de recomendação de filmes baseado no comportamento dos usuários
(MovieLens). O modelo central é uma rede neural **Neural Collaborative
Filtering (NCF)** treinada com **PyTorch**, comparada com baselines do
**scikit-learn**. O pipeline é reprodutível com **DVC** (4 stages), os
experimentos são rastreados no **MLflow** (com Model Registry) e tudo roda
containerizado com **Docker**.

> Pós-graduação em Engenharia de Machine Learning — FIAP
> Autor: **José Fernando Carlos — RM373126**

---

## Sumário

- [Problema](#problema)
- [Arquitetura](#arquitetura)
- [Stack](#stack)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Setup do zero](#setup-do-zero)
- [Pipeline (DVC)](#pipeline-dvc)
- [MLflow e Model Registry](#mlflow-e-model-registry)
- [Docker](#docker)
- [Resultados](#resultados)
- [Testes e qualidade](#testes-e-qualidade)
- [Documentação](#documentação)

---

## Problema

Uma empresa de e-commerce/streaming precisa recomendar itens com base no
comportamento de navegação. Formalizamos como **classificação binária de
afinidade**: dado um par `(usuário, filme)`, prever se o usuário vai **gostar**
do item (`rating >= 4`). A probabilidade prevista serve para ranquear e
recomendar os itens mais prováveis de agradar.

- **Dataset:** [MovieLens](https://www.kaggle.com/datasets/ayushimishra2809/movielens-dataset)
  (`ratings.csv`, `movies.csv`) — ~100k interações user-item.
- **Alvo:** `liked = 1` se `rating >= 4`, senão `0`.

## Arquitetura

```
data/raw (ratings, movies)
        │
        ▼  [stage 1: preprocess]   filtra ativos + alvo binário + split
data/processed/{train,val,test}.parquet
        │
        ▼  [stage 2: feature_eng]  índices contíguos + estatísticas (treino)
data/processed/feat_{train,val,test}.parquet + mappings.json
        │
        ├──────────────┐
        ▼              ▼
[stage 3: train]   (baselines sklearn)
NCF (PyTorch)           │
models/ncf.pt           │
        │               │
        ▼  [stage 4: evaluate]  6 métricas + relatório + metrics.json
        └──────────────┘
                │
                ▼
        MLflow (tracking + Model Registry → staging/production)
```

Detalhes em [`docs/architecture.md`](docs/architecture.md).

### Design patterns aplicados

| Padrão | Onde | Por quê |
|--------|------|---------|
| **Strategy** | `data/splitting.py` (`RandomSplitStrategy`, `TemporalSplitStrategy`) | Trocar a política de split sem alterar o pipeline |
| **Factory** | `models/factory.py` (`create_ncf`, `create_baselines`) | Isolar a construção dos modelos dos parâmetros |
| **Template Method** | `pipeline/base.py` (`PipelineStage.run`) | Esqueleto comum dos stages (setup, log, erro) |

## Stack

- **PyTorch** — rede neural (NCF, embeddings + MLP).
- **scikit-learn** — baselines e métricas.
- **MLflow** — tracking de experimentos + Model Registry.
- **DVC** — versionamento de dados e pipeline reprodutível.
- **Poetry** — gestão de dependências (prod/dev) + lock file.
- **Pydantic Settings + structlog** — config e logging.
- **ruff + pre-commit + pytest** — qualidade.

## Estrutura do repositório

```
.
├── src/recsys/
│   ├── config.py            # Settings (.env) + load_params (params.yaml)
│   ├── logging_config.py
│   ├── data/                # loader.py, splitting.py (Strategy)
│   ├── features/            # engineering.py
│   ├── models/              # ncf.py, baselines.py, factory.py, training.py
│   ├── evaluation/          # metrics.py
│   ├── tracking/            # mlflow_tracker.py, registry.py
│   └── pipeline/            # base.py (Template Method) + 4 stages
├── scripts/                 # download_data, validate_env, promote_model, run_pipeline
├── tests/                   # pytest (Strategy, Factory, features, NCF, métricas)
├── docs/                    # architecture, ml_canvas, model_card, monitoring_plan
├── dvc.yaml / params.yaml   # pipeline e hiperparâmetros
├── Dockerfile / docker-compose.yml
├── pyproject.toml / poetry.lock
└── README.md / TUTORIAL_REPLICACAO.md
```

## Setup do zero

Pré-requisitos: Python 3.10–3.12, [Poetry](https://python-poetry.org/) e Git.

```bash
# 1. Instalar dependências (prod + dev) e hooks
make install            # poetry install + pre-commit install

# 2. Configurar variáveis de ambiente
cp .env.example .env

# 3. Validar o ambiente
make validate           # checa Python, libs e configs

# 4. Baixar o dataset (precisa de credenciais Kaggle em ~/.kaggle/kaggle.json)
make download-data
```

## Pipeline (DVC)

O pipeline tem **4 stages** declarados em `dvc.yaml`:

```bash
make repro              # = dvc repro (preprocess → feature_eng → train → evaluate)
```

Reexecuta apenas o que mudou (DVC compara hashes de código, dados e params).
Os hiperparâmetros ficam em `params.yaml` — alterá-los invalida os stages
afetados. Para rodar o pipeline sem DVC (ex.: dentro do container):

```bash
make train              # = python -m scripts.run_pipeline
```

## MLflow e Model Registry

```bash
make mlflow-ui          # abre a UI em http://localhost:5000
```

Cada execução registra **5 runs** (4 baselines + NCF) com params, métricas e o
modelo serializado. A NCF é registrada no **Model Registry** e promovida via
aliases:

```bash
make promote-staging      # alias "staging" na última versão
make promote-production   # alias "production" na última versão
```

## Docker

Imagem **multi-stage** (builder com Poetry + runtime enxuto) e `docker-compose`
com dois serviços: servidor MLflow e treino.

```bash
make docker-build       # docker compose build
make docker-up          # sobe MLflow + roda o pipeline de treino
```

## Resultados

Conjunto de teste (~18k interações, seed 42). Métricas reais geradas pelo
pipeline (`metrics.json` / [`docs/comparativo_baselines_ncf.md`](docs/comparativo_baselines_ncf.md)):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|--------|----------|-----------|--------|------|---------|--------|
| DummyClassifier | 0.5035 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4965 |
| LogisticRegression | 0.7104 | 0.7041 | 0.7186 | 0.7113 | 0.7741 | 0.7565 |
| RandomForest | 0.7119 | 0.7003 | 0.7337 | 0.7166 | 0.7792 | 0.7626 |
| GradientBoosting | 0.7119 | 0.7035 | 0.7253 | 0.7142 | 0.7751 | 0.7486 |
| **NCF (PyTorch)** | **0.7129** | 0.7000 | **0.7380** | **0.7185** | **0.7833** | **0.7669** |

A NCF supera todos os baselines em **F1, ROC-AUC e PR-AUC**. Os baselines são
fortes porque usam atributos derivados (taxa de "like" por usuário/filme), mas
a rede captura padrões latentes via embeddings, que rendem a melhor capacidade
de ranqueamento (ROC-AUC e PR-AUC).

## Testes e qualidade

```bash
make lint               # ruff check + format --check
make test               # pytest (15 testes)
```

## Documentação

- [`docs/architecture.md`](docs/architecture.md) — arquitetura e decisões.
- [`docs/ml_canvas.md`](docs/ml_canvas.md) — ML Canvas do problema.
- [`docs/model_card.md`](docs/model_card.md) — Model Card (performance, limitações, vieses).
- [`docs/monitoring_plan.md`](docs/monitoring_plan.md) — plano de monitoramento.
- [`TUTORIAL_REPLICACAO.md`](TUTORIAL_REPLICACAO.md) — recriar o projeto do zero, commit a commit.
