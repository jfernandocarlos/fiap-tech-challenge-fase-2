# Relatório Comparativo — Recomendador de Filmes

> **Gerado em:** 18/07/2026 09:19:47
> **Dataset:** MovieLens (interações user-item)
> **Alvo:** `liked` (rating >= threshold)
> **Seed:** 42

---

## 1. Métricas no conjunto de teste

| Modelo | ACCURACY | PRECISION | RECALL | F1 | ROC_AUC | PR_AUC |
|---|---|---|---|---|---|---|
| DummyClassifier | 0.5117 | 0.5117 | 1.0000 | 0.6770 | 0.5000 | 0.5117 |
| LogisticRegression | 0.7065 | 0.7138 | 0.7118 | 0.7128 | 0.7760 | 0.7687 |
| RandomForest | 0.7141 | 0.7126 | 0.7394 | 0.7257 | 0.7788 | 0.7668 |
| GradientBoosting | 0.7076 | 0.7010 | 0.7472 | 0.7234 | 0.7751 | 0.7581 |
| NCF_PyTorch | 0.7114 | 0.7050 | 0.7498 | 0.7267 | 0.7847 | 0.7812 |

---

## 2. Análise

- **Melhor F1 geral:** NCF_PyTorch (0.7267)
- **Posição da NCF no ranking de F1:** 1º de 5 modelos
- **NCF (PyTorch):** F1=0.7267 | ROC-AUC=0.7847 | PR-AUC=0.7812
- **Melhor baseline (sem rede neural):** RandomForest

A rede neural aprende embeddings latentes de usuários e filmes, capturando
padrões de afinidade que os baselines (apoiados apenas em médias) não modelam
diretamente.
