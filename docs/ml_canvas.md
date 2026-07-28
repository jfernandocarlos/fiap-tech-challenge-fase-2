# ML Canvas — Recomendador de Filmes

| Bloco | Descrição |
|-------|-----------|
| **Proposta de valor** | Recomendar filmes que o usuário tende a gostar, aumentando engajamento e tempo de uso da plataforma. |
| **Tarefa de ML** | Classificação binária de afinidade: prever `P(liked)` para o par `(usuário, filme)` e usar a probabilidade para ranquear. |
| **Fonte de dados** | MovieLens — `ratings.csv` (userId, movieId, rating, timestamp) e `movies.csv` (movieId, title, genres). |
| **Coleta** | Histórico de avaliações dos usuários (feedback explícito convertido em implícito: `rating >= 4` ⇒ gostou). |
| **Features** | Para a rede: índices de usuário e filme (embeddings). Para baselines: médias e taxas de "like" por usuário/filme, popularidade. |
| **Modelo** | Rede neural NCF (embeddings + MLP, PyTorch); baselines: Dummy, LogReg, RandomForest, GradientBoosting. |
| **Métricas offline** | Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC. Métrica principal: **PR-AUC/ROC-AUC** (qualidade de ranqueamento). |
| **Métricas de negócio** | CTR das recomendações, taxa de conclusão, diversidade/cobertura do catálogo. |
| **Decisões** | Itens com maior `P(liked)` entram na lista de recomendações do usuário. |
| **Predição** | Batch (pré-computa ranking por usuário) ou online (scoring sob demanda). |
| **Monitoramento** | Drift de distribuição de ratings, queda de métricas, cobertura. Ver `monitoring_plan.md`. |
| **Riscos** | Cold start (usuário/filme novo), viés de popularidade, concept drift de gosto. |
