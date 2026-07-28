# Plano de Monitoramento

Separa explicitamente **o que já está implementado** do que é **evolução
planejada**, conforme recomendado.

## Implementado neste projeto

- **Tracking de experimentos (MLflow):** params, métricas e modelo de cada run
  ficam registrados, permitindo comparar versões ao longo do tempo.
- **Model Registry:** versionamento do modelo com aliases `staging`/`production`,
  servindo de fonte única da versão em uso.
- **Métricas offline versionadas:** `metrics.json` (rastreado pelo DVC) e
  relatório comparativo em `docs/`, regenerados a cada `dvc repro`.
- **Versionamento de dados (DVC):** qualquer mudança no dataset ou nos params
  invalida os stages afetados, garantindo rastreabilidade.
- **Logging estruturado (structlog):** logs em formato consistente, prontos para
  ingestão por ferramentas externas.

## Evolução planejada (não implementado)

| Item | Descrição | Ferramenta candidata |
|------|-----------|----------------------|
| **Data drift** | Comparar distribuição de ratings/usuários em produção vs. treino | Evidently, NannyML |
| **Performance drift** | Acompanhar F1/ROC-AUC em janelas de tempo com feedback real | MLflow + jobs agendados |
| **Métricas de negócio** | CTR, conclusão, diversidade e cobertura do catálogo | Dashboard (Grafana) |
| **Alertas** | Disparo quando métrica cruza limiar | Alertmanager / webhook |
| **Retreino** | Pipeline agendado (`dvc repro`) com promoção automática condicionada a métricas | CI/CD + DVC |

## Gatilhos de retreino sugeridos

- Queda relativa de PR-AUC > 5% em relação à versão em produção.
- Drift significativo na distribuição de ratings (ex.: PSI > 0.2).
- Crescimento do catálogo/base de usuários acima de um limiar.
