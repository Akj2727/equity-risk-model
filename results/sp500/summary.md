# Results (sp500)

277 stocks, 117 out-of-sample months. A correct model's bias statistic lands within 1 +/- 0.131.

## Risk model forecasts

Mean bias statistic by test portfolio group:

| model | active | all | benchmark | long | longshort |
|---|---|---|---|---|---|
| sample | 1.020 | 1.052 | 1.014 | 1.094 | 1.028 |
| ledoit_wolf | 1.029 | 1.065 | 1.058 | 1.135 | 1.012 |
| statistical | 1.131 | 1.068 | 1.049 | 1.089 | 1.015 |
| fundamental_no_vra | 1.032 | 1.032 | 1.041 | 1.044 | 1.020 |
| fundamental | 1.014 | 1.012 | 1.023 | 1.033 | 0.990 |
| hybrid | 1.013 | 1.010 | 1.024 | 1.033 | 0.985 |

Across all test portfolios (MRAD = mean |bias - 1|, in_ci = share inside the 95% band):

| model | mean_bias | mrad | in_ci |
|---|---|---|---|
| sample | 1.052 | 0.066 | 0.961 |
| ledoit_wolf | 1.065 | 0.083 | 0.745 |
| statistical | 1.068 | 0.082 | 0.882 |
| fundamental_no_vra | 1.032 | 0.047 | 1.000 |
| fundamental | 1.012 | 0.041 | 0.980 |
| hybrid | 1.010 | 0.041 | 0.980 |

Long-only minimum-variance portfolio built from each model:

| model | predicted_vol | realised_vol |
|---|---|---|
| sample | 9.52% | 12.01% |
| ledoit_wolf | 9.31% | 12.04% |
| statistical | 9.35% | 12.08% |
| fundamental_no_vra | 11.96% | 12.17% |
| fundamental | 12.53% | 12.19% |
| hybrid | 12.54% | 12.33% |

## Optimiser backtest

| strategy | active_return | realised_te | predicted_te | te_ratio | info_ratio | turnover | names |
|---|---|---|---|---|---|---|---|
| MVO | 0.13% | 4.30% | 2.55% | 1.687 | 0.030 | 18.94% | 73.607 |
| MVO-Hybrid | -0.04% | 4.30% | 2.52% | 1.704 | -0.009 | 18.89% | 71.436 |
| MVO+AAF | 0.07% | 4.16% | 2.52% | 1.647 | 0.016 | 15.70% | 80.291 |
| Robust | 0.10% | 4.21% | 2.45% | 1.717 | 0.024 | 17.72% | 79.265 |
| Robust+AAF | 0.06% | 4.09% | 2.44% | 1.677 | 0.014 | 14.82% | 85.436 |

## Return attribution (annualised mean contribution to active return)

| strategy | country | industry | beta | momentum | size | resvol | liquidity | specific | total |
|---|---|---|---|---|---|---|---|---|---|
| MVO | 0.00% | -0.19% | -0.08% | 0.28% | 0.11% | -0.02% | 0.10% | 0.39% | 0.58% |
| MVO-Hybrid | -0.00% | -0.21% | -0.11% | 0.27% | 0.11% | -0.01% | 0.06% | 0.30% | 0.41% |
| MVO+AAF | 0.00% | -0.19% | -0.08% | 0.30% | 0.11% | -0.02% | 0.09% | 0.24% | 0.45% |
| Robust | -0.00% | -0.18% | -0.08% | 0.27% | 0.11% | -0.02% | 0.10% | 0.33% | 0.52% |
| Robust+AAF | -0.00% | -0.18% | -0.08% | 0.29% | 0.11% | -0.02% | 0.09% | 0.21% | 0.41% |

## Robust penalty sweep (fundamental model)

| kappa | active_return | realised_te | info_ratio | turnover | names |
|---|---|---|---|---|---|
| 0.000 | 0.13% | 4.30% | 0.030 | 18.94% | 73.607 |
| 0.500 | 0.12% | 4.25% | 0.027 | 18.33% | 76.521 |
| 1.000 | 0.10% | 4.21% | 0.024 | 17.72% | 79.265 |
| 2.000 | 0.09% | 4.15% | 0.022 | 16.57% | 84.769 |
| 4.000 | 0.05% | 4.03% | 0.011 | 14.51% | 94.521 |
