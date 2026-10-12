# Results (sim)

300 stocks, 59 out-of-sample months. A correct model's bias statistic lands within 1 +/- 0.184.

## Risk model forecasts

Mean bias statistic by test portfolio group:

| model | active | all | benchmark | long | longshort |
|---|---|---|---|---|---|
| sample | 0.965 | 1.009 | 1.030 | 0.973 | 1.065 |
| ledoit_wolf | 0.976 | 1.024 | 1.081 | 1.016 | 1.054 |
| statistical | 0.968 | 1.001 | 1.032 | 0.967 | 1.050 |
| fundamental_no_vra | 0.895 | 0.968 | 0.979 | 0.931 | 1.041 |
| fundamental | 0.891 | 0.964 | 0.958 | 0.932 | 1.032 |
| hybrid | 0.891 | 0.960 | 0.959 | 0.931 | 1.023 |

Across all test portfolios (MRAD = mean |bias - 1|, in_ci = share inside the 95% band):

| model | mean_bias | mrad | in_ci |
|---|---|---|---|
| sample | 1.009 | 0.070 | 0.961 |
| ledoit_wolf | 1.024 | 0.070 | 0.961 |
| statistical | 1.001 | 0.068 | 0.961 |
| fundamental_no_vra | 0.968 | 0.083 | 0.941 |
| fundamental | 0.964 | 0.082 | 0.941 |
| hybrid | 0.960 | 0.081 | 0.961 |

Long-only minimum-variance portfolio built from each model:

| model | predicted_vol | realised_vol |
|---|---|---|
| sample | 12.41% | 13.16% |
| ledoit_wolf | 12.10% | 13.10% |
| statistical | 12.62% | 13.11% |
| fundamental_no_vra | 14.74% | 13.32% |
| fundamental | 15.04% | 13.32% |
| hybrid | 15.06% | 13.13% |

## Optimiser backtest

| strategy | active_return | realised_te | predicted_te | te_ratio | info_ratio | turnover | names | abs_hidden_exposure |
|---|---|---|---|---|---|---|---|---|
| MVO | 3.24% | 3.07% | 2.56% | 1.199 | 1.056 | 26.31% | 86.153 | 0.224 |
| MVO-Hybrid | 3.54% | 2.57% | 2.49% | 1.033 | 1.375 | 27.85% | 86.949 | 0.045 |
| MVO+AAF | 2.78% | 2.77% | 2.49% | 1.113 | 1.003 | 21.16% | 93.203 | 0.200 |
| Robust | 3.04% | 2.87% | 2.39% | 1.199 | 1.059 | 24.64% | 92.119 | 0.208 |
| Robust+AAF | 2.54% | 2.57% | 2.32% | 1.107 | 0.988 | 19.84% | 99.576 | 0.184 |

## Return attribution (annualised mean contribution to active return)

| strategy | country | industry | beta | momentum | size | resvol | liquidity | value | quality | growth | specific | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MVO | 0.00% | 0.16% | -0.06% | 0.01% | 0.60% | -0.03% | 0.08% | 0.09% | -0.16% | 0.02% | 3.15% | 3.87% |
| MVO-Hybrid | 0.00% | 0.16% | -0.06% | -0.04% | 0.57% | -0.02% | 0.07% | 0.09% | -0.19% | 0.00% | 3.62% | 4.21% |
| MVO+AAF | 0.00% | 0.17% | -0.05% | 0.04% | 0.46% | -0.03% | 0.07% | 0.10% | -0.15% | 0.02% | 2.67% | 3.28% |
| Robust | 0.00% | 0.15% | -0.06% | 0.01% | 0.57% | -0.03% | 0.08% | 0.10% | -0.16% | 0.02% | 2.95% | 3.63% |
| Robust+AAF | 0.00% | 0.16% | -0.05% | 0.03% | 0.44% | -0.03% | 0.06% | 0.10% | -0.16% | 0.02% | 2.45% | 3.02% |

## Robust penalty sweep (fundamental model)

| kappa | active_return | realised_te | info_ratio | turnover | names |
|---|---|---|---|---|---|
| 0.000 | 3.24% | 3.07% | 1.056 | 26.31% | 86.153 |
| 0.500 | 3.14% | 2.97% | 1.057 | 25.48% | 89.000 |
| 1.000 | 3.04% | 2.87% | 1.059 | 24.64% | 92.119 |
| 2.000 | 2.83% | 2.67% | 1.060 | 22.94% | 98.271 |
| 4.000 | 2.29% | 2.29% | 0.999 | 19.53% | 110.949 |
