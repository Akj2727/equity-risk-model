"""End-to-end run: fit models, evaluate forecasts, backtest the optimisers.

    python -m erm.run                       # simulated universe, runs anywhere
    python -m erm.run --data sp500          # real S&P 500 data via yfinance
    python -m erm.run --data tickers --tickers my_nse_list.txt --fundamentals fund.csv
"""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .alpha import price_signal
from .backtest import DEFAULT_STRATEGIES, Strategy, run_backtest
from .data.simulate import simulate_universe
from .evaluation import evaluate_models
from .optimize import OptConfig
from .pipeline import ALL_MODELS, Pipeline

KAPPAS = [0.0, 0.5, 1.0, 2.0, 4.0]


def load(args):
    if args.data == "sim":
        uni, truth = simulate_universe()
        return uni, (lambda d: truth.signal.loc[d]), truth.hidden_loading
    from .data.real import download_yfinance, sp500_constituents, universe_from_frames
    sectors = None
    if args.data == "sp500":
        cons = sp500_constituents()
        tickers = cons["Symbol"].tolist()
        sectors = cons.set_index("Symbol")["GICS Sector"]
    else:
        tickers = [t.strip() for t in Path(args.tickers).read_text().split() if t.strip()]
    if args.max_names:
        tickers = tickers[: args.max_names]
    close, volume, shares, industry = download_yfinance(tickers, args.start, args.end, sectors)
    fund = pd.read_csv(args.fundamentals) if args.fundamentals else None
    uni = universe_from_frames(close, volume, shares, industry, fund)
    return uni, (lambda d: price_signal(uni, d)), None


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def md_table(df: pd.DataFrame, pct_cols=(), digits=3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(fmt_pct(v) if c in pct_cols else f"{v:.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plots(out: Path, ev, bt, sweep):
    bias = ev["bias"].pivot(index="model", columns="group", values="mean_bias").reindex(ALL_MODELS)
    ax = bias.drop(columns="all").plot.bar(figsize=(9, 4), rot=20)
    ci = ev["ci_halfwidth"]
    ax.axhspan(1 - ci, 1 + ci, color="grey", alpha=0.15, label="95% band")
    ax.axhline(1, color="black", lw=0.8)
    ax.set_ylabel("bias statistic")
    ax.set_title("Out-of-sample bias statistic by test portfolio group")
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.savefig(out / "bias.png", dpi=130)
    plt.close()

    s = bt.summary.set_index("strategy")
    ax = s[["predicted_te", "realised_te"]].plot.bar(figsize=(8, 4), rot=0)
    ax.set_ylabel("annual tracking error")
    ax.set_title("Predicted vs realised tracking error")
    plt.tight_layout()
    plt.savefig(out / "tracking_error.png", dpi=130)
    plt.close()

    m = bt.monthly.pivot(index="date", columns="strategy", values="active_return")
    ax = (1 + m).cumprod().plot(figsize=(9, 4))
    ax.set_ylabel("cumulative active return (growth of 1)")
    ax.set_title("Active return vs benchmark, after costs")
    plt.tight_layout()
    plt.savefig(out / "active_return.png", dpi=130)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(sweep["kappa"], sweep["turnover"], "o-", label="turnover")
    ax1.set_xlabel("robust penalty kappa")
    ax1.set_ylabel("one-way monthly turnover")
    ax2 = ax1.twinx()
    ax2.plot(sweep["kappa"], sweep["info_ratio"], "s--", color="C1", label="information ratio")
    ax2.set_ylabel("information ratio")
    fig.legend(loc="upper right", fontsize=8)
    ax1.set_title("Robust optimisation: penalty sweep")
    plt.tight_layout()
    plt.savefig(out / "robust_sweep.png", dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["sim", "sp500", "tickers"], default="sim")
    ap.add_argument("--tickers", help="file of tickers, whitespace separated")
    ap.add_argument("--fundamentals", help="long CSV: date,ticker,book_to_price,earnings_yield,roe,sales_growth")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-names", type=int, default=None)
    ap.add_argument("--eval-start-month", type=int, default=36,
                    help="months of history before out-of-sample evaluation starts")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or f"results/{args.data}")
    out.mkdir(parents=True, exist_ok=True)

    uni, signal_fn, hidden = load(args)
    print(f"universe: {uni.returns.shape[1]} stocks, {uni.returns.shape[0]} days")
    pipe = Pipeline(uni)
    dates = uni.month_ends()[args.eval_start_month:]

    print("evaluating risk models")
    ev = evaluate_models(pipe, dates, ALL_MODELS)
    print("backtesting")
    cfg = OptConfig()
    bt = run_backtest(pipe, dates, signal_fn, DEFAULT_STRATEGIES, cfg, hidden_loading=hidden)
    print("robust sweep")
    sweep_rows = []
    for k in KAPPAS:
        r = run_backtest(pipe, dates, signal_fn, [Strategy(f"k={k}", "fundamental", kappa=k)], cfg,
                         hidden_loading=hidden)
        row = r.summary.iloc[0].to_dict()
        row["kappa"] = k
        sweep_rows.append(row)
    sweep = pd.DataFrame(sweep_rows)

    ev["bias"].to_csv(out / "bias.csv", index=False)
    ev["min_variance"].to_csv(out / "min_variance.csv", index=False)
    bt.monthly.to_csv(out / "backtest_monthly.csv", index=False)
    bt.summary.to_csv(out / "backtest_summary.csv", index=False)
    bt.attribution.to_csv(out / "attribution.csv")
    sweep.to_csv(out / "robust_sweep.csv", index=False)
    plots(out, ev, bt, sweep)

    allb = ev["bias"][ev["bias"].group == "all"][["model", "mean_bias", "mrad", "in_ci"]]
    groups = ev["bias"].pivot(index="model", columns="group", values="mean_bias").reindex(ALL_MODELS)
    groups = groups.reset_index()
    summ = bt.summary.copy()
    pct = ("active_return", "realised_te", "predicted_te", "turnover", "abs_hidden_exposure")
    attr = bt.attribution.copy()
    attr.insert(0, "strategy", attr.index)
    attr = attr.reset_index(drop=True)

    text = [
        f"# Results ({args.data})",
        "",
        f"{uni.returns.shape[1]} stocks, {ev['months']} out-of-sample months. "
        f"A correct model's bias statistic lands within 1 +/- {ev['ci_halfwidth']:.3f}.",
        "",
        "## Risk model forecasts",
        "",
        "Mean bias statistic by test portfolio group:",
        "",
        md_table(groups),
        "",
        "Across all test portfolios (MRAD = mean |bias - 1|, in_ci = share inside the 95% band):",
        "",
        md_table(allb.reset_index(drop=True)),
        "",
        "Long-only minimum-variance portfolio built from each model:",
        "",
        md_table(ev["min_variance"], pct_cols=("predicted_vol", "realised_vol")),
        "",
        "## Optimiser backtest",
        "",
        md_table(summ, pct_cols=pct),
        "",
        "## Return attribution (annualised mean contribution to active return)",
        "",
        md_table(attr, pct_cols=tuple(c for c in attr.columns if c != "strategy")),
        "",
        "## Robust penalty sweep (fundamental model)",
        "",
        md_table(sweep[["kappa", "active_return", "realised_te", "info_ratio", "turnover", "names"]],
                 pct_cols=("active_return", "realised_te", "turnover")),
        "",
    ]
    (out / "summary.md").write_text("\n".join(text))
    print(f"wrote {out}/summary.md")


if __name__ == "__main__":
    main()
