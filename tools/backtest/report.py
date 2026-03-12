"""Reporter — metrics calculation and output formatting.

Calculates Sharpe ratio, max drawdown, win rate, profit factor, expectancy,
and other key metrics from backtest results. Formats output as console table,
CSV, and JSON.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "backtest_results"


def calculate_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Calculate performance metrics from a single backtest result.

    Args:
        result: Output from BacktestEngine.run()

    Returns:
        Dict with all computed metrics.
    """
    trades = result.get("closed_trades", [])
    equity_curve = result.get("equity_curve", [])

    if not trades:
        return _empty_metrics(result)

    # Win/loss classification
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]

    win_count = len(winners)
    loss_count = len(losers)
    total = len(trades)
    win_rate = win_count / total if total > 0 else 0.0

    # Gross profit / loss
    gross_profit = sum(t["pnl"] for t in winners) if winners else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 0.0

    avg_win = gross_profit / win_count if win_count > 0 else 0.0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0.0

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Avg win / avg loss ratio
    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

    # Expectancy: (win_rate * avg_win) - (loss_rate * avg_loss)
    loss_rate = 1.0 - win_rate
    expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Sharpe ratio from equity curve daily returns
    sharpe = _calculate_sharpe(equity_curve)

    # Max drawdown
    max_dd = result.get("max_drawdown", 0.0)

    # Average bars held
    avg_bars = sum(t["bars_held"] for t in trades) / total if total > 0 else 0.0

    # Exit reason breakdown
    exit_reasons: dict[str, int] = {}
    for t in trades:
        reason = t.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    total_return = result.get("total_return", 0.0)

    return {
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else 999.0,
        "avg_win_loss_ratio": round(avg_win_loss_ratio, 4) if avg_win_loss_ratio != float("inf") else 999.0,
        "expectancy": round(expectancy, 4),
        "total_return": round(total_return, 4),
        "trade_count": total,
        "avg_bars_held": round(avg_bars, 1),
        "exit_reasons": exit_reasons,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def _calculate_sharpe(equity_curve: list[dict[str, Any]], annualization: float = 252.0) -> float:
    """Calculate annualized Sharpe ratio from equity curve.

    Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
    """
    if len(equity_curve) < 2:
        return 0.0

    equities = [snap["equity"] for snap in equity_curve]
    returns = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

    if len(returns) < 2:
        return 0.0

    arr = np.array(returns, dtype=float)
    mean_ret = float(np.mean(arr))
    std_ret = float(np.std(arr, ddof=1))

    if std_ret == 0:
        return 0.0

    return mean_ret / std_ret * math.sqrt(annualization)


def _empty_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Return zero metrics when no trades were executed."""
    return {
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win_loss_ratio": 0.0,
        "expectancy": 0.0,
        "total_return": result.get("total_return", 0.0),
        "trade_count": 0,
        "avg_bars_held": 0.0,
        "exit_reasons": {},
        "gross_profit": 0.0,
        "gross_loss": 0.0,
    }


def format_summary_table(
    ranked_results: list[dict[str, Any]],
    sweep_mode: str,
    current_params: dict[str, Any] | None = None,
    top_n: int = 10,
) -> str:
    """Format ranked results as a console-friendly table.

    Shows top-N results + a "YOU ARE HERE" marker for current params.
    """
    lines: list[str] = []
    total = len(ranked_results)
    n_assets = ranked_results[0].get("n_assets", "?") if ranked_results else "?"

    lines.append(f"\n=== BACKTEST RESULTS: {sweep_mode} sweep ({total} combos, {n_assets} assets) ===\n")

    # Header based on sweep mode
    if sweep_mode in ("stops", "holding"):
        lines.append(f"{'Rank':>4}  {'SL%':>5}  {'TP%':>5}  {'Sharpe':>7}  {'MaxDD':>7}  "
                      f"{'WinRate':>7}  {'PF':>5}  {'AvgW/L':>7}  {'Trades':>6}  {'Return':>8}")
    elif sweep_mode == "atr-stops":
        lines.append(f"{'Rank':>4}  {'SL_ATR':>6}  {'TP_ATR':>6}  {'Sharpe':>7}  {'MaxDD':>7}  "
                      f"{'WinRate':>7}  {'PF':>5}  {'AvgW/L':>7}  {'Trades':>6}  {'Return':>8}")
    elif sweep_mode == "trailing":
        lines.append(f"{'Rank':>4}  {'Act%':>5}  {'Dist%':>5}  {'Sharpe':>7}  {'MaxDD':>7}  "
                      f"{'WinRate':>7}  {'PF':>5}  {'AvgW/L':>7}  {'Trades':>6}  {'Return':>8}")
    elif sweep_mode == "sizing":
        lines.append(f"{'Rank':>4}  {'MaxPos%':>7}  {'Risk%':>5}  {'Sharpe':>7}  {'MaxDD':>7}  "
                      f"{'WinRate':>7}  {'PF':>5}  {'AvgW/L':>7}  {'Trades':>6}  {'Return':>8}")
    else:
        lines.append(f"{'Rank':>4}  {'Sharpe':>7}  {'MaxDD':>7}  {'WinRate':>7}  {'PF':>5}  "
                      f"{'AvgW/L':>7}  {'Trades':>6}  {'Return':>8}")

    lines.append("-" * len(lines[-1]))

    # Top N results
    for i, r in enumerate(ranked_results[:top_n]):
        rank = i + 1
        m = r["metrics"]
        p = r["params"]

        sharpe = f"{m['sharpe']:.2f}"
        maxdd = f"{m['max_drawdown'] * 100:.1f}%"
        wr = f"{m['win_rate'] * 100:.0f}%"
        pf = f"{m['profit_factor']:.2f}"
        awl = f"{m['avg_win_loss_ratio']:.2f}"
        tc = f"{m['trade_count']}"
        ret = f"{m['total_return'] * 100:+.1f}%"

        if sweep_mode in ("stops", "holding"):
            sl = f"{p.get('default_stop_loss_pct', 0):.1f}%"
            tp = f"{p.get('default_take_profit_pct', 0):.1f}%"
            lines.append(f"{rank:>4}  {sl:>5}  {tp:>5}  {sharpe:>7}  {maxdd:>7}  "
                          f"{wr:>7}  {pf:>5}  {awl:>7}  {tc:>6}  {ret:>8}")
        elif sweep_mode == "atr-stops":
            sl_atr = f"{p.get('stop_loss_atr_mult', 0):.1f}"
            tp_atr = f"{p.get('take_profit_atr_mult', 0):.1f}"
            lines.append(f"{rank:>4}  {sl_atr:>6}  {tp_atr:>6}  {sharpe:>7}  {maxdd:>7}  "
                          f"{wr:>7}  {pf:>5}  {awl:>7}  {tc:>6}  {ret:>8}")
        elif sweep_mode == "trailing":
            act = f"{p.get('trailing_stop_activation_pct', 0):.1f}%"
            dist = f"{p.get('trailing_stop_distance_pct', 0):.1f}%"
            lines.append(f"{rank:>4}  {act:>5}  {dist:>5}  {sharpe:>7}  {maxdd:>7}  "
                          f"{wr:>7}  {pf:>5}  {awl:>7}  {tc:>6}  {ret:>8}")
        elif sweep_mode == "sizing":
            mp = f"{p.get('max_position_pct', 0):.0f}%"
            rk = f"{p.get('base_risk_per_trade_pct', 0):.1f}%"
            lines.append(f"{rank:>4}  {mp:>7}  {rk:>5}  {sharpe:>7}  {maxdd:>7}  "
                          f"{wr:>7}  {pf:>5}  {awl:>7}  {tc:>6}  {ret:>8}")
        else:
            lines.append(f"{rank:>4}  {sharpe:>7}  {maxdd:>7}  {wr:>7}  {pf:>5}  "
                          f"{awl:>7}  {tc:>6}  {ret:>8}")

    # "YOU ARE HERE" marker for current params
    if current_params is not None and ranked_results:
        current_result = None
        for i, r in enumerate(ranked_results):
            if r.get("is_current", False):
                current_result = (i + 1, r)
                break

        if current_result:
            rank_num, r = current_result
            m = r["metrics"]
            ret = f"{m['total_return'] * 100:+.1f}%"
            if rank_num > top_n:
                lines.append(f" ...")
            lines.append(
                f" CURRENT (rank {rank_num}/{total}): "
                f"Sharpe={m['sharpe']:.2f}  MaxDD={m['max_drawdown'] * 100:.1f}%  "
                f"WR={m['win_rate'] * 100:.0f}%  Return={ret}  "
                f"<-- YOU ARE HERE"
            )

    lines.append("")
    return "\n".join(lines)


def save_results(
    ranked_results: list[dict[str, Any]],
    sweep_mode: str,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Save results as CSV + best params JSON.

    Returns dict with file paths: {"csv": ..., "best_params": ...}
    """
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=None).strftime("%Y%m%d_%H%M%S")
    paths: dict[str, str] = {}

    # CSV with all results
    csv_path = out_dir / f"{sweep_mode}_{timestamp}.csv"
    if ranked_results:
        fieldnames = ["rank"]
        # Collect all param keys and metric keys
        param_keys = sorted(set().union(*(r.get("params", {}).keys() for r in ranked_results)))
        metric_keys = sorted(set().union(*(r.get("metrics", {}).keys() for r in ranked_results)) - {"exit_reasons"})
        fieldnames.extend([f"param_{k}" for k in param_keys])
        fieldnames.extend(metric_keys)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, r in enumerate(ranked_results):
                row: dict[str, Any] = {"rank": i + 1}
                for k in param_keys:
                    row[f"param_{k}"] = r.get("params", {}).get(k, "")
                for k in metric_keys:
                    row[k] = r.get("metrics", {}).get(k, "")
                writer.writerow(row)

        paths["csv"] = str(csv_path)

    # Best params JSON
    if ranked_results:
        best = ranked_results[0]
        best_path = out_dir / "best_params.json"
        with open(best_path, "w") as f:
            json.dump(best["params"], f, indent=2)
        paths["best_params"] = str(best_path)

    return paths
