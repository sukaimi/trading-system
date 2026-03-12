"""CLI entry point for the backtest module.

Usage:
    python -m tools.backtest --sweep stops --period 2y
    python -m tools.backtest --sweep atr-stops --assets AAPL,NVDA,SPY --period 6mo
    python -m tools.backtest --sweep full --period 2y --capital 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.backtest.data_loader import HistoricalDataLoader, CORE_ASSETS
from tools.backtest.engine import load_risk_params
from tools.backtest.param_sweep import ParameterSweeper
from tools.backtest.report import format_summary_table, save_results


VALID_SWEEP_MODES = ["stops", "atr-stops", "trailing", "sizing", "holding", "full"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.backtest",
        description="Backtest risk parameter optimization against historical data.",
    )
    parser.add_argument(
        "--sweep", required=True, choices=VALID_SWEEP_MODES,
        help="Sweep mode: stops, atr-stops, trailing, sizing, holding, full",
    )
    parser.add_argument(
        "--period", default="2y",
        help="yfinance period string (default: 2y)",
    )
    parser.add_argument(
        "--interval", default="1d",
        help="yfinance interval string (default: 1d)",
    )
    parser.add_argument(
        "--capital", type=float, default=10000.0,
        help="Initial portfolio equity in USD (default: 10000)",
    )
    parser.add_argument(
        "--assets", default=None,
        help="Comma-separated asset symbols (default: core 14)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: data/backtest_results/)",
    )
    parser.add_argument(
        "--no-friction", action="store_true",
        help="Disable trading friction simulation",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate equity curve plot for best combo",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trade details for best combo",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Parse assets
    symbols = args.assets.split(",") if args.assets else CORE_ASSETS

    print(f"Loading historical data for {len(symbols)} assets ({args.period}, {args.interval})...")

    # Load data
    loader = HistoricalDataLoader()
    data = loader.get_universe(symbols, period=args.period, interval=args.interval)

    if not data:
        print("ERROR: No data loaded. Check asset symbols and network connection.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(data)} assets: {', '.join(sorted(data.keys()))}")
    print(f"Running {args.sweep} sweep with {args.capital:.0f} USD capital...")

    # Run sweep
    sweeper = ParameterSweeper(
        initial_capital=args.capital,
        use_friction=not args.no_friction,
        max_workers=args.workers,
    )

    results = sweeper.sweep(args.sweep, data)

    if not results:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    current_params = load_risk_params()
    table = format_summary_table(results, args.sweep, current_params=current_params)
    print(table)

    # Save results
    saved = save_results(results, args.sweep, output_dir=args.output)
    for label, path in saved.items():
        print(f"{label} saved to: {path}")

    # Verbose: print per-trade details for best combo
    if args.verbose and results:
        best = results[0]
        print(f"\n=== BEST COMBO: Per-Trade Details ===")
        trades = best.get("result", {}).get("closed_trades", [])
        for t in trades[:50]:  # Cap at 50 trades
            print(f"  {t['asset']:>5} {t['direction']:>5} "
                  f"entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} "
                  f"pnl={t['pnl']:+.2f} ({t['pnl_pct'] * 100:+.1f}%) "
                  f"held={t['bars_held']}d reason={t['exit_reason']}")
        if len(trades) > 50:
            print(f"  ... ({len(trades) - 50} more trades)")

    # Plot equity curve for best combo
    if args.plot and results:
        _plot_equity(results[0], args.sweep, args.output)

    # Print best params as JSON
    if results:
        print(f"\nBest params (Sharpe={results[0]['metrics']['sharpe']:.4f}):")
        best_params = {k: v for k, v in results[0]["params"].items()
                       if k in ("default_stop_loss_pct", "default_take_profit_pct",
                                "stop_loss_atr_mult", "take_profit_atr_mult",
                                "trailing_stop_activation_pct", "trailing_stop_distance_pct",
                                "max_position_pct", "base_risk_per_trade_pct",
                                "max_holding_bars")}
        print(json.dumps(best_params, indent=2))


def _plot_equity(result: dict, sweep_mode: str, output_dir: str | None) -> None:
    """Generate equity curve plot if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    equity_curve = result.get("result", {}).get("equity_curve", [])
    if not equity_curve:
        return

    dates = [snap["date"] for snap in equity_curve]
    equities = [snap["equity"] for snap in equity_curve]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(equities)), equities, linewidth=1)
    ax.set_title(f"Equity Curve — Best {sweep_mode} sweep (Sharpe={result['metrics']['sharpe']:.2f})")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)

    from datetime import datetime
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "data" / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    plot_path = out_dir / f"equity_{sweep_mode}_{ts}.png"
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Equity curve saved to: {plot_path}")


if __name__ == "__main__":
    main()
