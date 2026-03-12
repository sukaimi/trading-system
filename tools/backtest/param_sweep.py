"""Parameter sweeper — generates parameter combinations and runs backtests in parallel.

Supports sweep modes: stops, atr-stops, trailing, sizing, holding, full.
Uses ProcessPoolExecutor for true parallel execution of CPU-bound backtests.
"""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import pandas as pd

from tools.backtest.engine import BacktestEngine, load_risk_params
from tools.backtest.report import calculate_metrics


def _frange(start: float, stop: float, step: float) -> list[float]:
    """Generate a range of floats, inclusive of stop."""
    result = []
    val = start
    while val <= stop + step / 10:  # Small epsilon for float rounding
        result.append(round(val, 4))
        val += step
    return result


def _run_single_backtest(args: tuple) -> dict[str, Any]:
    """Run a single backtest — top-level function for ProcessPoolExecutor pickling."""
    params, data_dict, initial_capital, use_friction = args
    engine = BacktestEngine(params=params, initial_capital=initial_capital, use_friction=use_friction)

    # Convert data_dict back to DataFrames (they are passed as dicts for pickling)
    data = {}
    for sym, df_dict in data_dict.items():
        data[sym] = pd.DataFrame(df_dict)

    result = engine.run(data)
    metrics = calculate_metrics(result)
    return {
        "params": params,
        "metrics": metrics,
        "result": result,
    }


class ParameterSweeper:
    """Generate parameter combinations and run backtests."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        use_friction: bool = True,
        max_workers: int | None = None,
    ):
        self.initial_capital = initial_capital
        self.use_friction = use_friction
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
        self._baseline = load_risk_params()

    def sweep(
        self,
        mode: str,
        data: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Run a parameter sweep and return ranked results.

        Args:
            mode: Sweep mode (stops, atr-stops, trailing, sizing, holding, full).
            data: Dict mapping symbol -> OHLCV DataFrame.

        Returns:
            List of result dicts sorted by Sharpe ratio (descending).
        """
        if mode == "full":
            return self._full_sweep(data)

        param_combos = self._generate_params(mode)
        results = self._run_parallel(param_combos, data)

        # Mark current params
        self._mark_current(results, mode)

        # Add n_assets for reporting
        n_assets = len(data)
        for r in results:
            r["n_assets"] = n_assets

        return results

    def _generate_params(self, mode: str) -> list[dict[str, Any]]:
        """Generate parameter combinations for a given sweep mode."""
        base = dict(self._baseline)

        if mode == "stops":
            return self._gen_stops(base)
        elif mode == "atr-stops":
            return self._gen_atr_stops(base)
        elif mode == "trailing":
            return self._gen_trailing(base)
        elif mode == "sizing":
            return self._gen_sizing(base)
        elif mode == "holding":
            return self._gen_holding(base)
        else:
            raise ValueError(f"Unknown sweep mode: {mode}")

    def _gen_stops(self, base: dict) -> list[dict]:
        sl_range = _frange(1.0, 8.0, 0.5)
        tp_range = _frange(3.0, 15.0, 1.0)
        combos = []
        for sl, tp in itertools.product(sl_range, tp_range):
            p = dict(base)
            p["default_stop_loss_pct"] = sl
            p["default_take_profit_pct"] = tp
            p["use_atr_stops"] = False  # Test fixed % stops
            combos.append(p)
        return combos

    def _gen_atr_stops(self, base: dict) -> list[dict]:
        sl_atr_range = _frange(1.0, 4.0, 0.5)
        tp_atr_range = _frange(2.0, 6.0, 0.5)
        combos = []
        for sl_atr, tp_atr in itertools.product(sl_atr_range, tp_atr_range):
            p = dict(base)
            p["stop_loss_atr_mult"] = sl_atr
            p["take_profit_atr_mult"] = tp_atr
            p["use_atr_stops"] = True
            combos.append(p)
        return combos

    def _gen_trailing(self, base: dict) -> list[dict]:
        act_range = _frange(1.0, 5.0, 0.5)
        dist_range = _frange(0.5, 3.0, 0.5)
        combos = []
        for act, dist in itertools.product(act_range, dist_range):
            p = dict(base)
            p["trailing_stop_activation_pct"] = act
            p["trailing_stop_distance_pct"] = dist
            combos.append(p)
        return combos

    def _gen_sizing(self, base: dict) -> list[dict]:
        pos_range = _frange(3.0, 10.0, 1.0)
        risk_range = _frange(0.5, 3.0, 0.5)
        combos = []
        for pos, risk in itertools.product(pos_range, risk_range):
            p = dict(base)
            p["max_position_pct"] = pos
            p["base_risk_per_trade_pct"] = risk
            combos.append(p)
        return combos

    def _gen_holding(self, base: dict) -> list[dict]:
        # Holding period in bars (days for daily data): 24h=1d, 168h=7d
        hold_range = [1, 2, 3, 4, 5, 6, 7]  # 1-7 days
        sl_range = _frange(1.0, 8.0, 0.5)
        combos = []
        for hold, sl in itertools.product(hold_range, sl_range):
            p = dict(base)
            p["max_holding_bars"] = hold
            p["default_stop_loss_pct"] = sl
            combos.append(p)
        return combos

    def _full_sweep(self, data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        """Run top-N from each sub-sweep combined.

        1. Run stops sweep, take top-5 SL/TP combos
        2. Run trailing sweep, take top-5 trailing combos
        3. Run sizing sweep, take top-3 sizing combos
        4. Cartesian product of top combos: 5 x 5 x 3 = 75 full backtests
        """
        # Sub-sweeps
        stops_results = self.sweep("stops", data)
        trailing_results = self.sweep("trailing", data)
        sizing_results = self.sweep("sizing", data)

        top_stops = stops_results[:5]
        top_trailing = trailing_results[:5]
        top_sizing = sizing_results[:3]

        # Combine top params
        combos = []
        for s, t, z in itertools.product(top_stops, top_trailing, top_sizing):
            p = dict(self._baseline)
            # Apply stops params
            p["default_stop_loss_pct"] = s["params"]["default_stop_loss_pct"]
            p["default_take_profit_pct"] = s["params"]["default_take_profit_pct"]
            # Apply trailing params
            p["trailing_stop_activation_pct"] = t["params"]["trailing_stop_activation_pct"]
            p["trailing_stop_distance_pct"] = t["params"]["trailing_stop_distance_pct"]
            # Apply sizing params
            p["max_position_pct"] = z["params"]["max_position_pct"]
            p["base_risk_per_trade_pct"] = z["params"]["base_risk_per_trade_pct"]
            combos.append(p)

        results = self._run_parallel(combos, data)
        self._mark_current(results, "full")

        n_assets = len(data)
        for r in results:
            r["n_assets"] = n_assets

        return results

    def _run_parallel(
        self,
        param_combos: list[dict[str, Any]],
        data: dict[str, pd.DataFrame],
    ) -> list[dict[str, Any]]:
        """Run backtests in parallel and return sorted results."""
        # Convert DataFrames to dicts for pickling across processes
        data_dict = {sym: df.to_dict() for sym, df in data.items()}

        args_list = [
            (params, data_dict, self.initial_capital, self.use_friction)
            for params in param_combos
        ]

        results: list[dict[str, Any]] = []

        # Use ProcessPoolExecutor for CPU-bound work
        # Fall back to sequential if only 1 worker or small batch
        if self.max_workers <= 1 or len(args_list) <= 2:
            for args in args_list:
                results.append(_run_single_backtest(args))
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_run_single_backtest, a): i for i, a in enumerate(args_list)}
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        idx = futures[future]
                        results.append({
                            "params": param_combos[idx],
                            "metrics": {"sharpe": -999, "total_return": -999, "trade_count": 0,
                                        "max_drawdown": 0, "win_rate": 0, "profit_factor": 0,
                                        "avg_win_loss_ratio": 0, "expectancy": 0, "avg_bars_held": 0,
                                        "exit_reasons": {}, "gross_profit": 0, "gross_loss": 0},
                            "result": {"error": str(e)},
                        })

        # Sort by Sharpe ratio descending
        results.sort(key=lambda r: r["metrics"].get("sharpe", -999), reverse=True)
        return results

    def _mark_current(self, results: list[dict[str, Any]], mode: str) -> None:
        """Mark the result that matches current production params."""
        for r in results:
            p = r["params"]
            is_current = True

            if mode == "stops" or mode == "holding":
                is_current = (
                    p.get("default_stop_loss_pct") == self._baseline.get("default_stop_loss_pct")
                    and p.get("default_take_profit_pct") == self._baseline.get("default_take_profit_pct")
                )
                if mode == "holding":
                    # Also check holding bars (convert 72h to 3 days for daily bars)
                    is_current = is_current and p.get("max_holding_bars") == 3
            elif mode == "atr-stops":
                is_current = (
                    p.get("stop_loss_atr_mult") == self._baseline.get("stop_loss_atr_mult")
                    and p.get("take_profit_atr_mult") == self._baseline.get("take_profit_atr_mult")
                )
            elif mode == "trailing":
                is_current = (
                    p.get("trailing_stop_activation_pct") == self._baseline.get("trailing_stop_activation_pct")
                    and p.get("trailing_stop_distance_pct") == self._baseline.get("trailing_stop_distance_pct")
                )
            elif mode == "sizing":
                is_current = (
                    p.get("max_position_pct") == self._baseline.get("max_position_pct")
                    and p.get("base_risk_per_trade_pct") == self._baseline.get("base_risk_per_trade_pct")
                )

            r["is_current"] = is_current
