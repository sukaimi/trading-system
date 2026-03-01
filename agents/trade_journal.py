"""Trade Journal Agent — Tier 1 (DeepSeek V3.2).

Records every trade decision — entries, exits, no-trades, killed trades.
Persists to data/trade_journal.json.
See TRADING_AGENT_PRD.md Section 3.6.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from core.llm_client import LLMClient
from core.logger import setup_logger
from core.schemas import (
    DevilsVerdict,
    JournalEntry,
    MarketContext,
    OrderConfirmation,
    SignalAlert,
    TradeOutcome,
    TradeThesis,
)

log = setup_logger("trading.trade_journal")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
JOURNAL_FILE = os.path.join(DATA_DIR, "trade_journal.json")


class TradeJournal:
    """Quantitative trade documentarian."""

    def __init__(self, llm_client: LLMClient | None = None):
        self._llm = llm_client or LLMClient()

    def record_entry(
        self,
        thesis: TradeThesis,
        verdict: DevilsVerdict,
        confirmation: OrderConfirmation,
        market_context: dict[str, Any] | None = None,
    ) -> JournalEntry:
        """Record a trade entry in the journal."""
        trade_id = self._generate_trade_id()

        mc = MarketContext()
        if market_context:
            mc = MarketContext(
                dxy=market_context.get("dxy", 0.0),
                vix=market_context.get("vix", 0.0),
                btc_rsi_14=market_context.get("btc_rsi_14", 0.0),
                regime=market_context.get("regime", ""),
            )

        entry = JournalEntry(
            trade_id=trade_id,
            timestamp_open=datetime.utcnow(),
            asset=thesis.asset,
            direction=thesis.direction,
            entry_price=confirmation.fill_price,
            position_size_pct=thesis.suggested_position_pct,
            stop_loss_price=None,
            take_profit_price=None,
            thesis_summary=thesis.thesis,
            thesis_confidence_original=thesis.confidence,
            thesis_confidence_after_devil=verdict.confidence_adjusted,
            devil_modifications=verdict.modifications,
            news_trigger=thesis.triggering_alert_id,
            confirming_signals=[
                s for s, present in [
                    ("fundamental", thesis.confirming_signals.fundamental.present),
                    ("technical", thesis.confirming_signals.technical.present),
                    ("cross_asset", thesis.confirming_signals.cross_asset.present),
                ] if present
            ],
            market_context=mc,
        )

        self._append_entry(entry.model_dump(mode="json"))
        log.info("Recorded trade entry: %s %s %s", trade_id, thesis.asset.value, thesis.direction.value)
        return entry

    def record_exit(
        self, trade_id: str, exit_data: dict[str, Any]
    ) -> JournalEntry | None:
        """Record a trade exit with outcome and lessons."""
        entries = self._load_journal()

        for i, entry_dict in enumerate(entries):
            if entry_dict.get("trade_id") == trade_id:
                entry_dict["timestamp_close"] = datetime.utcnow().isoformat()
                entry_dict["exit_price"] = exit_data.get("exit_price", 0.0)

                outcome = {
                    "pnl_usd": exit_data.get("pnl_usd", 0.0),
                    "pnl_pct": exit_data.get("pnl_pct", 0.0),
                    "thesis_correct": exit_data.get("thesis_correct"),
                    "exit_reason": exit_data.get("exit_reason", ""),
                    "hold_duration_hours": exit_data.get("hold_duration_hours", 0.0),
                }
                entry_dict["outcome"] = outcome

                # Generate lessons via DeepSeek
                lessons = self._generate_lessons(entry_dict)
                entry_dict["lessons"] = lessons

                entries[i] = entry_dict
                self._save_journal(entries)
                log.info("Recorded trade exit: %s", trade_id)
                return JournalEntry.model_validate(entry_dict)

        log.warning("Trade %s not found in journal", trade_id)
        return None

    def record_no_trade(
        self, signal: SignalAlert, reasoning: str
    ) -> dict[str, Any]:
        """Record a no-trade decision."""
        record = {
            "type": "no_trade",
            "timestamp": datetime.utcnow().isoformat(),
            "asset": signal.asset.value,
            "signal_strength": signal.signal_strength,
            "headline": signal.headline,
            "reasoning": reasoning,
        }
        self._append_entry(record)
        log.info("Recorded no-trade decision for %s", signal.asset.value)
        return record

    def record_killed_trade(
        self, thesis: TradeThesis, verdict: DevilsVerdict
    ) -> dict[str, Any]:
        """Record a trade killed by Devil's Advocate."""
        record = {
            "type": "killed_trade",
            "timestamp": datetime.utcnow().isoformat(),
            "asset": thesis.asset.value,
            "direction": thesis.direction.value,
            "thesis_summary": thesis.thesis,
            "confidence": thesis.confidence,
            "verdict": verdict.verdict.value,
            "flags_raised": verdict.flags_raised,
            "fatal_flaws": verdict.fatal_flaws,
            "final_reasoning": verdict.final_reasoning,
        }
        self._append_entry(record)
        log.info("Recorded killed trade for %s", thesis.asset.value)
        return record

    def assemble_weekly_package(
        self,
        week_ending: str,
        portfolio_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the weekly input package for the Strategist."""
        entries = self._load_journal()

        # Filter entries for this week
        week_entries = [
            e for e in entries
            if e.get("timestamp_open", e.get("timestamp", "")) <= week_ending
        ]

        # Compute stats
        trades = [e for e in week_entries if e.get("trade_id")]
        closed = [e for e in trades if e.get("outcome", {}).get("pnl_usd") is not None]
        wins = [e for e in closed if (e.get("outcome", {}).get("pnl_usd", 0) or 0) > 0]
        losses = [e for e in closed if (e.get("outcome", {}).get("pnl_usd", 0) or 0) < 0]
        no_trades = [e for e in week_entries if e.get("type") == "no_trade"]
        killed = [e for e in week_entries if e.get("type") == "killed_trade"]

        win_pcts = [(e.get("outcome", {}).get("pnl_pct", 0) or 0) for e in wins]
        loss_pcts = [(e.get("outcome", {}).get("pnl_pct", 0) or 0) for e in losses]

        package: dict[str, Any] = {
            "week_ending": week_ending,
            "trade_summary": {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "no_trades": len(no_trades),
                "killed_trades": len(killed),
                "win_rate": len(wins) / len(closed) if closed else 0.0,
                "avg_win_pct": sum(win_pcts) / len(win_pcts) if win_pcts else 0.0,
                "avg_loss_pct": sum(loss_pcts) / len(loss_pcts) if loss_pcts else 0.0,
            },
            "full_trade_journal": week_entries,
        }

        # Portfolio summary (optional — provided when called by WeeklyStrategist)
        if portfolio_state:
            equity = portfolio_state.get("equity", 0.0)
            initial = portfolio_state.get("initial_capital", equity)
            package["portfolio_summary"] = {
                "equity": equity,
                "initial_capital": initial,
                "return_pct": ((equity - initial) / initial * 100) if initial else 0.0,
                "drawdown_from_peak_pct": portfolio_state.get("drawdown_from_peak_pct", 0.0),
                "open_positions": len(portfolio_state.get("open_positions", [])),
            }

        # Current strategy params
        package["current_strategy_params"] = self._load_all_params()
        package["current_risk_params"] = self._load_risk_params()

        return package

    def _load_all_params(self) -> dict[str, Any]:
        """Load all agent parameter files for the weekly package."""
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        params: dict[str, Any] = {}
        for name in ("news_scout", "market_analyst", "devils_advocate"):
            path = os.path.join(config_dir, f"{name}_params.json")
            try:
                with open(path) as f:
                    params[name] = json.load(f)
            except Exception:
                params[name] = {}
        return params

    def _load_risk_params(self) -> dict[str, Any]:
        """Load risk parameters for the weekly package."""
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        path = os.path.join(config_dir, "risk_params.json")
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _generate_lessons(self, entry: dict[str, Any]) -> str:
        """Use DeepSeek to generate lessons from a closed trade."""
        prompt = (
            f"Analyze this completed trade and write 1-2 sentences of lessons learned.\n"
            f"Trade: {json.dumps(entry, default=str)}\n"
            f"Focus on: Was the thesis correct? What worked? What didn't?"
        )
        result = self._llm.call_deepseek(prompt)
        if isinstance(result, dict):
            return result.get("lessons", result.get("message", ""))
        return str(result)

    def _generate_trade_id(self) -> str:
        now = datetime.utcnow()
        return f"trade_{now.strftime('%Y%m%d_%H%M%S')}"

    def _load_journal(self) -> list[dict[str, Any]]:
        if not os.path.exists(JOURNAL_FILE):
            return []
        try:
            with open(JOURNAL_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []

    def _save_journal(self, entries: list[dict[str, Any]]) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(JOURNAL_FILE, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    def _append_entry(self, entry: dict[str, Any]) -> None:
        entries = self._load_journal()
        entries.append(entry)
        self._save_journal(entries)
