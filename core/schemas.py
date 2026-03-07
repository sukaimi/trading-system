"""Pydantic v2 schemas for all inter-agent communication.

Every message exchanged between agents conforms to one of these models.
See TRADING_AGENT_PRD.md Section 4 for the full communication map.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


def _validate_asset(v: str) -> str:
    """Validate asset symbol against the dynamic registry."""
    from core.asset_registry import validate_asset  # lazy import to avoid circular
    return validate_asset(v)


# ── Enums ──────────────────────────────────────────────────────────────

class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Urgency(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class SignalCategory(str, Enum):
    CENTRAL_BANK = "central_bank"
    GEOPOLITICAL = "geopolitical"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    MACRO = "macro"
    CRYPTO_SPECIFIC = "crypto_specific"
    PRECIOUS_METALS = "precious_metals"
    EQUITY = "equity"


class Verdict(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_WITH_MODIFICATION = "APPROVED_WITH_MODIFICATION"
    KILLED = "KILLED"


class TimeHorizon(str, Enum):
    SHORT = "1-3 days"
    MEDIUM = "1 week"
    SWING = "swing"


class CircuitBreakerAction(str, Enum):
    CLOSE_ALL = "CLOSE_ALL"
    CLOSE_LOSING = "CLOSE_LOSING_POSITIONS"
    HOLD = "HOLD"
    REDUCE_ALL = "REDUCE_ALL"
    HEDGE = "HEDGE"


class OrderStatus(str, Enum):
    FILLED = "Filled"
    PARTIAL = "PartiallyFilled"
    SUBMITTED = "Submitted"
    CANCELLED = "Cancelled"
    ERROR = "Error"


# ── Signal Alert (News Scout → Market Analyst) ────────────────────────

class SignalAlert(BaseModel):
    type: str = "signal_alert"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "news_scout"
    signal_id: str = ""
    asset: str
    signal_strength: float = Field(ge=0.0, le=1.0)

    @field_validator("asset")
    @classmethod
    def check_asset(cls, v: str) -> str:
        return _validate_asset(v)
    headline: str = Field(max_length=100)
    sentiment: Sentiment
    category: SignalCategory
    new_information: str
    raw_sources: list[str] = Field(default_factory=list)
    urgency: Urgency
    already_priced_in: bool = False
    confidence_in_classification: float = Field(ge=0.0, le=1.0)


# ── Confirming Signal (sub-model for TradeThesis) ─────────────────────

class ConfirmingSignal(BaseModel):
    present: bool
    description: str = ""


class ConfirmingSignals(BaseModel):
    fundamental: ConfirmingSignal = ConfirmingSignal(present=False)
    technical: ConfirmingSignal = ConfirmingSignal(present=False)
    cross_asset: ConfirmingSignal = ConfirmingSignal(present=False)
    chart_pattern: ConfirmingSignal = ConfirmingSignal(present=False)


# ── Trade Thesis (Market Analyst → Devil's Advocate) ──────────────────

class TradeThesis(BaseModel):
    type: str = "trade_thesis"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "market_analyst"
    model_used: str = "deepseek"
    escalated: bool = False
    asset: str

    @field_validator("asset")
    @classmethod
    def check_asset(cls, v: str) -> str:
        return _validate_asset(v)
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    thesis: str
    confirming_signals: ConfirmingSignals = Field(default_factory=ConfirmingSignals)
    entry_trigger: str = ""
    invalidation_level: str = ""
    time_horizon: TimeHorizon = TimeHorizon.SHORT
    suggested_position_pct: float = Field(ge=0.0, le=100.0)
    risk_reward_ratio: str = ""
    supporting_data: dict = Field(default_factory=dict)
    what_could_go_wrong: list[str] = Field(default_factory=list)
    triggering_alert_id: str = ""


# ── Devil's Verdict sub-models ────────────────────────────────────────

class CrowdedTradeCheck(BaseModel):
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""


class AssumptionStressTest(BaseModel):
    critical_assumption: str = ""
    if_wrong: str = ""
    worst_case_loss_usd: float = 0.0


class TimingCheck(BaseModel):
    flag: bool = False
    reasoning: str = ""


class CorrelationTrap(BaseModel):
    flag: bool = False
    post_trade_concentration: float = 0.0


class BaseRateCheck(BaseModel):
    available: bool = False
    historical_win_rate: float = 0.0
    sample_size: int = 0


class SecondOrderEffects(BaseModel):
    identified: list[str] = Field(default_factory=list)
    severity: str = "low"


class SizeCheck(BaseModel):
    proposed_pct: float = 0.0
    recommended_pct: float = 0.0


class DevilsChallenges(BaseModel):
    crowded_trade: CrowdedTradeCheck = Field(default_factory=CrowdedTradeCheck)
    assumption_stress_test: AssumptionStressTest = Field(default_factory=AssumptionStressTest)
    timing: TimingCheck = Field(default_factory=TimingCheck)
    correlation_trap: CorrelationTrap = Field(default_factory=CorrelationTrap)
    base_rate: BaseRateCheck = Field(default_factory=BaseRateCheck)
    second_order_effects: SecondOrderEffects = Field(default_factory=SecondOrderEffects)
    size_check: SizeCheck = Field(default_factory=SizeCheck)


# ── Devil's Verdict (Devil's Advocate → Risk Manager) ─────────────────

class DevilsVerdict(BaseModel):
    type: str = "devils_verdict"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "devils_advocate"
    original_thesis_id: str
    verdict: Verdict
    challenges: DevilsChallenges = Field(default_factory=DevilsChallenges)
    flags_raised: int = 0
    fatal_flaws: list[str] = Field(default_factory=list)
    modifications: list[str] = Field(default_factory=list)
    confidence_adjusted: float = Field(ge=0.0, le=1.0)
    final_reasoning: str = ""


# ── Execution Order (Risk Manager → Executor) ─────────────────────────

class ExecutionOrder(BaseModel):
    type: str = "execution_order"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "risk_manager"
    thesis_id: str
    asset: str

    @field_validator("asset")
    @classmethod
    def check_asset(cls, v: str) -> str:
        return _validate_asset(v)
    direction: Direction
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = Field(ge=0.0, le=100.0)


# ── Order Confirmation (Executor → Trade Journal) ─────────────────────

class OrderConfirmation(BaseModel):
    type: str = "order_confirmation"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    order_id: int = 0
    asset: str

    @field_validator("asset")
    @classmethod
    def check_asset(cls, v: str) -> str:
        return _validate_asset(v)
    direction: Direction
    quantity: float
    fill_price: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTED
    thesis_id: str = ""


# ── Order Error (Executor → Risk Manager + Telegram) ──────────────────

class OrderError(BaseModel):
    type: str = "order_error"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: str
    thesis_id: str = ""


# ── Market Context (sub-model for JournalEntry) ──────────────────────

class MarketContext(BaseModel):
    dxy: float = 0.0
    vix: float = 0.0
    btc_rsi_14: float = 0.0
    regime: str = ""


class TradeOutcome(BaseModel):
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    thesis_correct: Optional[bool] = None
    exit_reason: Optional[str] = None
    hold_duration_hours: Optional[float] = None
    mae_pct: Optional[float] = None
    mfe_pct: Optional[float] = None


# ── Journal Entry (Trade Journal → Storage) ───────────────────────────

class JournalEntry(BaseModel):
    trade_id: str
    timestamp_open: datetime = Field(default_factory=datetime.utcnow)
    timestamp_close: Optional[datetime] = None
    asset: str

    @field_validator("asset")
    @classmethod
    def check_asset(cls, v: str) -> str:
        return _validate_asset(v)
    direction: Direction
    entry_price: float
    exit_price: Optional[float] = None
    position_size_usd: float = 0.0
    position_size_pct: float = 0.0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    thesis_summary: str = ""
    thesis_confidence_original: float = 0.0
    thesis_confidence_after_devil: float = 0.0
    devil_modifications: list[str] = Field(default_factory=list)
    news_trigger: str = ""
    confirming_signals: list[str] = Field(default_factory=list)
    market_context: MarketContext = Field(default_factory=MarketContext)
    outcome: TradeOutcome = Field(default_factory=TradeOutcome)
    lessons: Optional[str] = None


# ── Weekly Strategy Directive (Strategist → Self-Optimizer) ───────────

class ParameterChange(BaseModel):
    target: str
    parameter: str
    old_value: float | str
    new_value: float | str
    reason: str


class WeeklyDirective(BaseModel):
    type: str = "weekly_strategy_directive"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "weekly_strategist"
    model: str = "sonnet-4.6"
    week_reviewed: str
    assessment: dict = Field(default_factory=dict)
    parameter_changes: list[ParameterChange] = Field(default_factory=list)
    next_week_focus: list[str] = Field(default_factory=list)
    risk_adjustments: dict = Field(default_factory=dict)


# ── Circuit Breaker Decision ──────────────────────────────────────────

class CircuitBreakerDecision(BaseModel):
    type: str = "circuit_breaker_decision"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "circuit_breaker"
    model: str = "sonnet-4.6"
    triggers_fired: list[str] = Field(default_factory=list)
    decision: CircuitBreakerAction
    positions_to_close: list[str] = Field(default_factory=list)
    positions_to_keep: list[str] = Field(default_factory=list)
    reasoning: str = ""
    resume_conditions: str = ""
    notify_owner: bool = True
    telegram_message: str = ""


# ── Heartbeat Status ─────────────────────────────────────────────────

class HeartbeatStatus(BaseModel):
    type: str = "heartbeat_status"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    checks: dict[str, bool] = Field(default_factory=dict)
    all_healthy: bool = True
    failures: list[str] = Field(default_factory=list)


# ── Optimization Applied (Self-Optimizer → Log) ──────────────────────

class OptimizationChange(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str
    parameter: str
    old_value: float | str
    new_value: float | str
    reason: str
    directive_source: str = ""
    version: int = 0


class OptimizationApplied(BaseModel):
    type: str = "optimization_applied"
    version: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    directive_week: str
    changes: list[OptimizationChange] = Field(default_factory=list)
    portfolio_value_at_change: float = 0.0
    cumulative_return_pct: float = 0.0
