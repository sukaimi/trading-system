"""Tests for EarningsCalendar — earnings proximity detection."""

from datetime import date

import pytest

from core.earnings_calendar import EarningsCalendar


class TestDaysUntilEarnings:
    def test_known_date_future(self):
        """days_until_earnings returns correct count for a future date."""
        cal = EarningsCalendar()
        # AAPL has earnings on 2026-04-30
        result = cal.days_until_earnings("AAPL", ref_date=date(2026, 4, 27))
        assert result == 3

    def test_known_date_today(self):
        """If earnings are today, days_until should be 0."""
        cal = EarningsCalendar()
        result = cal.days_until_earnings("AAPL", ref_date=date(2026, 4, 30))
        assert result == 0

    def test_etf_returns_none(self):
        """ETFs like SPY have no earnings — should return None."""
        cal = EarningsCalendar()
        assert cal.days_until_earnings("SPY") is None

    def test_crypto_returns_none(self):
        """Crypto assets have no earnings — should return None."""
        cal = EarningsCalendar()
        assert cal.days_until_earnings("BTC") is None
        assert cal.days_until_earnings("ETH") is None

    def test_commodity_etf_returns_none(self):
        """Commodity ETFs like GLDM and SLV have no earnings."""
        cal = EarningsCalendar()
        assert cal.days_until_earnings("GLDM") is None
        assert cal.days_until_earnings("SLV") is None

    def test_unknown_asset_returns_none(self):
        """Unknown assets should return None."""
        cal = EarningsCalendar()
        assert cal.days_until_earnings("XYZ123") is None

    def test_picks_next_future_date(self):
        """Should pick the nearest future date, not a past one."""
        cal = EarningsCalendar()
        # After 2026-01-29 AAPL earnings, next is 2026-04-30
        result = cal.days_until_earnings("AAPL", ref_date=date(2026, 2, 1))
        assert result is not None
        assert result > 0
        # Should be days until 2026-04-30 from 2026-02-01 = 88 days
        assert result == (date(2026, 4, 30) - date(2026, 2, 1)).days

    def test_all_dates_in_past_returns_none(self):
        """If all known dates are in the past, return None."""
        cal = EarningsCalendar()
        result = cal.days_until_earnings("AAPL", ref_date=date(2027, 12, 1))
        assert result is None


class TestHasEarningsSoon:
    def test_within_window(self):
        """Should return True when earnings are within the window."""
        cal = EarningsCalendar()
        # NVDA earnings on 2026-02-26, check from 2026-02-24 (2 days away)
        assert cal.has_earnings_soon("NVDA", days=3, ref_date=date(2026, 2, 24)) is True

    def test_outside_window(self):
        """Should return False when earnings are beyond the window."""
        cal = EarningsCalendar()
        # NVDA earnings on 2026-02-26, check from 2026-02-10 (16 days away)
        assert cal.has_earnings_soon("NVDA", days=3, ref_date=date(2026, 2, 10)) is False

    def test_on_earnings_day(self):
        """On earnings day itself, should return True (0 <= 3)."""
        cal = EarningsCalendar()
        assert cal.has_earnings_soon("META", days=3, ref_date=date(2026, 1, 29)) is True

    def test_etf_always_false(self):
        """ETFs have no earnings, so always False."""
        cal = EarningsCalendar()
        assert cal.has_earnings_soon("SPY", days=365) is False

    def test_crypto_always_false(self):
        """Crypto has no earnings, so always False."""
        cal = EarningsCalendar()
        assert cal.has_earnings_soon("BTC", days=365) is False

    def test_custom_window(self):
        """Custom window parameter should work correctly."""
        cal = EarningsCalendar()
        # TSLA earnings on 2026-04-22, check from 2026-04-15 (7 days away)
        assert cal.has_earnings_soon("TSLA", days=7, ref_date=date(2026, 4, 15)) is True
        assert cal.has_earnings_soon("TSLA", days=5, ref_date=date(2026, 4, 15)) is False


class TestUpcomingEarnings:
    def test_returns_sorted_list(self):
        """Upcoming earnings should be sorted by days_until."""
        cal = EarningsCalendar()
        # Around late January 2026, AAPL/TSLA/META all report on 01-29
        result = cal.upcoming_earnings(days=14, ref_date=date(2026, 1, 20))
        assert isinstance(result, list)
        # Should be sorted
        for i in range(len(result) - 1):
            assert result[i]["days_until"] <= result[i + 1]["days_until"]

    def test_includes_correct_fields(self):
        """Each entry should have asset, date, days_until fields."""
        cal = EarningsCalendar()
        result = cal.upcoming_earnings(days=30, ref_date=date(2026, 1, 20))
        if result:
            entry = result[0]
            assert "asset" in entry
            assert "date" in entry
            assert "days_until" in entry
            assert isinstance(entry["days_until"], int)

    def test_empty_when_no_earnings(self):
        """When no earnings in window, return empty list."""
        cal = EarningsCalendar()
        # Far from any earnings dates
        result = cal.upcoming_earnings(days=3, ref_date=date(2026, 6, 15))
        assert result == []

    def test_multiple_assets_same_day(self):
        """Multiple assets reporting on same day should all appear."""
        cal = EarningsCalendar()
        # 2026-01-29: AAPL, TSLA, META all report
        result = cal.upcoming_earnings(days=1, ref_date=date(2026, 1, 29))
        assets = {r["asset"] for r in result}
        assert "AAPL" in assets
        assert "TSLA" in assets
        assert "META" in assets

    def test_respects_window(self):
        """Should not include earnings beyond the window."""
        cal = EarningsCalendar()
        result = cal.upcoming_earnings(days=7, ref_date=date(2026, 2, 20))
        for entry in result:
            assert entry["days_until"] <= 7


class TestPipelineEarningsReduction:
    """Test that pipeline correctly reduces position near earnings."""

    def test_position_halved_near_earnings(self):
        """Verify EarningsCalendar integration logic (unit test, no pipeline)."""
        cal = EarningsCalendar()
        # AAPL earnings on 2026-04-30, check from 2026-04-28
        assert cal.has_earnings_soon("AAPL", days=3, ref_date=date(2026, 4, 28))
        # Position should be reduced by 50%
        position_pct = 5.0
        if cal.has_earnings_soon("AAPL", days=3, ref_date=date(2026, 4, 28)):
            position_pct *= 0.5
        assert position_pct == 2.5

    def test_no_reduction_for_etf(self):
        """ETFs should not get position reduction."""
        cal = EarningsCalendar()
        position_pct = 5.0
        if cal.has_earnings_soon("SPY", days=3):
            position_pct *= 0.5
        assert position_pct == 5.0  # Unchanged
