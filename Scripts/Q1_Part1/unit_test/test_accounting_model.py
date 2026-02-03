"""
Unit Tests for Q1: Balance Sheet Modeling with Accounting Identities
=====================================================================
Tests the deterministic accounting model implementation using pytest.

Tests cover:
1. Accounting identity compliance (A = L + E)
2. Circularity convergence in interest/debt calculations
3. Retained earnings consistency
4. Balance sheet state validation
5. Multi-period forecasting
6. Driver assumptions validation

References:
- Velez-Pareja (2007): Forecasting Financial Statements with No Plugs
- pytest documentation: https://docs.pytest.org/
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deterministic_accounting_model import (
    DriverAssumptions,
    BalanceSheetState,
    DeterministicAccountingModel
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def default_drivers():
    """Default driver assumptions for testing."""
    return DriverAssumptions(
        revenue_growth=0.10,
        cogs_as_pct_revenue=0.60,
        opex_as_pct_revenue=0.20,
        tax_rate=0.21,
        days_sales_outstanding=45,
        days_inventory_outstanding=60,
        days_payable_outstanding=30,
        capex_as_pct_revenue=0.08,
        depreciation_as_pct_ppe=0.10,
        dividend_payout_ratio=0.30,
        target_cash_balance=100_000,
        interest_rate_on_debt=0.05
    )


@pytest.fixture
def initial_balanced_state():
    """Initial balance sheet state that satisfies A = L + E."""
    return BalanceSheetState(
        cash=100_000,
        accounts_receivable=150_000,
        inventory=200_000,
        ppe_net=500_000,
        other_assets=50_000,
        accounts_payable=100_000,
        short_term_debt=50_000,
        long_term_debt=300_000,
        other_liabilities=50_000,
        common_stock=200_000,
        retained_earnings=300_000
    )


@pytest.fixture
def accounting_model():
    """Create instance of DeterministicAccountingModel."""
    return DeterministicAccountingModel()


# ==============================================================================
# Unit Tests: BalanceSheetState
# ==============================================================================

class TestBalanceSheetState:
    """Tests for BalanceSheetState dataclass."""

    def test_total_assets_calculation(self, initial_balanced_state):
        """Test that total assets is sum of all asset components."""
        state = initial_balanced_state
        expected = (state.cash + state.accounts_receivable + state.inventory +
                   state.ppe_net + state.other_assets)
        assert state.total_assets() == expected

    def test_total_liabilities_calculation(self, initial_balanced_state):
        """Test that total liabilities is sum of all liability components."""
        state = initial_balanced_state
        expected = (state.accounts_payable + state.short_term_debt +
                   state.long_term_debt + state.other_liabilities)
        assert state.total_liabilities() == expected

    def test_total_equity_calculation(self, initial_balanced_state):
        """Test that total equity is sum of equity components."""
        state = initial_balanced_state
        expected = state.common_stock + state.retained_earnings
        assert state.total_equity() == expected

    def test_accounting_identity_holds(self, initial_balanced_state):
        """Test that A = L + E for valid balance sheet."""
        state = initial_balanced_state
        assert state.validate_identity(tolerance=1e-6)

        # Verify numerically
        assets = state.total_assets()
        liab_equity = state.total_liabilities() + state.total_equity()
        assert abs(assets - liab_equity) < 1e-6

    def test_accounting_identity_fails_for_imbalanced(self):
        """Test that validate_identity returns False for imbalanced state."""
        imbalanced = BalanceSheetState(
            cash=100_000,
            accounts_receivable=150_000,
            inventory=200_000,
            ppe_net=500_000,
            other_assets=50_000,
            accounts_payable=100_000,
            short_term_debt=50_000,
            long_term_debt=300_000,
            other_liabilities=50_000,
            common_stock=200_000,
            retained_earnings=200_000  # Changed to make imbalanced
        )
        assert not imbalanced.validate_identity(tolerance=1e-6)


# ==============================================================================
# Unit Tests: DriverAssumptions
# ==============================================================================

class TestDriverAssumptions:
    """Tests for DriverAssumptions dataclass."""

    def test_driver_creation(self, default_drivers):
        """Test that drivers are created with correct values."""
        drivers = default_drivers
        assert drivers.revenue_growth == 0.10
        assert drivers.cogs_as_pct_revenue == 0.60
        assert drivers.tax_rate == 0.21

    def test_negative_growth_allowed(self):
        """Test that negative revenue growth is accepted (recession scenario)."""
        drivers = DriverAssumptions(
            revenue_growth=-0.05,  # 5% decline
            cogs_as_pct_revenue=0.60,
            opex_as_pct_revenue=0.20,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.08,
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.30,
            target_cash_balance=100_000,
            interest_rate_on_debt=0.05
        )
        assert drivers.revenue_growth == -0.05

    def test_zero_dividend_payout(self):
        """Test that zero dividend payout (retention policy) is valid."""
        drivers = DriverAssumptions(
            revenue_growth=0.10,
            cogs_as_pct_revenue=0.60,
            opex_as_pct_revenue=0.20,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.08,
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.0,  # No dividends
            target_cash_balance=100_000,
            interest_rate_on_debt=0.05
        )
        assert drivers.dividend_payout_ratio == 0.0


# ==============================================================================
# Unit Tests: DeterministicAccountingModel
# ==============================================================================

class TestDeterministicAccountingModel:
    """Tests for the deterministic accounting model."""

    def test_single_period_identity_compliance(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """
        CRITICAL TEST: Verify accounting identity A = L + E holds after forecast.

        This is the core requirement from Velez-Pareja: the balance sheet must
        balance by construction, not by adjustment.
        """
        initial_revenue = 1_000_000

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        # Accounting identity must hold
        assert new_state.validate_identity(tolerance=1e-6), \
            f"Identity violated: A={new_state.total_assets():.2f}, " \
            f"L+E={new_state.total_liabilities()+new_state.total_equity():.2f}"

    def test_circularity_convergence(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """
        Test that interest-debt circularity converges within reasonable iterations.

        Velez-Pareja recommends 3-5 iterations for typical cases.
        """
        initial_revenue = 1_000_000

        _, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        iterations = auxiliary['iterations_to_converge']
        assert iterations <= 10, f"Circularity took {iterations} iterations to converge"

    def test_retained_earnings_consistency(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """
        Test that RE(t) = RE(t-1) + Net Income - Dividends.

        This is a fundamental accounting relationship.
        """
        initial_revenue = 1_000_000

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        # Expected retained earnings
        expected_re = (initial_balanced_state.retained_earnings +
                      auxiliary['net_income'] -
                      auxiliary['dividends'])

        assert abs(new_state.retained_earnings - expected_re) < 1e-6, \
            f"RE mismatch: got {new_state.retained_earnings:.2f}, expected {expected_re:.2f}"

    def test_revenue_growth_applied_correctly(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """Test that revenue grows by the specified rate."""
        initial_revenue = 1_000_000
        expected_revenue = initial_revenue * (1 + default_drivers.revenue_growth)

        _, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        assert abs(auxiliary['revenue'] - expected_revenue) < 1e-6

    def test_working_capital_calculations(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """Test that AR, Inventory, AP are calculated from DSO, DIO, DPO."""
        initial_revenue = 1_000_000

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        revenue = auxiliary['revenue']
        cogs = auxiliary['cogs']

        # Expected working capital
        expected_ar = revenue * (default_drivers.days_sales_outstanding / 365)
        expected_inventory = cogs * (default_drivers.days_inventory_outstanding / 365)
        expected_ap = cogs * (default_drivers.days_payable_outstanding / 365)

        assert abs(new_state.accounts_receivable - expected_ar) < 1e-6
        assert abs(new_state.inventory - expected_inventory) < 1e-6
        assert abs(new_state.accounts_payable - expected_ap) < 1e-6

    def test_ppe_evolution(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """Test that PP&E(t) = PP&E(t-1) + CapEx - Depreciation."""
        initial_revenue = 1_000_000

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=default_drivers,
            previous_revenue=initial_revenue
        )

        # Expected PP&E
        capex = auxiliary['revenue'] * default_drivers.capex_as_pct_revenue
        depreciation = auxiliary['depreciation']
        expected_ppe = initial_balanced_state.ppe_net + capex - depreciation

        assert abs(new_state.ppe_net - expected_ppe) < 1e-6

    def test_no_negative_taxes(
        self, accounting_model, initial_balanced_state
    ):
        """Test that taxes are non-negative even with losses."""
        # Create drivers that lead to a loss
        loss_drivers = DriverAssumptions(
            revenue_growth=-0.50,  # 50% revenue decline
            cogs_as_pct_revenue=0.90,  # High cost
            opex_as_pct_revenue=0.30,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.02,
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.0,
            target_cash_balance=50_000,
            interest_rate_on_debt=0.05
        )

        model = DeterministicAccountingModel()
        _, auxiliary = model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=loss_drivers,
            previous_revenue=1_000_000
        )

        assert auxiliary['taxes'] >= 0, "Taxes should not be negative"


# ==============================================================================
# Integration Tests: Multi-Period Forecasting
# ==============================================================================

class TestMultiPeriodForecasting:
    """Integration tests for multi-period forecasting."""

    def test_multi_period_all_identities_hold(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """
        CRITICAL: Verify A = L + E holds for ALL forecast periods.

        This is the main value proposition of the Velez-Pareja approach.
        """
        initial_revenue = 1_000_000
        periods = 8

        states, auxiliaries = accounting_model.forecast_multi_period(
            initial_state=initial_balanced_state,
            initial_revenue=initial_revenue,
            drivers_by_period=[default_drivers] * periods,
            periods=periods
        )

        for i, state in enumerate(states):
            assert state.validate_identity(tolerance=1e-6), \
                f"Identity violated in period {i+1}"

    def test_revenue_compounds_correctly(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """Test that revenue compounds correctly over multiple periods."""
        initial_revenue = 1_000_000
        periods = 5

        _, auxiliaries = accounting_model.forecast_multi_period(
            initial_state=initial_balanced_state,
            initial_revenue=initial_revenue,
            drivers_by_period=[default_drivers] * periods,
            periods=periods
        )

        current_rev = initial_revenue
        for aux in auxiliaries:
            expected_rev = current_rev * (1 + default_drivers.revenue_growth)
            assert abs(aux['revenue'] - expected_rev) < 1e-6
            current_rev = aux['revenue']

    def test_varying_drivers_per_period(
        self, accounting_model, initial_balanced_state
    ):
        """Test forecasting with different drivers each period."""
        # Create different drivers for each period
        drivers_list = []
        for growth in [0.10, 0.05, 0.08, -0.02, 0.12]:
            drivers_list.append(DriverAssumptions(
                revenue_growth=growth,
                cogs_as_pct_revenue=0.60,
                opex_as_pct_revenue=0.20,
                tax_rate=0.21,
                days_sales_outstanding=45,
                days_inventory_outstanding=60,
                days_payable_outstanding=30,
                capex_as_pct_revenue=0.08,
                depreciation_as_pct_ppe=0.10,
                dividend_payout_ratio=0.30,
                target_cash_balance=100_000,
                interest_rate_on_debt=0.05
            ))

        states, auxiliaries = accounting_model.forecast_multi_period(
            initial_state=initial_balanced_state,
            initial_revenue=1_000_000,
            drivers_by_period=drivers_list,
            periods=5
        )

        # All identities must still hold
        for i, state in enumerate(states):
            assert state.validate_identity(tolerance=1e-6), \
                f"Identity violated in period {i+1} with varying drivers"


# ==============================================================================
# Edge Case Tests
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_revenue_growth(
        self, accounting_model, initial_balanced_state
    ):
        """Test with zero revenue growth (stagnant business)."""
        drivers = DriverAssumptions(
            revenue_growth=0.0,
            cogs_as_pct_revenue=0.60,
            opex_as_pct_revenue=0.20,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.08,
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.30,
            target_cash_balance=100_000,
            interest_rate_on_debt=0.05
        )

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=drivers,
            previous_revenue=1_000_000
        )

        assert new_state.validate_identity(tolerance=1e-6)
        assert auxiliary['revenue'] == 1_000_000

    def test_high_growth_scenario(
        self, accounting_model, initial_balanced_state
    ):
        """Test with high revenue growth (hypergrowth scenario)."""
        drivers = DriverAssumptions(
            revenue_growth=0.50,  # 50% growth
            cogs_as_pct_revenue=0.60,
            opex_as_pct_revenue=0.20,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.15,  # High capex for growth
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.0,  # No dividends, reinvest
            target_cash_balance=100_000,
            interest_rate_on_debt=0.05
        )

        new_state, _ = accounting_model.forecast_one_period(
            previous_state=initial_balanced_state,
            drivers=drivers,
            previous_revenue=1_000_000
        )

        assert new_state.validate_identity(tolerance=1e-6)

    def test_zero_debt_scenario(self, accounting_model):
        """Test with zero initial debt."""
        debt_free_state = BalanceSheetState(
            cash=200_000,
            accounts_receivable=150_000,
            inventory=200_000,
            ppe_net=500_000,
            other_assets=50_000,
            accounts_payable=100_000,
            short_term_debt=0,
            long_term_debt=0,
            other_liabilities=50_000,
            common_stock=400_000,
            retained_earnings=550_000
        )

        drivers = DriverAssumptions(
            revenue_growth=0.10,
            cogs_as_pct_revenue=0.60,
            opex_as_pct_revenue=0.20,
            tax_rate=0.21,
            days_sales_outstanding=45,
            days_inventory_outstanding=60,
            days_payable_outstanding=30,
            capex_as_pct_revenue=0.08,
            depreciation_as_pct_ppe=0.10,
            dividend_payout_ratio=0.30,
            target_cash_balance=100_000,
            interest_rate_on_debt=0.05
        )

        new_state, auxiliary = accounting_model.forecast_one_period(
            previous_state=debt_free_state,
            drivers=drivers,
            previous_revenue=1_000_000
        )

        assert new_state.validate_identity(tolerance=1e-6)
        assert auxiliary['interest_expense'] == 0


# ==============================================================================
# Performance Tests
# ==============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_large_number_of_periods(
        self, accounting_model, initial_balanced_state, default_drivers
    ):
        """Test that model handles many periods without issues."""
        periods = 100

        states, auxiliaries = accounting_model.forecast_multi_period(
            initial_state=initial_balanced_state,
            initial_revenue=1_000_000,
            drivers_by_period=[default_drivers] * periods,
            periods=periods
        )

        assert len(states) == periods
        # Spot check: all identities still hold
        for state in states:
            assert state.validate_identity(tolerance=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
