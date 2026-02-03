"""
Deterministic Accounting Model
===============================
Implementation of constrained balance sheet forecasting per Velez-Pareja and Shahnazarian.

This is the CORRECT approach for Part 1. It builds financial statements that balance by 
construction, avoiding the "plug" problem and maintaining accounting identities.

Key Principles:
1. NO statistical forecasting of balance sheet items directly
2. Balance sheet is DERIVED from drivers and accounting rules
3. The model is DETERMINISTIC - no stochastic elements in the core logic
4. ML/LLM should ONLY forecast the drivers (x(t)), not the accounting relationships

References:
- Velez-Pareja (2007): Forecasting Financial Statements with No Plugs and No Circularity
- Shahnazarian (2004): Dynamic Microeconometric Simulation Model

Model Form (from prompt):
    y(t+1) = f(x(t), y(t)) + n(t)
    
Where:
- y(t) = Balance sheet stocks at time t (e.g., Receivables, PP&E, Debt, Equity)
- x(t) = Exogenous drivers and policies (e.g., Sales Growth, DSO, CapEx Policy)
- f(...) = DETERMINISTIC function derived from accounting rules (this module)
- n(t) = Stochastic noise (residual, not part of core model)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DriverAssumptions:
    """
    Exogenous drivers x(t) that determine the balance sheet forecast.
    These are the ONLY things that should be forecasted using ML/statistics.
    """
    # Income Statement Drivers
    revenue_growth: float  # e.g., 0.05 = 5% growth
    cogs_as_pct_revenue: float  # e.g., 0.60 = 60% of revenue
    opex_as_pct_revenue: float  # e.g., 0.20 = 20% of revenue
    tax_rate: float  # e.g., 0.21 = 21% tax rate
    
    # Working Capital Policies
    days_sales_outstanding: float  # DSO - Receivables collection period
    days_inventory_outstanding: float  # DIO - Inventory holding period
    days_payable_outstanding: float  # DPO - Payables payment period
    
    # Investment Policies
    capex_as_pct_revenue: float  # Capital expenditures as % of revenue
    depreciation_as_pct_ppe: float  # Depreciation as % of gross PP&E
    
    # Financing Policies
    dividend_payout_ratio: float  # % of net income paid as dividends
    target_cash_balance: float  # Minimum cash to maintain
    interest_rate_on_debt: float  # Interest rate on borrowings


@dataclass
class BalanceSheetState:
    """
    Balance sheet stocks y(t) at a point in time.
    These are DERIVED, not forecasted directly.
    """
    # Assets
    cash: float
    accounts_receivable: float
    inventory: float
    ppe_net: float
    other_assets: float
    
    # Liabilities
    accounts_payable: float
    short_term_debt: float
    long_term_debt: float
    other_liabilities: float
    
    # Equity
    common_stock: float
    retained_earnings: float
    
    def total_assets(self) -> float:
        return (self.cash + self.accounts_receivable + self.inventory + 
                self.ppe_net + self.other_assets)
    
    def total_liabilities(self) -> float:
        return (self.accounts_payable + self.short_term_debt + 
                self.long_term_debt + self.other_liabilities)
    
    def total_equity(self) -> float:
        return self.common_stock + self.retained_earnings
    
    def validate_identity(self, tolerance: float = 1e-6) -> bool:
        """Verify Assets = Liabilities + Equity using relative tolerance for large values."""
        assets = self.total_assets()
        liab_equity = self.total_liabilities() + self.total_equity()
        difference = abs(assets - liab_equity)
        # Use relative tolerance for large values to handle floating-point precision
        scale = max(abs(assets), abs(liab_equity), 1.0)
        return difference < tolerance * scale


class DeterministicAccountingModel:
    """
    Core deterministic model implementing Velez-Pareja's no-plug, no-circularity approach.
    
    This is f(x(t), y(t)) from the prompt's model form.
    """
    
    def __init__(self):
        self.history = []
    
    def forecast_one_period(
        self,
        previous_state: BalanceSheetState,
        drivers: DriverAssumptions,
        previous_revenue: float
    ) -> Tuple[BalanceSheetState, Dict[str, float]]:
        """
        Forecast balance sheet for t+1 given state at t and driver assumptions.
        
        This is the deterministic function f(x(t), y(t)).
        
        The key insight from Velez-Pareja: we build the statements SEQUENTIALLY:
        1. Income Statement (IS) - driven by revenue growth and margin assumptions
        2. Cash Budget (CB) - tracks all cash inflows and outflows
        3. Balance Sheet (BS) - with Cash and Debt as BALANCING ITEMS (not forecasted)
        
        This ensures the balance sheet balances BY CONSTRUCTION, not by adjustment.
        
        Args:
            previous_state: Balance sheet at time t (y(t))
            drivers: Exogenous driver assumptions x(t)
            previous_revenue: Revenue at time t (needed for growth calculation)
            
        Returns:
            Tuple of (new_state, auxiliary_info)
        """
        # ===================================================================
        # STEP 1: Build Income Statement
        # ===================================================================
        revenue = previous_revenue * (1 + drivers.revenue_growth)
        cogs = revenue * drivers.cogs_as_pct_revenue
        gross_profit = revenue - cogs
        operating_expenses = revenue * drivers.opex_as_pct_revenue
        
        # EBITDA (before interest, taxes, depreciation, amortization)
        ebitda = gross_profit - operating_expenses
        
        # Depreciation
        depreciation = previous_state.ppe_net * drivers.depreciation_as_pct_ppe
        ebit = ebitda - depreciation
        
        # CIRCULARITY ALERT: Interest expense depends on debt, which depends on 
        # financing needs, which depends on net income, which depends on interest.
        # We solve this iteratively (Velez-Pareja's approach)
        
        # Initial guess for interest expense
        total_debt_estimate = (previous_state.short_term_debt + 
                              previous_state.long_term_debt)
        interest_expense = total_debt_estimate * drivers.interest_rate_on_debt
        
        # Iterate to solve circularity
        for iteration in range(100):  # Max 100 iterations
            ebt = ebit - interest_expense
            taxes = max(0, ebt * drivers.tax_rate)  # No negative tax
            net_income = ebt - taxes
            
            # ===================================================================
            # STEP 2: Build Cash Budget (determines financing needs)
            # ===================================================================
            
            # Operating Cash Flow (indirect method)
            ocf = net_income + depreciation  # Simplified: add back non-cash charges
            
            # Working capital changes
            new_ar = revenue * (drivers.days_sales_outstanding / 365)
            new_inventory = cogs * (drivers.days_inventory_outstanding / 365)
            new_ap = cogs * (drivers.days_payable_outstanding / 365)
            
            delta_ar = new_ar - previous_state.accounts_receivable
            delta_inventory = new_inventory - previous_state.inventory
            delta_ap = new_ap - previous_state.accounts_payable
            
            # Working capital increase uses cash
            ocf_after_wc = ocf - delta_ar - delta_inventory + delta_ap
            
            # Investing Cash Flow
            capex = revenue * drivers.capex_as_pct_revenue
            icf = -capex  # Negative = cash outflow
            
            # Financing Cash Flow (dividends)
            dividends = net_income * drivers.dividend_payout_ratio
            
            # Net cash flow before financing
            net_cf_before_financing = ocf_after_wc + icf - dividends
            
            # Cash position before debt adjustment
            cash_before_balancing = previous_state.cash + net_cf_before_financing
            
            # ===================================================================
            # STEP 3: DERIVE the balancing item (Debt or Cash)
            # ===================================================================
            # This is the key insight: we don't forecast cash or debt.
            # We DERIVE them from the cash budget.
            
            if cash_before_balancing < drivers.target_cash_balance:
                # Need to borrow
                borrowing_needed = drivers.target_cash_balance - cash_before_balancing
                new_short_term_debt = previous_state.short_term_debt + borrowing_needed #의미: 1년 안에 갚아야 하는 빚입니다.
                new_cash = drivers.target_cash_balance
            else:
                # Have excess cash, can pay down debt
                excess_cash = cash_before_balancing - drivers.target_cash_balance
                debt_paydown = min(excess_cash, previous_state.short_term_debt)
                new_short_term_debt = previous_state.short_term_debt - debt_paydown
                new_cash = cash_before_balancing - debt_paydown
            
            # Recalculate total debt and interest expense
            new_total_debt = new_short_term_debt + previous_state.long_term_debt
            new_interest_expense = new_total_debt * drivers.interest_rate_on_debt
            
            # Check convergence
            if abs(new_interest_expense - interest_expense) < 1e-6:
                # Converged!
                break
            
            interest_expense = new_interest_expense
        
        # ===================================================================
        # STEP 4: Build Balance Sheet (now everything is determined)
        # ===================================================================
        
        new_ppe_net = previous_state.ppe_net + capex - depreciation
        new_retained_earnings = previous_state.retained_earnings + net_income - dividends
        
        new_state = BalanceSheetState(
            # Assets
            cash=new_cash,
            accounts_receivable=new_ar,
            inventory=new_inventory,
            ppe_net=new_ppe_net,
            other_assets=previous_state.other_assets,  # Assume constant for simplicity
            
            # Liabilities
            accounts_payable=new_ap,
            short_term_debt=new_short_term_debt,
            long_term_debt=previous_state.long_term_debt,  # Assume no LT changes
            other_liabilities=previous_state.other_liabilities,  # Assume constant
            
            # Equity
            common_stock=previous_state.common_stock,  # No new equity issued
            retained_earnings=new_retained_earnings
        )
        
        # Verify accounting identity
        if not new_state.validate_identity():
            raise ValueError(
                f"Accounting identity violated! "
                f"Assets={new_state.total_assets():.2f}, "
                f"Liabilities+Equity={new_state.total_liabilities()+new_state.total_equity():.2f}"
            )
        
        # Auxiliary information for analysis
        auxiliary = {
            'revenue': revenue,
            'cogs': cogs,
            'gross_profit': gross_profit,
            'operating_expenses': operating_expenses,
            'ebitda': ebitda,
            'depreciation': depreciation,
            'ebit': ebit,
            'interest_expense': interest_expense,
            'ebt': ebt,
            'taxes': taxes,
            'net_income': net_income,
            'ocf': ocf_after_wc,
            'icf': icf,
            'fcf': ocf_after_wc + icf,
            'dividends': dividends,
            'iterations_to_converge': iteration + 1
        }
        
        return new_state, auxiliary
    
    def forecast_multi_period(
        self,
        initial_state: BalanceSheetState,
        initial_revenue: float,
        drivers_by_period: list[DriverAssumptions],
        periods: int
    ) -> Tuple[list[BalanceSheetState], list[Dict[str, float]]]:
        """
        Forecast multiple periods iteratively.
        
        Args:
            initial_state: Starting balance sheet
            initial_revenue: Starting revenue
            drivers_by_period: List of driver assumptions for each period
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (states, auxiliary_info) for each period
        """
        states = [initial_state]
        auxiliaries = []
        current_revenue = initial_revenue
        
        for i in range(periods):
            drivers = drivers_by_period[i] if i < len(drivers_by_period) else drivers_by_period[-1]
            
            new_state, auxiliary = self.forecast_one_period(
                previous_state=states[-1],
                drivers=drivers,
                previous_revenue=current_revenue
            )
            
            states.append(new_state)
            auxiliaries.append(auxiliary)
            current_revenue = auxiliary['revenue']
        
        return states[1:], auxiliaries  # Don't return initial state


def main():
    """Demonstrate the deterministic model."""
    print("=" * 80)
    print("DETERMINISTIC ACCOUNTING MODEL")
    print("Per Velez-Pareja (2007) and Shahnazarian (2004)")
    print("=" * 80)
    print()
    
    # Define initial state
    initial_state = BalanceSheetState(
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
    
    print("Initial Balance Sheet:")
    print(f"  Total Assets: ${initial_state.total_assets():,.0f}")
    print(f"  Total Liabilities: ${initial_state.total_liabilities():,.0f}")
    print(f"  Total Equity: ${initial_state.total_equity():,.0f}")
    print(f"  Balanced: {initial_state.validate_identity()}")
    print()
    
    # Define driver assumptions
    drivers = DriverAssumptions(
        revenue_growth=0.10,  # 10% growth
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
    
    # Create model and forecast
    model = DeterministicAccountingModel()
    initial_revenue = 1_000_000
    
    states, auxiliaries = model.forecast_multi_period(
        initial_state=initial_state,
        initial_revenue=initial_revenue,
        drivers_by_period=[drivers] * 5,  # Same drivers for 5 periods
        periods=5
    )
    
    # Display results
    print("FORECAST RESULTS (5 periods):")
    print("=" * 80)
    print()
    
    for i, (state, aux) in enumerate(zip(states, auxiliaries)):
        print(f"Period {i+1}:")
        print(f"  Revenue: ${aux['revenue']:,.0f}")
        print(f"  Net Income: ${aux['net_income']:,.0f}")
        print(f"  Free Cash Flow: ${aux['fcf']:,.0f}")
        print(f"  Total Assets: ${state.total_assets():,.0f}")
        print(f"  Total Debt: ${state.short_term_debt + state.long_term_debt:,.0f}")
        print(f"  Total Equity: ${state.total_equity():,.0f}")
        print(f"  Identity Valid: {state.validate_identity()}")
        print(f"  Converged in {aux['iterations_to_converge']} iterations")
        print()
    
    print("=" * 80)
    print("KEY INSIGHT:")
    print("Notice that ALL balance sheets balance perfectly (Assets = Liabilities + Equity)")
    print("This is because we DERIVED the balance sheet from accounting rules,")
    print("not by forecasting line items independently with ML.")
    print("=" * 80)


if __name__ == "__main__":
    main()
