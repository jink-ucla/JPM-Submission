"""
Balance Sheet Model
-------------------
Mathematical model for balance sheet forecasting with accounting identities.

Questions addressed: Part 1, Q1-Q2

Key accounting identities:
1. Assets = Liabilities + Equity
2. Change in Cash = Operating CF + Investing CF + Financing CF
3. Retained Earnings(t) = Retained Earnings(t-1) + Net Income - Dividends

Based on:
- Velez-Pareja(07, 09): No plugs, no circularity approach
- Pelaez(11): Analytical solution to circularity
"""

import numpy as np
from typing import Dict, List, Tuple


class BalanceSheetModel:
    """
    A model to forecast balance sheets while maintaining accounting identities.

    The model handles the circularity problem where interest expense depends on
    debt levels, which in turn depend on financing needs, which depend on net income,
    which depends on interest expense.
    """

    def __init__(self):
        """Initialize the balance sheet model with accounting relationships."""
        self.accounting_identities = {
            'fundamental_equation': 'Assets = Liabilities + Equity',
            'cash_flow': 'Delta_Cash = CF_Operating + CF_Investing + CF_Financing',
            'retained_earnings': 'RE(t) = RE(t-1) + Net_Income - Dividends'
        }

    def validate_accounting_identity(self, assets: float, liabilities: float,
                                    equity: float, tolerance: float = 1e-6) -> bool:
        """
        Validate the fundamental accounting equation.

        Args:
            assets: Total assets
            liabilities: Total liabilities
            equity: Total equity
            tolerance: Acceptable deviation

        Returns:
            True if identity holds within tolerance
        """
        return abs(assets - (liabilities + equity)) < tolerance

    def construct_balance_sheet_equations(self) -> Dict[str, str]:
        """
        Define the mathematical equations governing balance sheet evolution.

        Returns:
            Dictionary of balance sheet field equations
        """
        equations = {
            # Assets
            'Cash(t)': 'Cash(t-1) + CF_Operating(t) + CF_Investing(t) + CF_Financing(t)',
            'Accounts_Receivable(t)': 'Revenue(t) * Days_Sales_Outstanding / 365',
            'Inventory(t)': 'COGS(t) * Days_Inventory_Outstanding / 365',
            'Total_Current_Assets(t)': 'Cash(t) + Accounts_Receivable(t) + Inventory(t) + Other_Current_Assets(t)',
            'PP&E(t)': 'PP&E(t-1) + CapEx(t) - Depreciation(t)',
            'Total_Assets(t)': 'Total_Current_Assets(t) + PP&E(t) + Intangibles(t) + Other_Assets(t)',

            # Liabilities
            'Accounts_Payable(t)': 'COGS(t) * Days_Payable_Outstanding / 365',
            'Short_Term_Debt(t)': 'f(financing_needs, credit_policy)',
            'Total_Current_Liabilities(t)': 'Accounts_Payable(t) + Short_Term_Debt(t) + Other_Current_Liabilities(t)',
            'Long_Term_Debt(t)': 'Long_Term_Debt(t-1) + New_Debt_Issued(t) - Debt_Repayment(t)',
            'Total_Liabilities(t)': 'Total_Current_Liabilities(t) + Long_Term_Debt(t) + Other_LT_Liabilities(t)',

            # Equity
            'Retained_Earnings(t)': 'Retained_Earnings(t-1) + Net_Income(t) - Dividends(t)',
            'Common_Stock(t)': 'Common_Stock(t-1) + New_Equity_Issued(t)',
            'Total_Equity(t)': 'Common_Stock(t) + Retained_Earnings(t) + Other_Equity(t)',

            # Income Statement (linked)
            'Net_Income(t)': 'Revenue(t) - COGS(t) - Operating_Expenses(t) - Interest_Expense(t) - Taxes(t)',
            'Interest_Expense(t)': 'Total_Debt(t) * Interest_Rate(t)',
        }
        return equations

    def solve_circularity(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Solve the circularity problem in balance sheet forecasting.

        The circularity arises because:
        - Interest expense depends on debt level
        - Debt level depends on financing needs
        - Financing needs depend on net income
        - Net income depends on interest expense

        Args:
            params: Dictionary of input parameters

        Returns:
            Solved balance sheet with consistent values
        """
        # Iterative solution following Velez-Pareja approach
        max_iterations = 100
        tolerance = 1e-6

        # Initialize with guess
        interest_expense = params.get('interest_expense_guess', 0.0)

        for iteration in range(max_iterations):
            # Calculate net income given interest expense
            net_income = (params['revenue'] - params['cogs'] -
                         params['operating_expenses'] - interest_expense -
                         params['taxes'])

            # Calculate financing needs
            financing_needs = (params['capex'] + params['working_capital_increase'] -
                             net_income + params['dividends'])

            # Calculate debt level
            debt = params['existing_debt'] + max(0, financing_needs)

            # Calculate new interest expense
            new_interest_expense = debt * params['interest_rate']

            # Check convergence
            if abs(new_interest_expense - interest_expense) < tolerance:
                return {
                    'net_income': net_income,
                    'debt': debt,
                    'interest_expense': new_interest_expense,
                    'converged': True,
                    'iterations': iteration + 1
                }

            interest_expense = new_interest_expense

        return {
            'converged': False,
            'iterations': max_iterations
        }

    def forecast_balance_sheet(self, historical_bs: np.ndarray,
                               historical_is: np.ndarray,
                               horizon: int = 1) -> np.ndarray:
        """
        Forecast balance sheet for future periods.

        Args:
            historical_bs: Historical balance sheet data (T x N)
            historical_is: Historical income statement data (T x M)
            horizon: Number of periods to forecast

        Returns:
            Forecasted balance sheet (horizon x N)
        """
        # Placeholder for the actual forecasting logic
        # This will be implemented with TensorFlow in tensorflow_model.py
        raise NotImplementedError("Use TensorFlowBalanceSheetModel for forecasting")


def main():
    """Demonstrate balance sheet model functionality."""
    model = BalanceSheetModel()

    # Show accounting equations
    print("Balance Sheet Equations:")
    print("=" * 80)
    equations = model.construct_balance_sheet_equations()
    for field, equation in equations.items():
        print(f"{field:30s} = {equation}")

    # Example: Solve circularity problem
    print("\n\nSolving Circularity Problem:")
    print("=" * 80)
    params = {
        'revenue': 1000000,
        'cogs': 600000,
        'operating_expenses': 200000,
        'taxes': 40000,
        'capex': 50000,
        'working_capital_increase': 10000,
        'dividends': 20000,
        'existing_debt': 300000,
        'interest_rate': 0.05,
        'interest_expense_guess': 15000
    }

    solution = model.solve_circularity(params)
    if solution['converged']:
        print(f"Solution converged in {solution['iterations']} iterations:")
        print(f"  Net Income: ${solution['net_income']:,.2f}")
        print(f"  Total Debt: ${solution['debt']:,.2f}")
        print(f"  Interest Expense: ${solution['interest_expense']:,.2f}")
    else:
        print("Solution did not converge")

    # Validate accounting identity
    assets = 1000000
    liabilities = 400000
    equity = 600000
    is_valid = model.validate_accounting_identity(assets, liabilities, equity)
    print(f"\n\nAccounting Identity Valid: {is_valid}")
    print(f"  Assets: ${assets:,}")
    print(f"  Liabilities: ${liabilities:,}")
    print(f"  Equity: ${equity:,}")
    print(f"  Assets - (Liabilities + Equity): ${assets - (liabilities + equity):,.2f}")


if __name__ == "__main__":
    main()
