# Unit Tests for JP Morgan MLCOE 2026 Internship Submission

This document describes the unit and integration tests for the Part 1 submission.

## Requirements

```bash
pip install pytest pytest-cov numpy scipy tensorflow tensorflow-probability
```

## Running Tests

### Activate Virtual Environment First

```bash
cd /cvib2/apps/personal/jink/JPM-Internal/submission
source venv/bin/activate
```

### Run Tests for Each Question

```bash
# Q1: Balance Sheet Modeling (22 tests)
cd Scripts/Q1_Part1/unit_test && python -m pytest test_accounting_model.py -v

# Q2: State-Space Filtering (38 tests)
cd Scripts/Q2_Part1/unit_test && python -m pytest test_kalman_filter.py test_particle_flow.py -v

# Q3: Deep Choice Model (20 tests)
cd Scripts/Q3_Part1/unit_test && python -m pytest test_choice_model.py -v
```

### Run Specific Test Class or Function

```bash
cd Scripts/Q1_Part1/unit_test

# Run all tests in a class
python -m pytest test_accounting_model.py::TestBalanceSheetState -v

# Run a specific test
python -m pytest test_accounting_model.py::TestBalanceSheetState::test_accounting_identity_holds -v
```

## Test Organization

### Q1: Balance Sheet Modeling (`Q1_Part1/unit_test/`)

- `test_accounting_model.py`: Tests for the deterministic accounting model
  - `TestBalanceSheetState`: Tests for balance sheet state validation
  - `TestDriverAssumptions`: Tests for driver assumptions
  - `TestDeterministicAccountingModel`: Tests for single-period forecasting
  - `TestMultiPeriodForecasting`: Integration tests for multi-period forecasts
  - `TestEdgeCases`: Edge case handling
  - `TestPerformance`: Performance tests

**Key Test Categories:**
1. **Accounting Identity Compliance**: Verifies A = L + E holds after every forecast
2. **Circularity Convergence**: Tests interest-debt circularity resolution
3. **Retained Earnings Consistency**: Validates RE = RE_prev + NI - Dividends
4. **Working Capital Calculations**: Tests DSO, DIO, DPO implementations

### Q2: State-Space Filtering (`Q2_Part1/unit_test/`)

- `test_kalman_filter.py`: Tests for Kalman Filter
  - `TestKalmanFilterInitialization`: Initialization tests
  - `TestKalmanFilterPredict`: Prediction step tests
  - `TestKalmanFilterUpdate`: Update step tests
  - `TestKalmanFilterIntegration`: Full filtering sequence tests
  - `TestNumericalStability`: Stability tests
  - `TestNEESNIS`: Statistical consistency tests

- `test_particle_flow.py`: Tests for Particle Flow Filters
  - `TestEDHParticleFlowFilter`: EDH flow tests
  - `TestFlowJacobian`: Jacobian computation tests
  - `TestFilteringPerformance`: Performance tests
  - `TestNumericalStability`: Stability tests
  - `TestHighDimensional`: High-dimensional tests (Hu(2021))

**Key Test Categories:**
1. **NEES Distribution**: Verifies filter consistency via chi-squared test
2. **Joseph Stabilization**: Compares numerical stability
3. **Covariance Positive Definiteness**: Ensures valid covariance matrices
4. **Flow Convergence**: Tests particle migration

### Q3: Deep Choice Model (`Q3_Part1/unit_test/`)

- `test_choice_model.py`: Tests for Deep Context-Dependent Choice Model
  - `TestContextEncoder`: Context encoder tests
  - `TestProductEncoder`: Product encoder tests
  - `TestProbabilityAxioms`: **Critical** probability axiom tests
  - `TestGradientFlow`: Gradient flow verification
  - `TestLossComputation`: Loss function tests
  - `TestTraining`: Training integration tests
  - `TestIIAViolation`: IIA violation detection tests
  - `TestEdgeCases`: Edge case handling

**Key Test Categories:**
1. **Probability Axioms**: Verifies P(j|S) >= 0 and sum = 1
2. **Gradient Flow**: Ensures no vanishing gradients
3. **Training Convergence**: Verifies loss decreases
4. **IIA Violation**: Tests context-dependent substitution patterns

## Test Coverage Summary

| Component | Unit Tests | Integration Tests | Total | Status |
|-----------|------------|-------------------|-------|--------|
| Q1 Accounting Model | 17 | 5 | 22 | ✅ Pass |
| Q2 Kalman Filter | 16 | 5 | 21 | ✅ Pass |
| Q2 Particle Flow | 14 | 3 | 17 | ✅ Pass |
| Q3 Choice Model | 16 | 4 | 20 | ✅ Pass |
| **Total** | **63** | **17** | **80** | ✅ **All Pass** |

## Expected Test Results

All 80 tests pass with the provided virtual environment.

```bash
# Expected output for Q1
cd Scripts/Q1_Part1/unit_test && python -m pytest -v
======================== 22 passed in 1.62s ========================

# Expected output for Q2
cd Scripts/Q2_Part1/unit_test && python -m pytest -v
======================== 38 passed in 9.67s ========================

# Expected output for Q3
cd Scripts/Q3_Part1/unit_test && python -m pytest -v
======================== 20 passed in 21.37s ========================
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
pytest --tb=short --junitxml=test-results.xml
```
