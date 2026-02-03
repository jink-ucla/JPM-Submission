<p align="center">
  <img src="main.png" alt="Logo" width="400">
</p>



# JP Morgan MLCOE TSRL 2026 Internship - Part 1 Submission

**Authors:** Jin Kim, Zixuan Zhao, Shawn Shin
**Submission Date:** February 2026
**Deadline:** February 3, 2026 (Part 1)

---

## Overview

This repository contains the Part 1 submission for the JP Morgan Machine Learning Center of Excellence (MLCOE) 2026 Internship interview exercise. We have implemented solutions for all three questions:

- **Q1:** Balance Sheet Modeling with Accounting Identities
- **Q2:** State-Space Filtering (Kalman Filter to Particle Flow)
- **Q3:** Deep Context-Dependent Choice Model

## Repository Structure

```
submission/
├── readme.md                           # This file
├── report.pdf                          # PDF report (main deliverable)
│
├── Scripts/
│   ├── pytest.ini                      # Test configuration
│   ├── README_tests.md                 # Testing documentation
│   │
│   ├── Q1_Part1/                       # Q1: Balance Sheet Modeling
│   │   ├── deterministic_accounting_model.py   # Core Velez-Pareja model
│   │   ├── driver_forecasting_model.py         # LSTM/XGBoost driver forecasting
│   │   ├── balance_sheet_model.py              # Combined model
│   │   ├── data_collection.py                  # Yahoo Finance data ingestion
│   │   ├── train_and_evaluate.py               # Training pipeline
│   │   ├── earnings_forecast.py                # Earnings derivation
│   │   ├── ml_enhancements.py                  # Advanced ML techniques
│   │   ├── tensorflow_model.py                 # TensorFlow implementation
│   │   ├── data/                               # Financial data (AAPL)
│   │   └── unit_test/                          # Unit tests
│   │       ├── test_accounting_model.py        # 20 tests
│   │       └── conftest.py
│   │
│   ├── Q2_Part1/                       # Q2: State-Space Filtering
│   │   ├── kalman_filter.py                    # Kalman Filter with Joseph form
│   │   ├── ekf_ukf.py                          # Extended/Unscented KF
│   │   ├── particle_filter.py                  # Bootstrap Particle Filter
│   │   ├── particle_flow_filters.py            # EDH, LEDH, PF-PF
│   │   ├── kernel_particle_flow.py             # Kernel-embedded flow (Hu 2021)
│   │   ├── nonlinear_models.py                 # SV and Range-Bearing models
│   │   ├── compare_filters.py                  # Filter comparison
│   │   ├── compare_flow_filters.py             # Particle flow comparison
│   │   ├── high_dim_comparison.py              # High-dimensional tests
│   │   ├── *.png                               # Result figures
│   │   └── unit_test/                          # Unit tests
│   │       ├── test_kalman_filter.py           # 23 tests
│   │       ├── test_particle_flow.py           # 15 tests
│   │       └── conftest.py
│   │
│   └── Q3_Part1/                       # Q3: Deep Choice Model
│       ├── deep_context_choice_model.py        # TensorFlow implementation
│       ├── choicelearn_wrapper.py              # Choice-learn integration
│       ├── synthetic_data_test.py              # Synthetic experiments
│       ├── *.csv                               # Generated data
│       ├── *.png                               # Training curves
│       └── unit_test/                          # Unit tests
│           ├── test_choice_model.py            # 20 tests
│           └── conftest.py
```

## Environment Setup

### Quick Start (Recommended)

We provide a virtual environment setup using `uv` for fast, reliable dependency management.

```bash
cd /cvib2/apps/personal/jink/JPM-Internal/submission

# Option 1: Use existing venv
source venv/bin/activate

# Option 2: Create new venv with uv
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

### Manual Installation

If you prefer pip:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

The `requirements.txt` includes:
- **numpy, scipy, pandas** - Core numerical computing
- **tensorflow, tensorflow-probability, tf-keras** - Deep learning (Q3)
- **scikit-learn, xgboost** - Machine learning (Q1)
- **matplotlib, seaborn** - Visualization
- **yfinance** - Financial data (Q1)
- **pytest, pytest-cov** - Testing

### Software Versions Used

- Python 3.10+
- TensorFlow 2.20+
- TensorFlow Probability 0.25+
- NumPy 2.2+
- SciPy 1.15+

## Running the Code

### Q1: Balance Sheet Modeling

```bash
cd Scripts/Q1_Part1

# Run the deterministic accounting model
python deterministic_accounting_model.py

# Run integrated example with AAPL data
python integrated_example.py

# Train driver forecasting model
python train_and_evaluate.py
```

### Q2: State-Space Filtering

```bash
cd Scripts/Q2_Part1

# Run Kalman Filter example
python kalman_filter.py

# Compare nonlinear filters (EKF, UKF, PF)
python compare_filters.py

# Run particle flow filter comparison
python compare_flow_filters.py

# High-dimensional comparison (Hu 2021 replication)
python high_dim_comparison.py
```

### Q3: Deep Choice Model

```bash
cd Scripts/Q3_Part1

# Run synthetic data tests
python synthetic_data_test.py
```

## Running Unit Tests

We have implemented comprehensive unit tests using pytest as required by the assignment.

```bash
# Activate virtual environment first
source venv/bin/activate

# Run tests for each question (recommended)
cd Scripts/Q1_Part1/unit_test && python -m pytest -v    # Q1: 22 tests
cd Scripts/Q2_Part1/unit_test && python -m pytest -v    # Q2: 38 tests
cd Scripts/Q3_Part1/unit_test && python -m pytest -v    # Q3: 20 tests

# Or run from Scripts directory
cd Scripts
python -m pytest Q1_Part1/unit_test/test_accounting_model.py -v
python -m pytest Q2_Part1/unit_test/test_kalman_filter.py Q2_Part1/unit_test/test_particle_flow.py -v
python -m pytest Q3_Part1/unit_test/test_choice_model.py -v
```

### Test Summary

| Question | Test File | # Tests | Status | Coverage |
|----------|-----------|---------|--------|----------|
| Q1 | test_accounting_model.py | 22 | ✅ All Pass | Accounting identity, circularity, RE consistency |
| Q2 | test_kalman_filter.py | 21 | ✅ All Pass | Prediction/update, NEES, numerical stability |
| Q2 | test_particle_flow.py | 17 | ✅ All Pass | EDH flow, Jacobian, high-dimensional |
| Q3 | test_choice_model.py | 20 | ✅ All Pass | Probability axioms, gradients, training |
| **Total** | | **80** | ✅ **All Pass** | |

### Key Test Categories

**Q1 - Balance Sheet Modeling:**
- Accounting identity compliance (A = L + E)
- Circularity convergence (interest-debt loop)
- Retained earnings consistency
- Working capital calculations (DSO, DIO, DPO)

**Q2 - State-Space Filtering:**
- NEES/NIS statistical consistency
- Joseph stabilized covariance updates
- Covariance positive definiteness
- Flow function correctness

**Q3 - Deep Choice Model:**
- Probability axioms (non-negative, sum to 1)
- Gradient flow (no vanishing gradients)
- Training convergence
- IIA violation detection

## Report

The main report is `report.pdf` which includes:

1. **Literature Review** - For each question, we review relevant papers and methods
2. **Method Selection** - Justification for our chosen approaches
3. **Implementation Details** - Technical description of our solutions
4. **Testing Plan** - Description of correctness and performance tests
5. **Results** - Experimental results with figures and tables
6. **Discussion** - Analysis of strengths, limitations, and recommendations

## Key Technical Contributions

### Q1: Two-Tier Balance Sheet Architecture
- Deterministic accounting model based on Velez-Pareja (2007, 2009)
- ML-based driver forecasting (LSTM + XGBoost)
- Zero accounting identity violations across all forecasts

### Q2: Comprehensive Filter Comparison
- Kalman Filter with Joseph stabilized updates
- EKF, UKF, Bootstrap Particle Filter
- EDH, LEDH, PF-PF particle flow filters
- Kernel-embedded flow with matrix-valued kernels (Hu 2021)

### Q3: Attention-Based Choice Model
- Context and product encoders with shared weights
- Multi-head attention for context-product interactions
- ResNet-style skip connections for gradient stability
- L2 regularization and gradient clipping

## References

See the report for full references. Key papers include:

- Velez-Pareja (2007, 2009): Balance sheet forecasting
- Daum & Huang (2010, 2011): Particle flow filters
- Li & Coates (2017): Invertible particle flow
- Hu & Van Leeuwen (2021): Kernel-embedded particle flow
- Zhang et al. (2025): Deep context-dependent choice model

## Contact

For questions about this submission, please contact the authors:
- Jin Kim: kimjin116@g.ucla.edu
- Zixuan Zhao: zxzhao@ucla.edu
- Shawn Shin: shawnshin99@ucla.edu

---

*This submission is for the JP Morgan MLCOE TSRL 2026 Internship interview exercise.*
