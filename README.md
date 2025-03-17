# Demo Repository: Factor Models, Portfolio Optimization & Regime Clustering

This repo is **for demonstration purposes only**. It’s a stripped-down snapshot from a private repository, so some proprietary or more advanced techniques have been omitted or replaced with simpler stubs. **Please note** that this is not meant to be a fully advanced or production-ready codebase.

## Contents

```plaintext
├── plotting                         # Plotting helpers
│   ├── factors_modelval.py
│   ├── factors.py
│   ├── portfolio_opt.py
│   └── regime_clustering.py
│
├── factor_model                     # Proprietary (intro samples only)
│   ├── fundamental
│   │   ├── exposure_builder.py       
│   │   ├── factor_definitions.py     
│   │   └── fundamental_fitter.py     
│   ├── statistical_fitter.py         
│   └── factor_model.py               
│
├── portfolio_opt                    # Proprietary (intro samples only)
│   ├── constraints
│   │   ├── notional
│   │   │   ├── basket_notional_constraint.py
│   │   │   └── turnover_constraint.py
│   │   ├── base.py
│   │   ├── setup_constraint.py
│   │   └── structured_constraint.py
│   ├── objectives
│   │   ├── base.py
│   │   ├── factor_risk_objective.py
│   │   └── linear_objective.py
│   ├── settings.py       
│   └── cvxpy_wrapper.py              
│
├── regimes                          # Proprietary (backend code not included)
│   ├── helpers
│   │   └── windowing.py             
│   └── utils.py                     
│
├── signals                          # A sample momentum signal
│   └── momentum.py                  
│
├── Sample 1 - Factor Models.ipynb
├── Sample 2 - Regime Clustering.ipynb
│
└── README.md
```

### Main Modules

1. **Factor Model**  
   Demonstrates how to build both fundamental and statistical factor risk models.  
   - *Fundamental approach*: uses categorical or numeric exposures, then fits factor returns via simple OLS or Huber regression.  
   - *Statistical approach*: a basic iterative SVD-based model to estimate factors.

2. **Portfolio Optimization**  
   Includes a lightweight CVXPY wrapper and sample constraints/objectives (e.g., turnover constraints, factor risk minimization).  
   - This is only a **demo**—actual advanced constraints or solvers are omitted or stubbed.

3. **Regime Detection**  
   Illustrates clustering-based regime identification using placeholders for Earth Mover’s Distance (EMD) and K-means.  
   - For background, see [Earth Mover's Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) and [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering). </br>
   More literature on the subject can be found [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4758243).

4. **Signals**  
   A toy momentum signal generator (`momentum.py`).

---

## Disclaimer

- **Not Advanced**: This code is deliberately simplified and is not intended as a robust, production-grade system.  
- **Partial Extraction**: Certain proprietary algorithms and advanced logic are **not** included (using serialized files instead).  
- **For Demo Only**: Provided as a reference for exploring basic factor modeling, optimization, and clustering concepts.

---

## References

- [CVXPY Documentation](https://www.cvxpy.org/) – For modeling and solving convex optimization problems.
- [Convex Optimization by Boyd & Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/) – Foundational text on convex optimization techniques.
- [Fama-French Factor Models](https://www.investopedia.com/terms/f/famafrenchthreefactormodel.asp) – Introduction to popular fundamental factor models.
- [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) – Basis for the statistical factor model fitter using SVD.
- [Robust Regression (HuberRegressor)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html) – Technique for robust regression used in factor model fitting.
- [Earth Mover’s Distance](https://en.wikipedia.org/wiki/Earth_mover%27s_distance) – Measure used in regime clustering to compare distributions.
- [K-means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) – Standard algorithm for partitioning data into clusters.
