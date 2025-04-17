# 🧠 Fundamental Split Technique (FST)

**FST** is a robust, model-agnostic algorithm designed to enhance regression model reliability by iteratively identifying and training on data points that are hardest to predict. By focusing on problematic samples, FST achieves more stable and improved predictive performance compared to traditional random splitting techniques.

---
## Installation

You can install the `fst` package in the following ways:

### Option 1: Install directly from GitHub

```bash
pip install git+https://github.com/abuistar8/fundamental-split-technique.git
```

### Option 2: Install from local directory

If you’ve cloned or downloaded the repo:
```bash
cd fundamental-split-technique
pip install .
```
Use it like this is python
```python
from fst import FST
```

## 📌 Overview

Traditional random train-test splits can lead to inconsistent and unreliable performance. FST counters this by:
- Running multiple randomized train-test splits.
- Tracking which data points consistently lead to poor model performance.
- "Fixing" those points into the training set for future iterations.

---

## 🔍 How It Works

1. **Initialization**
   - All data points start as `non-fixed`.
   - An empty `fixed` set is created.

2. **Inner Iterations (within each outer iteration)**
   - Randomly split `non-fixed` points into train/test.
   - Train the model with: `fixed` + inner train.
   - Evaluate on inner test.
   - Track how often each point is involved in a poor-performing test.

3. **Evaluation**
   - Calculate a *bad ratio* for each non-fixed point.
   - Move the worst-performing percentage into the fixed set.

4. **Stopping Criteria**
   - A predefined portion of data has been fixed.
   - No new "bad" points are detected.
   - All points are fixed.

5. **Final Model**
   - Train the final model using fixed points.
   - Evaluate on the remaining non-fixed points.

---

## ⚙️ Parameters

| Parameter               | Description                                                     | Default     |
|------------------------|-----------------------------------------------------------------|-------------|
| `bad_r2_threshold`     | Threshold below which a run is considered bad (based on R²)    | `-0.01`     |
| `outer_split_bad_frac` | Fraction of worst non-fixed points fixed per outer iteration   | `0.30`      |
| `max_fixed_frac`       | Max fraction of data allowed in the fixed set                  | `0.80`      |
| `n_inner`              | Number of inner random splits per outer iteration              | `100`       |
| `regressor`            | Base regression model used                                      | `LinearRegression` |

---

## 💻 Usage

```python
from fst import FST
from sklearn.ensemble import RandomForestRegressor

# Initialize the algorithm
fst = FST(
    regressor=RandomForestRegressor(n_estimators=100),
    bad_r2_threshold=0.0,
    outer_split_bad_frac=0.3,
    max_fixed_frac=0.8,
    n_inner=100,
    random_state=42
)

# Prepare your data
X, y = fst.prepare_data(
    df=your_dataframe,
    target_col='target',
    feature_cols=['feature1', 'feature2', ...],
    categorical_cols=['cat1', 'cat2', ...],
    date_col='date_column'
)

# Train
fst.fit(X, y)

# Predict
predictions = fst.predict(X_new)

# Evaluate
performance = fst.evaluate_partition_performance(X, y)

# Visualize
fst.plot_r2_progression()

# Feature Importance
importances = fst.get_feature_importances()

```



## 🌟 Advantages Over Traditional Splitting

✅ Improved R² Scores by leveraging hard-to-predict samples.

✅ Model Robustness through repeated learning from difficult cases.

✅ Better Data Insights via identification of "problematic" points.

✅ Adaptive Splitting tailored to your dataset characteristics.



---

🚀 Practical Applications

FST is especially beneficial when:

Your data contains heterogeneous subpopulations.

Prediction difficulty varies across data points.

Random splits yield unstable model performance.

You need strong generalization on unseen data.



---

⚠️ Limitations

⏳ More computationally intensive than random splitting.

📉 May reduce test set size if many points get fixed.

⚙️ Effectiveness varies depending on dataset complexity.



---

📚 References

Snee, R. D. (1977). Validation of Regression Models: Methods and Examples.

Shao, J. (1993). Linear Model Selection by Cross-validation.



---

📈 Future Work

Integration with classification problems.

Auto-tuning of thresholds.

Multi-objective optimization for balancing performance and generalization.
