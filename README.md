# Agro-Environmental Simulation Plant Health Prediction (Multi-Target Machine Learning)

> **Note:** This is a lightweight academic project designed for agricultural-environmental simulation and multi-target plant health prediction.

This repository focuses on predicting plant health using environmental simulation and soil sensor data through a **multi-target machine learning** approach. The study targets three core tasks:

1. **Failure Flag Prediction:** Predicts failure/plant death (Binary Classification) using `failure_flag`.
2. **Suitability Score Prediction:** Predicts suitability scores (Regression) using `suitability_score`.
3. **Stress Level Prediction:** Predicts stress levels (Multiclass Classification) using `stress_level`.

---

## Repository Structure

- `train_pipeline.py`: Consolidated machine learning pipeline handling feature engineering, preprocessing, train/test split, and model training/evaluations for all targets.
- `dataset.csv`: Main dataset used in the project.
- `requirements.txt`: Python library dependencies.

---

## Dataset Reference

**Main Dataset:** `dataset.csv`
- Records: 543,210
- Columns: 25
- Variables:
  - Environmental/Soil Features: `soil_type`, `soil_moisture_pct`, `soil_temp_c`, `air_temp_c`, `light_intensity_par`, `soil_ph`, `nitrogen_ppm`, `phosphorus_ppm`, `potassium_ppm`, etc.
  - Targets: `suitability_score`, `stress_level`, `failure_flag`

*Note: The dataset is simulated/synthetic; therefore, high performance metrics should not be directly expected in real-world agricultural environments.*

---

## Installation

```bash
python -m venv .venv
# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1
# Install dependencies
pip install -r requirements.txt
```

---

## Execution

```bash
python train_pipeline.py
```

Models are exported as serialized joblib artifacts under the `models/` directory.

---

## Methodology Summary

- **Preprocessing:** Separate numeric/categorical streams managed using `ColumnTransformer` + `Pipeline`.
- **Imbalance Management:** SMOTE evaluation for `failure_flag`, and `class_weight` adjustments for multiclass tasks.
- **Modeling:** Classical ML + Ensembles + Deep Learning models (ANN/CNN/LSTM).
- **Model Selection:** GridSearchCV based on primary metrics comparison.
- **Explainability:** SHAP feature importance analysis.
- **Robustness:** Sensitivity analysis under Gaussian noise.

---

## License
This project is licensed under the MIT License.
