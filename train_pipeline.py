import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("MLPipeline")

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Target Columns
TARGET_FAILURE = "failure_flag"
TARGET_SUITABILITY = "suitability_score"
TARGET_STRESS = "stress_level"

def main():
    log.info("Starting Multi-Target Plant Health Prediction Pipeline...")
    
    dataset_path = "dataset.csv"
    if not os.path.exists(dataset_path):
        log.error(f"Dataset not found at {dataset_path}. Please check file location.")
        sys.exit(1)
        
    t0 = time.time()
    log.info("Loading dataset...")
    df = pd.read_csv(dataset_path)
    log.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns in {time.time() - t0:.2f}s.")
    
    # 1. Feature Cleansing & Separation
    # Exclude ID-like and target columns
    feature_cols = [c for c in df.columns if c not in [TARGET_FAILURE, TARGET_SUITABILITY, TARGET_STRESS, "location_id"]]
    X = df[feature_cols]
    
    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Group numerical features by skewness thresholds
    skewed_cols = []
    normal_cols = []
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 1.5:
            skewed_cols.append(col)
        else:
            normal_cols.append(col)
            
    log.info(f"Features: {len(num_cols)} numerical ({len(skewed_cols)} skewed, {len(normal_cols)} normal), {len(cat_cols)} categorical.")
    
    # 2. Train / Test Split
    y_fail = df[TARGET_FAILURE]
    y_suit = df[TARGET_SUITABILITY]
    y_stress = df[TARGET_STRESS]
    
    X_train, X_test, y_fail_train, y_fail_test, y_suit_train, y_suit_test, y_stress_train, y_stress_test = train_test_split(
        X, y_fail, y_suit, y_stress,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_fail
    )
    log.info(f"Train set: {len(X_train):,}, Test set: {len(X_test):,}")
    
    # 3. Preprocessing Pipelines
    skewed_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
    ])
    
    normal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    transformers = []
    if skewed_cols:
        transformers.append(("skewed", skewed_pipeline, skewed_cols))
    if normal_cols:
        transformers.append(("normal", normal_pipeline, normal_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    log.info("Fitting preprocessing pipeline on training data...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    log.info(f"Preprocessed matrix shape: {X_train_proc.shape}")
    
    # Save Preprocessor
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    log.info("Saved preprocessor to 'models/preprocessor.joblib'.")
    
    # 4. Train Task 1: Binary Classification (Failure Flag)
    log.info("--- Task 1: Training Binary Classifier (Failure Flag) ---")
    fail_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1, verbose=-1)
    fail_model.fit(X_train_proc, y_fail_train)
    y_fail_pred = fail_model.predict(X_test_proc)
    
    log.info("Task 1 Results (Failure Flag):")
    log.info(f"Accuracy: {accuracy_score(y_fail_test, y_fail_pred):.4f}")
    log.info("\n" + classification_report(y_fail_test, y_fail_pred))
    joblib.dump(fail_model, "models/fail_classifier.joblib")
    
    # 5. Train Task 2: Regression (Suitability Score)
    log.info("--- Task 2: Training Regressor (Suitability Score) ---")
    suit_model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1, verbose=-1)
    suit_model.fit(X_train_proc, y_suit_train)
    y_suit_pred = suit_model.predict(X_test_proc)
    
    log.info("Task 2 Results (Suitability Score):")
    log.info(f"RMSE: {np.sqrt(mean_squared_error(y_suit_test, y_suit_pred)):.4f}")
    log.info(f"MAE: {mean_absolute_error(y_suit_test, y_suit_pred):.4f}")
    log.info(f"R2: {r2_score(y_suit_test, y_suit_pred):.4f}")
    joblib.dump(suit_model, "models/suitability_regressor.joblib")
    
    # 6. Train Task 3: Multiclass Classification (Stress Level)
    log.info("--- Task 3: Training Multiclass Classifier (Stress Level) ---")
    stress_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1, verbose=-1)
    stress_model.fit(X_train_proc, y_stress_train)
    y_stress_pred = stress_model.predict(X_test_proc)
    
    log.info("Task 3 Results (Stress Level):")
    log.info(f"Accuracy: {accuracy_score(y_stress_test, y_stress_pred):.4f}")
    log.info("\n" + classification_report(y_stress_test, y_stress_pred))
    joblib.dump(stress_model, "models/stress_classifier.joblib")
    
    log.info("All model artifacts saved successfully under the 'models/' folder.")
    log.info("Multi-Target Pipeline Run Complete.")

if __name__ == "__main__":
    main()
