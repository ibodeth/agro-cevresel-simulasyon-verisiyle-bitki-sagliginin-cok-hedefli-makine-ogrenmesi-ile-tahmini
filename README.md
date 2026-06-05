# Agro-Environmental Simulation Plant Health Prediction

A multi-target machine learning pipeline to predict plant health, stress levels, and soil suitability scores based on environmental simulation datasets.

## How it Works
The machine learning pipeline processes numeric and categorical feature inputs through a column transformer. It handles class imbalance in target parameters using SMOTE and trains multiple classifiers and regression models (Artificial Neural Networks, LSTMs, and Ensemble models) to output plant failure likelihood, suitability scores, and stress levels.

## Tech Stack
- **Languages/Frameworks:** Python
- **Services/Libraries:** scikit-learn, joblib, imbalanced-learn, SHAP, TensorFlow, Keras
- **Infrastructure:** N/A

## Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ibodeth/agro-cevresel-simulasyon-verisiyle-bitki-sagliginin-cok-hedefli-makine-ogrenmesi-ile-tahmini.git
   cd agro-cevresel-simulasyon-verisiyle-bitki-sagliginin-cok-hedefli-makine-ogrenmesi-ile-tahmini
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline:
   ```bash
   python train_pipeline.py
   ```

## License
MIT
