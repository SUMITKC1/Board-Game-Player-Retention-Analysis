# Board Game Player Retention Analysis

Predict player churn using the Kaggle dataset: "Prediction of User Loss in Mobile Games".

## Project Structure

```
/
├── data/                      # Kaggle CSVs 
│   ├── train.csv
│   ├── test.csv
│   ├── dev.csv
│   ├── level_seq.csv
│   └── level_meta.csv
├── notebooks/
│   └── churn_model.ipynb      # EDA, feature engineering, models, evaluation
├── src/
│   ├── preprocess.py          # load_data(), create_features()
│   └── churn_model.py         # train/evaluate/save XGBoost
├── models/
│   └── churn_xgb.pkl          # saved XGBoost model (created after training)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset
- Player-level splits: `train.csv`, `test.csv`, `dev.csv` (with churn labels in train/dev)
- Level data: `level_seq.csv` (player level progression), `level_meta.csv` (level metadata)

## Tech Stack
- Python, pandas, scikit-learn, XGBoost
- SQL, Tableau, AWS, Docker

## Results
- Achieved **87% accuracy** (ROC-AUC **0.82**) on validation with XGBoost baseline.

## Quickstart (Local)
1. Place the five CSVs into `data/`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```
3. Train and save the model:
   ```bash
   python -m src.churn_model
   ```
4. Explore notebook:
   ```bash
   jupyter lab
   ```

## Quickstart (Docker)
1. Build the image:
   ```bash
   docker build -t churn-retention .
   ```
2. Run with the project mounted (so outputs persist):
   ```bash
   docker run --rm -it -p 8888:8888 -v %cd%:/app churn-retention
   ```
   Then open the printed Jupyter URL in your browser.

## Notebook Guide (`notebooks/churn_model.ipynb`)
- EDA: player activity trends, retention curves
- Feature engineering: avg session length, recency, purchase count, level progression
- Models: Logistic Regression, Random Forest, XGBoost
- Metrics: Accuracy, ROC-AUC; feature importance plots
- Output: saves best model to `models/churn_xgb.pkl`

## Notes
- Column names in Kaggle data can vary; preprocessing normalizes common variants (e.g., `user_id` → `player_id`, `is_churn` → `churn`). Adjust mappings in `src/preprocess.py` if needed.
- For inference/serving, you can add a simple Flask API that loads `models/churn_xgb.pkl` and exposes a `/predict` endpoint.
