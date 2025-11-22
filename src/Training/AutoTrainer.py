"""Utility functions that retrain the models with the freshest data."""

from __future__ import annotations

"""Utility functions that retrain models after a data refresh."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import sqlite3
import tensorflow as tf
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.Features.feature_engineering import augment_features


DATASET_TABLE = "dataset_2012-24_new"


def _load_dataset(dataset_path: Path) -> pd.DataFrame:
    con = sqlite3.connect(dataset_path)
    try:
        frame = pd.read_sql_query(f'SELECT * FROM "{DATASET_TABLE}"', con)
    finally:
        con.close()
    return frame


def _prepare_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    frame = augment_features(frame)
    ml_targets = frame['Home-Team-Win']
    ou_targets = frame['OU-Cover']

    drop_cols = {'index', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1', 'Home-Team-Win'}
    ml_frame = frame.drop(columns=[col for col in drop_cols if col in frame.columns])
    ml_frame = ml_frame.drop(columns=['Score', 'OU', 'OU-Cover'], errors='ignore')

    ou_frame = frame.drop(columns=['index', 'TEAM_NAME', 'TEAM_NAME.1', 'Date', 'Date.1'], errors='ignore')
    ou_frame = ou_frame.drop(columns=['Score'], errors='ignore')

    ou_mask = ou_targets != 2
    ou_frame = ou_frame.loc[ou_mask]
    ou_targets = ou_targets.loc[ou_mask]
    ou_frame['OU'] = frame.loc[ou_mask, 'OU']

    ml_frame = ml_frame.fillna(0).astype(float)
    ou_frame = ou_frame.fillna(0).astype(float)

    return ml_frame, ml_targets, ou_frame, ou_targets


def _train_xgb_classifier(features: pd.DataFrame, targets: pd.Series) -> xgb.Booster:
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        'max_depth': 4,
        'eta': 0.05,
        'objective': 'multi:softprob',
        'num_class': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
    }
    booster = xgb.train(params, dtrain, num_boost_round=600, evals=[(dtest, 'eval')], verbose_eval=False)
    return booster


def _train_nn_classifier(features: pd.DataFrame, targets: pd.Series) -> tf.keras.Model:
    X = tf.keras.utils.normalize(features.values.astype(float), axis=1)
    y = targets.values.astype(int)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    return model


def retrain_models(dataset_path: Path = Path('Data/dataset.sqlite')) -> Dict[str, Path]:
    """Retrain the XGBoost and NN models with the freshest dataset."""

    frame = _load_dataset(dataset_path)
    if frame.empty:
        raise RuntimeError('Dataset is empty - refresh data before retraining.')

    ml_frame, ml_targets, ou_frame, ou_targets = _prepare_features(frame)

    output_paths: Dict[str, Path] = {}

    ml_booster = _train_xgb_classifier(ml_frame, ml_targets)
    ou_booster = _train_xgb_classifier(ou_frame, ou_targets)
    xgb_dir = Path('Models') / 'XGBoost_Models'
    xgb_dir.mkdir(parents=True, exist_ok=True)
    ml_model_path = xgb_dir / 'latest_moneyline.json'
    ou_model_path = xgb_dir / 'latest_total.json'
    ml_booster.save_model(ml_model_path)
    ou_booster.save_model(ou_model_path)
    output_paths['xgb_ml'] = ml_model_path
    output_paths['xgb_ou'] = ou_model_path

    nn_dir = Path('Models') / 'NN_Models'
    nn_dir.mkdir(parents=True, exist_ok=True)
    nn_ml_path = nn_dir / 'latest_moneyline.keras'
    nn_ou_path = nn_dir / 'latest_total.keras'
    ml_nn = _train_nn_classifier(ml_frame, ml_targets)
    ou_nn = _train_nn_classifier(ou_frame, ou_targets)
    ml_nn.save(nn_ml_path)
    ou_nn.save(nn_ou_path)
    output_paths['nn_ml'] = nn_ml_path
    output_paths['nn_ou'] = nn_ou_path

    return output_paths

