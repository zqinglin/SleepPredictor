import os
import json
import re
import argparse
from typing import Dict, List, Optional

import yaml
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib

from .data_io import read_wrist, read_mattress, read_air
from .features import build_features
from .labels_csv import load_labels_from_csvs


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict) -> pd.DataFrame:
    tz_name = cfg['night_window']['timezone']
    air = read_air(cfg['paths']['exports_dir'], cfg['air_monitor']['device_id'], tz_name)

    feats_all = []
    for key, person in cfg['people'].items():
        wrist = read_wrist(cfg['paths']['exports_dir'], person['wrist_id'], tz_name)
        matt = read_mattress(cfg['paths']['exports_dir'], person['mattress_id'], tz_name)
        feat = build_features(
            wrist=wrist,
            mattress=matt,
            air=air,
            start_hour=cfg['night_window']['start_hour'],
            end_hour=cfg['night_window']['end_hour'],
            in_bed_value=cfg['mattress']['in_bed_value']
        )
        feat['person'] = key
        feats_all.append(feat)
    feats = pd.concat(feats_all, ignore_index=True)

    # 轻量滞后/滚动特征：在与标签合并前构造，避免信息泄露（仅用历史窗口）
    lag_base_cols = [
        'wrist_心率_mean',
        'mattress_in_bed_sum',
        'air_二氧化碳_mean',
    ]
    lag_base_cols = [c for c in lag_base_cols if c in feats.columns]
    if lag_base_cols:
        feats = feats.sort_values(['person', 'night_id']).reset_index(drop=True)
        for c in lag_base_cols:
            grp = feats.groupby('person')[c]
            feats[f'{c}_lag1'] = grp.shift(1)
            feats[f'{c}_roll3_mean'] = grp.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            feats[f'{c}_trend3'] = grp.shift(1) - grp.shift(3)

    # 标签对齐：night_id对应的日期为 night_id+1 的白天（起床后填写）
    feats['label_date'] = pd.to_datetime(feats['night_id']) + pd.Timedelta(days=1)
    feats['label_date'] = feats['label_date'].dt.date

    # 读取 labels_dir 下的CSV标签
    labels = load_labels_from_csvs(cfg)

    data = feats.merge(labels, left_on=['label_date', 'person'], right_on=['date', 'person'], how='inner')
    data = data.drop(columns=['date'])
    return data


def train(cfg_path: str = 'src/config.yaml', n_estimators: int = 400) -> None:
    cfg = load_config(cfg_path)
    data = build_dataset(cfg)

    # 目标：5个维度分 + 总分(total)
    target_cols = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'total']
    drop_cols = set(['night_id', 'label_date'] + target_cols)
    feature_cols_all = [c for c in data.columns if c not in drop_cols]

    # 数值与类别特征划分（引入 person 类别特征）
    cat_features = [c for c in ['person'] if c in feature_cols_all]
    numeric_features = [c for c in feature_cols_all if c not in cat_features and pd.api.types.is_numeric_dtype(data[c])]

    X = data[numeric_features + cat_features].copy()
    y = data[target_cols].copy()

    pre = ColumnTransformer([
        ('num', Pipeline(steps=[('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    base = RandomForestRegressor(n_estimators=n_estimators, random_state=cfg['training']['random_state'])
    model = MultiOutputRegressor(base)

    pipe = Pipeline([
        ('pre', pre),
        ('model', model)
    ])

    kf = KFold(n_splits=min(cfg['training']['cv_folds'], max(2, len(data) // 5)), shuffle=True, random_state=cfg['training']['random_state'])
    maes = []
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        maes.append(mean_absolute_error(y_te, y_pred, multioutput='raw_values'))
    cv_mae = np.vstack(maes).mean(axis=0)

    os.makedirs(cfg['paths']['models_dir'], exist_ok=True)
    meta = {
        'feature_cols': numeric_features + cat_features,
        'target_cols': target_cols,
        'cv_mae': cv_mae.tolist()
    }

    pipe.fit(X, y)
    joblib.dump(pipe, os.path.join(cfg['paths']['models_dir'], 'sleep_model.joblib'))
    with open(os.path.join(cfg['paths']['models_dir'], 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # 保存评估
    with open(os.path.join(cfg['paths']['models_dir'], 'cv_mae.json'), 'w', encoding='utf-8') as f:
        json.dump({'targets': target_cols, 'cv_mae': meta['cv_mae']}, f, ensure_ascii=False, indent=2)

    print('训练完成，CV-MAE per target:', cv_mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--n_estimators', type=int, default=400)
    args = parser.parse_args()
    train(args.config, args.n_estimators)
