import os
import json
import argparse
from typing import Dict, Optional

import yaml
import pandas as pd
import joblib

from .data_io import read_wrist, read_mattress, read_air
from .features import build_features


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def infer_for_person(cfg: Dict, person_key: str, ref_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    tz_name = cfg['night_window']['timezone']
    person = cfg['people'][person_key]
    air = read_air(cfg['paths']['exports_dir'], cfg['air_monitor']['device_id'], tz_name)
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
    # 构造标签日期（night_id+1天）
    feat['label_date'] = pd.to_datetime(feat['night_id']) + pd.Timedelta(days=1)
    feat['label_date'] = feat['label_date'].dt.date

    # 选择一晚：默认最新；若指定日期，则优先精确匹配，否则回退到最近不超过该日期的一晚
    if ref_date is None:
        last_row = feat.sort_values('label_date').iloc[-1:]
        return last_row
    ref_date = pd.to_datetime(ref_date).date()
    exact = feat[feat['label_date'] == ref_date]
    if not exact.empty:
        return exact.tail(1)
    prev = feat[feat['label_date'] <= ref_date]
    if not prev.empty:
        # 回退到最近不超过该日期的可用一晚
        return prev.sort_values('label_date').tail(1)
    # 仍无则使用最新一晚，避免0样本
    return feat.sort_values('label_date').iloc[-1:]


def predict(cfg_path: str = 'src/config.yaml', person_key: str = 'person_A', ref_date: Optional[str] = None) -> None:
    cfg = load_config(cfg_path)
    model = joblib.load(os.path.join(cfg['paths']['models_dir'], 'sleep_model.joblib'))
    with open(os.path.join(cfg['paths']['models_dir'], 'meta.json'), 'r', encoding='utf-8') as f:
        meta = json.load(f)

    feat = infer_for_person(cfg, person_key, ref_date)
    X = feat[meta['feature_cols']]
    if len(X) == 0:
        raise ValueError('该日期前后无可用数据，请尝试不指定 --date 或选择其他日期')
    y_pred = model.predict(X)

    cols = meta['target_cols']
    pred = pd.DataFrame(y_pred, columns=cols)
    # 附带客观指标
    out = pd.concat([
        feat[['label_date', 'sleep_duration_h', 'sleep_start_hour']].reset_index(drop=True),
        pred
    ], axis=1)

    # 分数后处理：dim1..dim5 in [1,5] 四舍五入；total in [0,100] 四舍五入
    score_cols = [c for c in cols if c != 'total' and c in out.columns]
    if score_cols:
        out[score_cols] = out[score_cols].clip(1, 5).round(1)
    if 'total' in out.columns:
        out['total'] = out['total'].clip(0, 100).round(2)
        out['sleep_index'] = out['total']
    else:
        if score_cols:
            total_calc = out[score_cols].sum(axis=1) / 25.0 * 100.0
            out['total'] = total_calc.clip(0, 100).round(2)
            out['sleep_index'] = out['total']
        else:
            # 兜底：若既无total又无维度分，按0处理
            out['total'] = 0.0
            out['sleep_index'] = 0.0

    os.makedirs(cfg['paths']['outputs_dir'], exist_ok=True)
    save_path = os.path.join(cfg['paths']['outputs_dir'], f'prediction_{person_key}.csv')
    out.to_csv(save_path, index=False)
    print('保存预测到:', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    parser.add_argument('--person', default='person_A')
    parser.add_argument('--date', default=None, help='YYYY-MM-DD，不填则用最新一夜；无数据则回退到最近一晚')
    args = parser.parse_args()
    predict(args.config, args.person, args.date)
