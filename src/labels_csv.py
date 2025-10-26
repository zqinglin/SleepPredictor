import os
import re
from typing import Dict, List

import pandas as pd
import numpy as np


DIM_COLS = ['睡眠平静程度', '容易清醒程度', '容易入睡程度', '醒后振奋程度', '睡眠满意程度']


def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ['utf-8-sig', 'utf-8', 'gbk']
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f'无法读取CSV: {path}')


def _infer_person_from_ids(cfg: Dict, wrist_id: int, mattress_id: int) -> str:
    for key, person in cfg['people'].items():
        if person.get('wrist_id') == wrist_id and person.get('mattress_id') == mattress_id:
            return key
    # 允许仅凭其中一个ID匹配
    for key, person in cfg['people'].items():
        if person.get('wrist_id') == wrist_id or person.get('mattress_id') == mattress_id:
            return key
    raise ValueError(f'无法根据设备ID匹配人员: 手环{wrist_id} 床垫{mattress_id}')


def _parse_date_maybe_year(s: str, default_year: int = 2025) -> pd.Timestamp:
    s = str(s).strip()
    if not s:
        return pd.NaT
    # 若不含年份，则补全年份
    if re.search(r'\b20\d{2}\b', s) is None:
        s2 = f'{default_year}/{s}'
    else:
        s2 = s
    try:
        return pd.to_datetime(s2, errors='coerce')
    except Exception:
        return pd.NaT


def load_labels_from_csvs(cfg: Dict, base_dir: str = '.') -> pd.DataFrame:
    files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and ('手环' in f and '床垫' in f)]
    dfs: List[pd.DataFrame] = []
    for fname in files:
        m = re.search(r'手环(\d+).*?床垫(\d+)', fname)
        if not m:
            continue
        wrist_id = int(m.group(1))
        mattress_id = int(m.group(2))
        person_key = _infer_person_from_ids(cfg, wrist_id, mattress_id)
        path = os.path.join(base_dir, fname)
        df = _read_csv_robust(path)
        if '日期' not in df.columns:
            raise ValueError(f'{fname} 缺少列: 日期')
        for c in DIM_COLS:
            if c not in df.columns:
                raise ValueError(f'{fname} 缺少列: {c}')
        out = pd.DataFrame()
        out['date'] = df['日期'].apply(lambda x: _parse_date_maybe_year(x).date() if pd.notna(x) else np.nan)
        out['person'] = person_key
        # 统一为 dim1..dim5 的顺序（按给定列顺序）
        for i, c in enumerate(DIM_COLS, start=1):
            out[f'dim{i}'] = pd.to_numeric(df[c], errors='coerce')
        # total 100分制（5项1-5分，满分25 → 映射到100）
        out['total'] = out[[f'dim{i}' for i in range(1, 6)]].sum(axis=1) / 25.0 * 100.0
        out = out.dropna(subset=['date'])
        dfs.append(out)
    if not dfs:
        raise RuntimeError('未在根目录找到“手环XXXX 床垫YYYY.csv”的标签文件。')
    labels = pd.concat(dfs, ignore_index=True)
    labels = labels.drop_duplicates(subset=['date', 'person']).sort_values(['person', 'date'])
    return labels
