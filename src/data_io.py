import os
import re
import io
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd


@dataclass
class DatasetPaths:
    exports_dir: str


def _find_file(exports_dir: str, keyword: str) -> Optional[str]:
    for fname in os.listdir(exports_dir):
        if keyword in fname and fname.endswith('.csv'):
            return os.path.join(exports_dir, fname)
    return None


def _sanitize_bracket_arrays(text: str) -> str:
    # 将方括号内的逗号替换为分号，避免CSV误切分
    def repl(m: re.Match) -> str:
        inner = m.group(0)
        return inner.replace(',', ';')
    # 非贪婪匹配单行内的方括号片段
    return re.sub(r"\[[^\[\]\n]*\]", repl, text)


def _read_csv_robust(path: str) -> pd.DataFrame:
    attempts = [
        {'encoding': 'utf-8-sig', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'utf-8', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'gbk', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'cp936', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'latin1', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'ISO-8859-1', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'cp1252', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'utf-16', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'utf-16le', 'engine': 'python', 'on_bad_lines': 'skip'},
        {'encoding': 'utf-16be', 'engine': 'python', 'on_bad_lines': 'skip'},
    ]
    last_err = None
    for opts in attempts:
        try:
            return pd.read_csv(path, sep=',', **opts)
        except Exception as e:
            last_err = e
            continue
    # 兜底：预处理方括号数组
    try:
        with open(path, 'r', encoding='latin1', errors='ignore') as f:
            raw = f.read()
        sanitized = _sanitize_bracket_arrays(raw)
        return pd.read_csv(io.StringIO(sanitized), sep=',', engine='python', on_bad_lines='skip')
    except Exception as e2:
        last_err = e2
    raise last_err if last_err else RuntimeError(f'无法读取CSV: {path}')


def _localize_asia_shanghai(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts, errors='coerce')
    return ts.dt.tz_localize('Asia/Shanghai', nonexistent='shift_forward', ambiguous='infer')


def read_wrist(exports_dir: str, device_id: int, tz_name: str) -> pd.DataFrame:
    path = _find_file(exports_dir, f"手环{device_id}_")
    if path is None:
        raise FileNotFoundError(f"未找到手环CSV: 手环{device_id}_*.csv")
    df = _read_csv_robust(path)
    df.rename(columns={"时间": "timestamp"}, inplace=True)
    df['timestamp'] = _localize_asia_shanghai(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    # 统一重要字段存在性
    for col in ['心率', '血氧', '高压', '低压', '浅睡时间', '深睡时间', '快速眼动时间', '清醒时间', '睡眠状态', '睡眠开始时间', '睡眠结束时间', '睡眠评分']:
        if col not in df.columns:
            df[col] = pd.NA
    # 可选：删除可能仍含有长数组的列，避免后续特征工程受影响
    for noisy_col in ['连续血氧', '连续心率']:
        if noisy_col in df.columns:
            df.drop(columns=[noisy_col], inplace=True)
    return df


def read_mattress(exports_dir: str, device_id: int, tz_name: str) -> pd.DataFrame:
    path = _find_file(exports_dir, f"智慧床垫{device_id}_")
    if path is None:
        raise FileNotFoundError(f"未找到床垫CSV: 智慧床垫{device_id}_*.csv")
    df = _read_csv_robust(path)
    df.rename(columns={"时间": "timestamp"}, inplace=True)
    df['timestamp'] = _localize_asia_shanghai(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    # 关键字段兜底
    for col in ['心率', '呼吸率', '在床垫状态']:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def read_air(exports_dir: str, device_id: int, tz_name: str) -> pd.DataFrame:
    path = _find_file(exports_dir, f"空气监测{device_id}_")
    if path is None:
        raise FileNotFoundError(f"未找到空气监测CSV: 空气监测{device_id}_*.csv")
    df = _read_csv_robust(path)
    df.rename(columns={"时间": "timestamp"}, inplace=True)
    df['timestamp'] = _localize_asia_shanghai(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df
