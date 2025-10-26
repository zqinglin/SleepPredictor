from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
import numpy as np


@dataclass
class NightWindow:
    start_hour: int
    end_hour: int


def assign_night_id(ts: pd.Series, start_hour: int, end_hour: int) -> pd.Series:
    local = ts.dt.tz_convert('Asia/Shanghai')
    date = local.dt.date
    hour = local.dt.hour
    # 规则：start_hour..23 属于当日night_id；0..end_hour 属于前一日night_id
    day = pd.to_datetime(date)
    anchor = day
    anchor = anchor.where(hour >= start_hour, anchor - pd.Timedelta(days=1))
    return anchor.dt.date


def _flatten_cols(prefix: str, cols) -> list:
    flat = []
    for col in pd.Index(cols).to_flat_index():
        if isinstance(col, tuple):
            name = '_'.join([str(c) for c in col if c])
        else:
            name = str(col)
        flat.append(f"{prefix}{name}")
    return flat


def aggregate_wrist(df: pd.DataFrame, night_id: pd.Series) -> pd.DataFrame:
    g = df.copy()
    g['night_id'] = night_id
    agg = g.groupby('night_id').agg({
        '心率': ['mean', 'std', 'min', 'max'],
        '血氧': ['mean', 'min'],
        '高压': 'mean',
        '低压': 'mean',
        '浅睡时间': 'max',
        '深睡时间': 'max',
        '快速眼动时间': 'max',
        '清醒时间': 'max',
        '睡眠评分': 'max'
    })
    agg.columns = _flatten_cols('wrist_', agg.columns)
    return agg.reset_index()


def aggregate_mattress(df: pd.DataFrame, night_id: pd.Series, in_bed_value: int) -> pd.DataFrame:
    g = df.copy()
    g['night_id'] = night_id
    # 在床布尔（兼容3或4为在床，不依赖单一配置）
    g['in_bed'] = g.get('在床垫状态').isin([in_bed_value, 3, 4]).astype(float)
    agg = g.groupby('night_id').agg({
        '心率': ['mean', 'std', 'min', 'max'],
        '呼吸率': ['mean', 'std', 'min', 'max'],
        'in_bed': ['mean', 'sum'],
    })
    agg.columns = _flatten_cols('mattress_', agg.columns)
    return agg.reset_index()


def aggregate_air(df: pd.DataFrame, night_id: pd.Series) -> pd.DataFrame:
    g = df.copy()
    g['night_id'] = night_id
    agg = g.groupby('night_id').agg({
        '二氧化碳': ['mean', 'max'],
        'PM2.5': ['mean', 'max'],
        'PM10': ['mean', 'max'],
        '甲醛': 'mean',
        'TVOC(挥发性有机化合物)': 'mean',
        '温度': ['mean', 'min', 'max'],
        '湿度': ['mean', 'min', 'max']
    })
    agg.columns = _flatten_cols('air_', agg.columns)
    return agg.reset_index()


def _sum_inbed_segments(mi: pd.DataFrame, gap_threshold_minutes: int = 15) -> Tuple[pd.Timestamp, pd.Timedelta]:
    if mi.empty:
        return pd.NaT, pd.Timedelta(0)
    mi = mi.sort_values('timestamp')
    gap = pd.Timedelta(minutes=gap_threshold_minutes)
    first_start = None
    start = None
    last = None
    total = pd.Timedelta(0)
    for ts in mi['timestamp']:
        if start is None:
            start = ts
            last = ts
            if first_start is None:
                first_start = ts
            continue
        if ts - last > gap:
            total += (last - start)
            start = ts
            last = ts
        else:
            last = ts
    if start is not None and last is not None:
        total += (last - start)
    return first_start, total


def build_features(wrist: pd.DataFrame, mattress: pd.DataFrame, air: pd.DataFrame, start_hour: int, end_hour: int, in_bed_value: int) -> pd.DataFrame:
    # 对齐时间戳列名
    assert 'timestamp' in wrist.columns and 'timestamp' in mattress.columns and 'timestamp' in air.columns
    wrist_nid = assign_night_id(wrist['timestamp'], start_hour, end_hour)
    matt_nid = assign_night_id(mattress['timestamp'], start_hour, end_hour)
    air_nid = assign_night_id(air['timestamp'], start_hour, end_hour)

    fw = aggregate_wrist(wrist, wrist_nid)
    fm = aggregate_mattress(mattress, matt_nid, in_bed_value)
    fa = aggregate_air(air, air_nid)

    # 逐夜合并
    feat = fw.merge(fm, on='night_id', how='outer')
    feat = feat.merge(fa, on='night_id', how='left')
    feat = feat.sort_values('night_id').reset_index(drop=True)

    # 额外客观指标：手环优先（起止）> 手环分段总和 > 床垫分段总和
    obj_frames = []

    # A) 手环起止：若存在非空的“睡眠开始时间/睡眠结束时间”，按最早开始与最晚结束计算总时长
    if {'睡眠开始时间', '睡眠结束时间'}.issubset(wrist.columns) and wrist[['睡眠开始时间', '睡眠结束时间']].notna().any().any():
        wr = wrist[['timestamp', '睡眠开始时间', '睡眠结束时间']].dropna()
        wr['night_id'] = wrist_nid.loc[wr.index]
        starts = wr.groupby('night_id')['睡眠开始时间'].min()
        ends = wr.groupby('night_id')['睡眠结束时间'].max()
        oa = pd.DataFrame({'night_id': starts.index})
        oa['sleep_start'] = pd.to_datetime(starts.values, errors='coerce')
        oa['sleep_end'] = pd.to_datetime(ends.reindex(starts.index).values, errors='coerce')
        for col in ['sleep_start', 'sleep_end']:
            s = oa[col]
            if s.dt.tz is None:
                oa[col] = s.dt.tz_localize('Asia/Shanghai', nonexistent='shift_forward', ambiguous='infer')
            else:
                oa[col] = s.dt.tz_convert('Asia/Shanghai')
        oa['sleep_duration_h'] = (oa['sleep_end'] - oa['sleep_start']).dt.total_seconds() / 3600.0
        oa['sleep_start_hour'] = oa['sleep_start'].dt.hour + oa['sleep_start'].dt.minute / 60.0
        obj_frames.append(oa[['night_id', 'sleep_duration_h', 'sleep_start_hour']])

    # B) 手环分段总和：浅/深/REM 最大值之和，推断为总睡眠时长（小时）
    cols = [c for c in ['浅睡时间', '深睡时间', '快速眼动时间'] if c in wrist.columns]
    if cols:
        wr2 = wrist[['timestamp'] + cols].copy()
        wr2['night_id'] = wrist_nid
        sums = wr2.groupby('night_id')[cols].max().sum(axis=1)
        if not sums.empty:
            ob = pd.DataFrame({'night_id': sums.index, 'sleep_duration_h': np.nan})
            # 单位推断：<=16 视为小时；介于16与24*60 视为分钟并/60；>24*60 视为异常忽略
            def conv(v):
                try:
                    if v <= 16:
                        return float(v)
                    if v <= 24 * 60:
                        return float(v) / 60.0
                    return np.nan
                except Exception:
                    return np.nan
            ob['sleep_duration_h'] = sums.apply(conv).values
            obj_frames.append(ob[['night_id', 'sleep_duration_h']])

    # C) 床垫分段总和：在床标记的多段合计，入睡时间为首段开始
    m = mattress.copy()
    m['night_id'] = matt_nid
    m['in_bed_flag'] = m.get('在床垫状态').isin([in_bed_value, 3, 4])
    oc_list = []
    for nid, grp in m.groupby('night_id'):
        mi = grp[grp['in_bed_flag'] & grp['timestamp'].notna()][['timestamp']]
        first_start, total = _sum_inbed_segments(mi, gap_threshold_minutes=15)
        if pd.notna(first_start) and total.total_seconds() > 0:
            oc_list.append({
                'night_id': nid,
                'sleep_start_hour': (first_start.tz_convert('Asia/Shanghai').hour + first_start.tz_convert('Asia/Shanghai').minute / 60.0) if first_start.tzinfo else (first_start.hour + first_start.minute / 60.0),
                'sleep_duration_h': total.total_seconds() / 3600.0
            })
    if oc_list:
        oc = pd.DataFrame(oc_list)
        obj_frames.append(oc)

    if obj_frames:
        # 依次合并，前者优先（A > B > C）
        obj = None
        for fr in obj_frames:
            obj = fr if obj is None else obj.combine_first(fr)
        feat = feat.merge(obj[['night_id', 'sleep_duration_h', 'sleep_start_hour']].drop_duplicates('night_id'), on='night_id', how='left')

    return feat
