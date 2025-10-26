import re
from typing import Dict, List

import pdfplumber
import pandas as pd


def extract_scores_from_pdf(pdf_path: str) -> pd.DataFrame:
    rows: List[Dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for tbl in tables:
                # 过滤空表
                if not tbl or len(tbl) < 2:
                    continue
                # 尝试识别表头
                header = [c.strip() if isinstance(c, str) else c for c in tbl[0]]
                # 简单启发：含有“日期/时间/评分”等关键词
                header_text = ''.join([h or '' for h in header])
                if not any(k in header_text for k in ['日期', '时间', '评分', '分数', '睡眠']):
                    continue
                for r in tbl[1:]:
                    cells = [(c or '').strip() if isinstance(c, str) else '' for c in r]
                    if not any(cells):
                        continue
                    row = {'raw': cells}
                    rows.append(row)
    # 轻量规则解析：尝试用正则抓取
    parsed: List[Dict] = []
    for row in rows:
        text = ' '.join(row['raw'])
        # 日期
        m_date = re.search(r'(20\d{2}[-/.]\d{1,2}[-/.]\d{1,2})', text)
        date_str = m_date.group(1) if m_date else None
        # 五个维度分与总分（示例：\d{1,3}）
        numbers = [int(x) for x in re.findall(r'(?:^|[^\d])(\d{1,3})(?:[^\d]|$)', text)]
        # 启发式：取前5个为子项，若存在第6个作为总分
        dims = numbers[:5] if len(numbers) >= 5 else None
        total = numbers[5] if len(numbers) >= 6 else None
        if date_str and dims:
            parsed.append({
                'date': pd.to_datetime(date_str).date(),
                'dim1': dims[0], 'dim2': dims[1], 'dim3': dims[2], 'dim4': dims[3], 'dim5': dims[4],
                'total': total
            })
    return pd.DataFrame(parsed).drop_duplicates()
