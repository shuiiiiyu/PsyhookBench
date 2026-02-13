#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

# ===================== 你只需要改这两行 =====================
INPUT_CSV = r"E:\Phyhookbench\TEST\label_title_final.csv"
OUT_CSV   = r"E:\Phyhookbench\TEST\label_title_final_stats\nonzero_count.csv"
# ============================================================

HOOK_COLS = [str(i) for i in range(1, 9)]  # 列名 "1"~"8"


def read_csv_auto(path: Path) -> pd.DataFrame:
    """自动尝试常见编码读取 CSV，避免 UnicodeDecodeError。"""
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def main():
    in_path = Path(INPUT_CSV).expanduser().resolve()
    out_path = Path(OUT_CSV).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{in_path}")

    df = read_csv_auto(in_path)

    missing = [c for c in HOOK_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少列：{missing}\n现有列：{list(df.columns)}")

    # 确保 1~8 列可比较，无法解析的当 0
    for c in HOOK_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 每行 1~8 中非0列数
    nonzero_count = df[HOOK_COLS].ne(0).sum(axis=1)

    # 只统计 2~8 的分布
    dist = nonzero_count.value_counts().reindex(range(2, 9), fill_value=0).sort_index()

    out_df = pd.DataFrame({
        "nonzero_cols_k": dist.index.astype(int),
        "rows": dist.values.astype(int),
    })

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("✅ 已统计 two_or_more_nonzero 的细分分布（2~8）：")
    print(out_df.to_string(index=False))
    print(f"✅ 已保存：{out_path}")


if __name__ == "__main__":
    main()
