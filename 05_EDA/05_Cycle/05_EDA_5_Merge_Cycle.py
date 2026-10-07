"""
cycle_rows.csv + cycle_units.csv merge

목적
  05_EDA_4_Outlier_Cycle.py 결과인 행 단위 Table(cycle_rows)에 Cycle 단위 Table(cycle_units)의 지표를 붙인다.

원칙
  - key = source / segment_id / cycle_id, cycle_rows 기준 left merge (행 수 유지)
  - cycle_units는 key당 1행 (many_to_one 검증)
  - cycle_eval == global_only 행은 cycle에 속하지 않아 cycle_id가 비어 있음 → units 컬럼은 NaN

실행:  python 05_EDA_5_Merge_Cycle.py   (05_EDA_4_Outlier_Cycle.py 먼저 실행)
출력:  results/cycle_outlier/merged_cycle_rows_cycle_units.csv
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "cycle_outlier"
KEY = ["source", "segment_id", "cycle_id"]


def main():
    rows = pd.read_csv(OUT / "cycle_rows.csv")
    units = pd.read_csv(OUT / "cycle_units.csv")
    rows["cycle_id"] = rows["cycle_id"].astype("Int64")     # NaN 때문에 float로 읽힘
    int_cols = units.select_dtypes("int64").columns          # 미매칭 행 NaN 때문에 float로 바뀌지 않도록
    units[int_cols] = units[int_cols].astype("Int64")

    merged = rows.merge(units, on=KEY, how="left", validate="many_to_one", indicator=True)
    assert len(merged) == len(rows)
    unmatched = merged.loc[merged["_merge"] == "left_only"]
    assert unmatched["cycle_id"].isna().all(), "cycle_id가 있는데 units에 없는 행 존재"
    merged = merged.drop(columns="_merge")

    out = OUT / "merged_cycle_rows_cycle_units.csv"
    merged.to_csv(out, index=False)
    print(f"rows {len(rows)} + units {len(units)} → {merged.shape}  (units 미매칭 {len(unmatched)}행: "
          f"{unmatched['cycle_eval'].value_counts().to_dict()})")
    print("저장:", out)


if __name__ == "__main__":
    main()
