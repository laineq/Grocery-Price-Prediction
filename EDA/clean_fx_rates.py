#!/usr/bin/env python3
import csv
from pathlib import Path

INPUT_PATH = Path("exchange-rate/fx_rate_20170101.csv")
OUTPUT_PATH = Path("exchange-rate/fx_rate_mxn_usd_to_cad_2017_2026.csv")
START_DATE = "2017-01-01"
END_DATE = "2026-12-31"


def clean_fx_rates() -> None:
    lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()

    marker = '"OBSERVATIONS"'
    if marker not in lines:
        return

    start = lines.index(marker) + 1
    reader = csv.DictReader(lines[start:])

    rows = []
    for row in reader:
        row_date = row.get("date", "")
        if START_DATE <= row_date <= END_DATE:
            rows.append(
                {
                    "date": row_date,
                    "MXN_CAD": row.get("FXMMXNCAD", ""),
                    "USD_CAD": row.get("FXMUSDCAD", ""),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "MXN_CAD", "USD_CAD"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    clean_fx_rates()
