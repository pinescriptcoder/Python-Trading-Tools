"""
Download Binance BTCUSDT 1m monthly archives, extract them,
merge all history into a single CSV file suitable for MultiCharts.

Default data directory:
    C:\\binance_data\\BTCUSDT

Usage:
    python binance_btcusdt_all.py
    python binance_btcusdt_all.py "D:\\MarketData\\BTCUSDT"
"""

import os
import sys
import glob
import zipfile
from datetime import date

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m"
START = date(2017, 8, 1)
CHUNK_SIZE = 1024 * 1024

DEFAULT_DATA_DIR = r"C:\binance_data\BTCUSDT"

BINANCE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


def get_paths():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    return {
        "data_dir": data_dir,
        "zip_dir": os.path.join(data_dir, "zip"),
        "extract_dir": os.path.join(data_dir, "csv"),
        "output_file": os.path.join(data_dir, "BTCUSDT_1m_ALL.csv"),
    }


def generate_urls():
    today = date.today()
    last = date(today.year, today.month, 1) - relativedelta(months=1)

    result = []
    current = START

    while current <= last:
        name = f"BTCUSDT-1m-{current.strftime('%Y-%m')}.zip"
        url = f"{BASE_URL}/{name}"
        result.append((url, name))
        current += relativedelta(months=1)

    return result


def download_file(url, filename, save_dir):
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"  SKIPPED (already exists): {filename}")
        return

    resp = requests.get(url, stream=True, timeout=60)

    if resp.status_code == 404:
        print(f"  NOT FOUND (404): {filename}")
        return

    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))

    with open(filepath, "wb") as f, tqdm(
        desc=filename,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"  OK: {filename}")


def download_archives(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    files = generate_urls()

    print(f"Files to download: {len(files)}\n")

    for i, (url, name) in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {name}")

        try:
            download_file(url, name, data_dir)
        except requests.RequestException as e:
            print(f"  ERROR: {e}")


def extract_all_zips(zip_dir, extract_dir):
    zips = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))

    if not zips:
        print(f"No ZIP files found in: {zip_dir}")
        return False

    os.makedirs(extract_dir, exist_ok=True)

    print(f"ZIP archives found: {len(zips)}")

    extracted_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Extracting {len(zips)} archives...")

    for zpath in zips:
        out_path = os.path.join(
            extract_dir,
            os.path.basename(zpath).replace(".zip", ".csv")
        )

        if os.path.exists(out_path):
            skipped_count += 1
            continue

        try:
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(extract_dir)
                extracted_count += 1
        except Exception as e:
            error_count += 1
            print(f"  Extraction error {os.path.basename(zpath)}: {e}")

    print(f"Archives extracted: {extracted_count}")
    print(f"Archives skipped: {skipped_count}")
    print(f"Extraction errors: {error_count}")
    print("Extraction completed\n")
    return True


def load_and_merge(extract_dir):
    csv_files = sorted(glob.glob(os.path.join(extract_dir, "*.csv")))

    print(f"CSV files found: {len(csv_files)}")

    chunks = []
    loaded_files = 0
    failed_files = 0

    for fpath in csv_files:
        try:
            df = pd.read_csv(
                fpath,
                header=None,
                names=BINANCE_COLS,
                usecols=["open_time", "open", "high", "low", "close", "volume"],
            )
            chunks.append(df)
            loaded_files += 1

        except Exception as e:
            failed_files += 1
            print(f"  Read error {os.path.basename(fpath)}: {e}")

    total = sum(len(c) for c in chunks)

    print(f"CSV files loaded: {loaded_files}")
    print(f"CSV files failed: {failed_files}")
    print(f"Rows loaded: {total:,}\n")

    return pd.concat(chunks, ignore_index=True)


def normalize_timestamps(df):
    mask_us = df["open_time"] > 100_000_000_000_000

    us_count = int(mask_us.sum())

    if us_count:
        print(f"Microsecond timestamps detected: {us_count:,}. Converting to milliseconds.")
        df.loc[mask_us, "open_time"] = df.loc[mask_us, "open_time"] // 1000

    return df


def convert_and_save(df, output_file):
    print("Converting timestamps and sorting...")

    rows_before_cleanup = len(df)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")

    nan_count = int(df["open_time"].isna().sum())

    df.dropna(subset=["open_time"], inplace=True)
    df["open_time"] = df["open_time"].astype("int64")

    if nan_count:
        print(f"Header rows removed: {nan_count:,}")

    df = normalize_timestamps(df)

    ts_min = 1_262_304_000_000
    ts_max = 1_924_992_000_000

    bad = ((df["open_time"] < ts_min) | (df["open_time"] > ts_max)).sum()

    if bad:
        print(f"Out-of-range rows removed: {bad:,}")

    df = df[(df["open_time"] >= ts_min) & (df["open_time"] <= ts_max)].copy()

    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    dt = df["dt"]

    df["Date"] = (
        dt.dt.month.astype(str).str.zfill(2) + "/" +
        dt.dt.day.astype(str).str.zfill(2) + "/" +
        dt.dt.year.astype(str)
    )

    df["Time"] = (
        dt.dt.hour.astype(str).str.zfill(2) + ":" +
        dt.dt.minute.astype(str).str.zfill(2)
    )

    df.sort_values("open_time", ascending=True, inplace=True)
    before_dedup = len(df)
    df.drop_duplicates(subset="open_time", inplace=True)
    duplicates_removed = before_dedup - len(df)
    print(f"Duplicate rows removed: {duplicates_removed:,}")

    out = df[["Date", "Time", "open", "high", "low", "close", "volume"]].copy()

    out.columns = [
        "Date", "Time", "Open", "High", "Low", "Close", "Volume"
    ]

    out.to_csv(output_file, index=False)

    print(f"Rows before cleanup: {rows_before_cleanup:,}")
    print(f"Rows saved: {len(out):,}")
    print(
        f"Range: {out['Date'].iloc[0]} {out['Time'].iloc[0]}"
        f" -> {out['Date'].iloc[-1]} {out['Time'].iloc[-1]}"
    )
    print(f"Output file: {output_file}")


def main():
    paths = get_paths()

    download_archives(paths["zip_dir"])

    if not extract_all_zips(paths["zip_dir"], paths["extract_dir"]):
        return

    df = load_and_merge(paths["extract_dir"])

    convert_and_save(df, paths["output_file"])


if __name__ == "__main__":
    try:
        main()
    finally:
        input("\nPress Enter to exit...")
