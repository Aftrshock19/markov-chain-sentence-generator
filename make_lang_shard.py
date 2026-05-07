#!/usr/bin/env python3
import argparse
from pathlib import Path

import orjson
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lang-code", default="es")
    args = parser.parse_args()

    total_bytes = args.input.stat().st_size

    with args.input.open("rb") as fin, args.output.open("wb") as fout:
        with tqdm(total=total_bytes, desc="Filtering language shard", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for line in fin:
                pbar.update(len(line))
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                if not isinstance(obj, dict):
                    continue

                if str(obj.get("lang_code", "")).lower() == args.lang_code.lower():
                    fout.write(line)

if __name__ == "__main__":
    main()