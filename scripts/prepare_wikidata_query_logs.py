import argparse
import json
import os
import re
from csv import DictReader
from typing import TextIO
from urllib.parse import unquote_plus

from tqdm import tqdm


def clean(sparql: str) -> str:
    return re.sub(r"\s+", " ", sparql, flags=re.DOTALL).strip()


def prepare_file(
    in_file: str,
    out_files: dict[str, TextIO],
    seen: set[str],
) -> tuple[int, int]:
    num_total = 0
    num_duplicate = 0

    reader = DictReader(
        open(in_file, "r"),
        delimiter="\t",
        fieldnames=["sparql", "timestamp", "source", "user_agent"],
    )

    for row in reader:
        if row["source"] not in out_files:
            continue

        sparql = clean(unquote_plus(row["sparql"]))
        num_total += 1
        if sparql in seen:
            num_duplicate += 1
            continue

        seen.add(sparql)
        out_files[row["source"]].write(json.dumps(sparql) + "\n")

    return num_total, num_duplicate


def prepare(args: argparse.Namespace):
    sources = []
    if not args.robotic_only:
        sources.append("organic")
    if not args.organic_only:
        sources.append("robotic")

    files = {}
    for source in sources:
        out_file = os.path.join(args.output, f"{source}.raw.jsonl")
        if os.path.exists(out_file) and not args.overwrite:
            print(f"Output file for {source} in {args.output} already exist")
            continue
        files[source] = open(out_file, "w")

    if not files:
        return

    num_total = 0
    num_duplicate = 0
    seen = set()

    for file in tqdm(
        args.input,
        desc="Processing files",
        leave=False,
        disable=not args.progress,
    ):
        total, duplicate = prepare_file(file, files, seen)
        num_total += total
        num_duplicate += duplicate

    for f in files.values():
        f.close()

    print(
        f"{num_duplicate:,} / {num_total:,} duplicates "
        f"({num_duplicate / max(num_total, 1):.1%})"
    )
    print(f"Sources: {sources}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="+", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--organic-only", action="store_true")
    source.add_argument("--robotic-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args)
