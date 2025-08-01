import argparse
import json
import os
import time

from search_index import IndexData, SimilarityIndex

from grasp.utils import Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="File to index")
    parser.add_argument("out", type=str, help="Output index dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--example-query",
        type=str,
        default=None,
        help="Example query to find matches for",
    )
    return parser.parse_args()


def clean(s: str) -> str:
    return " ".join(s.split())


def build(args: argparse.Namespace) -> None:
    data_file = f"{args.out}/data.tsv"
    offsets_file = f"{args.out}/offsets.bin"
    print(f"Building index from {args.input} to {args.out}")
    index_dir = os.path.join(args.out, "index")
    if not os.path.exists(args.out) or args.overwrite:
        os.makedirs(index_dir, exist_ok=True)
        with open(args.input, "r") as inf, open(data_file, "w") as of:
            # write header
            of.write("question\tscore (unused)\tparaphrases\tjson_data\n")

            for line in inf:
                line = line.rstrip()

                sample = Sample(**json.loads(line))

                of.write(
                    "{}\t0\t{}\t{}\n".format(
                        clean(sample.question),
                        ";;;".join(clean(p) for p in sample.paraphrases),
                        line,
                    )
                )

        IndexData.build(data_file, offsets_file)
        data = IndexData.load(data_file, offsets_file)

        start = time.perf_counter()
        SimilarityIndex.build(
            data,
            index_dir,
            show_progress=True,
            batch_size=args.batch_size,
        )
        end = time.perf_counter()
        print(f"Built index in {end - start:.2f} seconds")

    else:
        data = IndexData.load(data_file, offsets_file)

    start = time.perf_counter()
    index = SimilarityIndex.load(data, index_dir)
    end = time.perf_counter()
    print(f"Loaded index in {end - start:.2f} seconds")

    if not args.example_query:
        return

    start = time.perf_counter()
    matches = index.find_matches(args.example_query)
    end = time.perf_counter()
    print(f"Found {len(matches)} matches in {1000 * (end - start):.2f} ms:")
    for i, (id, score) in enumerate(matches):
        print(f"{i + 1}. {score:.4f}: {index.get_name(id)}")


if __name__ == "__main__":
    build(parse_args())
