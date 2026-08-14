import argparse
import json

from grasp.sparql.utils import (
    get_basic_auth,
    get_qlever_endpoint,
    load_qlever_prefixes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get the SPARQL prefixes used by QLever instances."
    )
    parser.add_argument("name", type=str, help="Name of the public QLever endpoint")
    return parser.parse_args()


def get(args: argparse.Namespace):
    endpoint = get_qlever_endpoint(args.name)
    prefixes = load_qlever_prefixes(endpoint, get_basic_auth(args.name))
    print(json.dumps(prefixes, indent=2))


if __name__ == "__main__":
    get(parse_args())
