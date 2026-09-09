import argparse

import fastapi
import requests
import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        default=9999,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--endpoints",
        type=str,
        nargs="+",
        help="GRASP endpoints for knowledge graphs given as <kg>=<endpoint>",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Year of the competition",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    app = fastapi.FastAPI(title=f"TEXT2SPARQL {args.year} API")

    dataset_to_endpoint = {}
    for endpoint in args.endpoints:
        kg, endpoint = endpoint.split("=", 1)
        dataset_to_endpoint[f"https://text2sparql.aksw.org/{args.year}/{kg}/"] = (
            kg,
            endpoint,
        )

    @app.get("/")
    async def text2sparql(question: str, dataset: str):
        if dataset not in dataset_to_endpoint:
            raise fastapi.HTTPException(404, f"Unknown dataset {dataset}")

        kg, endpoint = dataset_to_endpoint[dataset]
        try:
            output = requests.post(
                f"{endpoint}/run",
                json={
                    "task": "sparql-qa",
                    "input": question,
                    "knowledge_graphs": [kg],
                },
            )
            output.raise_for_status()
        except requests.HTTPError as e:
            raise fastapi.HTTPException(
                e.response.status_code,
                f"Error while calling backend:\n{e}",
            )
        except Exception as e:
            raise fastapi.HTTPException(
                500,
                f"Unexpected error while calling backend:\n{e}",
            )

        try:
            data = output.json()
            sparql = data["output"]["sparql"]
            assert isinstance(sparql, str)
        except Exception as e:
            raise fastapi.HTTPException(
                500,
                f"Error parsing response from backend:\n{e}",
            )

        return {"dataset": dataset, "question": question, "query": sparql}

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
