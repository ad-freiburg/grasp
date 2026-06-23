import argparse
import asyncio
import json
import os
import threading
import time
from logging import INFO, FileHandler, Logger
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi import Request as HTTPRequest
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from universal_ml_utils.configuration import load_config
from universal_ml_utils.logging import get_logger

from grasp.baselines.grisp.run import (
    GRISPModel,
    GRISPRunConfig,
    generate,
    load_model_and_tokenizer,
)
from grasp.baselines.grisp.train import GRISPTrainConfig
from grasp.baselines.grisp.utils import load_sparql_parser
from grasp.manager import load_kg_manager

active_connections = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve GRISP model over HTTP/WebSocket"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to GRISP server configuration YAML",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


class GRISPServerConfig(GRISPRunConfig):
    run_directory: str
    selection_run: str | None = None
    device: str = "auto"
    dtype: str = "auto"

    port: int = 6790
    max_connections: int = 10
    max_generation_time: int = 300
    max_idle_time: int = 300
    log_outputs: str | None = None


class Request(BaseModel):
    question: str


def serve(config: GRISPServerConfig, log_level: int | str | None = None) -> None:
    app = FastAPI()
    logger = get_logger("GRISP SERVER", log_level)
    if config.log_outputs is not None:
        os.makedirs(os.path.dirname(config.log_outputs), exist_ok=True)
        output_logger = Logger("GRISP JSONL OUTPUTS")
        output_logger.addHandler(
            FileHandler(config.log_outputs, mode="a", encoding="utf-8")
        )
        output_logger.setLevel(INFO)
    else:
        output_logger = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # load model(s)
    train_cfg_path = os.path.join(config.run_directory, "config.yaml")
    train_cfg = GRISPTrainConfig(**load_config(train_cfg_path))

    logger.info(f"Loading model from {config.run_directory}")
    model, skeleton_tokenizer = load_model_and_tokenizer(
        config.run_directory,
        config.device,
        config.dtype,
        logger,
    )
    skeleton_model = GRISPModel(model, config.skeleton_disable_adapter)

    if train_cfg.type == "skeleton" and config.selection_run is not None:
        logger.info(f"Loading selection model from {config.selection_run}")
        sel_model, selection_tokenizer = load_model_and_tokenizer(
            config.selection_run,
            config.device,
            config.dtype,
            logger,
        )
        selection_model = GRISPModel(sel_model, config.selection_disable_adapter)
    else:
        selection_model = GRISPModel(model, config.selection_disable_adapter)
        selection_tokenizer = skeleton_tokenizer

    # load KG manager
    manager = load_kg_manager(config.knowledge_graph)

    manager.load_models()

    # load parser
    parser = load_sparql_parser()

    logger.info(f"GRISP config:\n{config.model_dump_json(indent=2)}")

    logger.info(
        f"GRISP server ready: kg={config.knowledge_graph.kg}, "
        f"model={skeleton_model.model.config.name_or_path}"  # type: ignore
    )

    @app.get("/knowledge_graphs")
    async def _knowledge_graphs():
        return [config.knowledge_graph.kg]

    @app.get("/config")
    async def _config():
        return config.model_dump()

    @app.post("/run")
    async def _run(request: Request, http_request: HTTPRequest):
        global active_connections

        if active_connections >= config.max_connections:
            logger.warning(
                "HTTP run request refused: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            raise HTTPException(
                status_code=503,
                detail="Server too busy, try again later",
            )

        active_connections += 1
        logger.info(f"HTTP run request started ({active_connections=:,})")

        stop_event = threading.Event()

        try:

            def run_generate() -> dict:
                generator = generate(
                    skeleton_model,
                    skeleton_tokenizer,
                    config,
                    request.question,
                    manager,
                    parser,
                    logger,
                    selection_model,
                    selection_tokenizer,
                )

                output = None
                for output in generator:
                    if stop_event.is_set():
                        break

                if output is None:
                    raise RuntimeError("No output produced")

                return output

            async def monitor_disconnect():
                while not stop_event.is_set():
                    if await http_request.is_disconnected():
                        logger.info("Client disconnected, stopping generation")
                        stop_event.set()
                        return
                    await asyncio.sleep(1)

            disconnect_task = asyncio.create_task(monitor_disconnect())

            try:
                output = await asyncio.wait_for(
                    asyncio.to_thread(run_generate),
                    timeout=config.max_generation_time,
                )
            except asyncio.TimeoutError:
                stop_event.set()
                logger.warning(
                    f"Generation hit time limit of {config.max_generation_time:,} seconds"
                )
                raise HTTPException(
                    status_code=504,
                    detail=f"Generation hit time limit of {config.max_generation_time:,} seconds",
                )
            except HTTPException as e:
                raise e
            except Exception as exc:
                logger.error(f"Unexpected error with HTTP run request:\n{exc}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to handle request:\n{exc}",
                )
            finally:
                stop_event.set()
                disconnect_task.cancel()

            if stop_event.is_set() and await http_request.is_disconnected():
                return {}

            if output_logger is not None:
                output_logger.info(json.dumps(output))

            return output

        finally:
            active_connections -= 1
            logger.info(f"HTTP run request finished ({active_connections=:,})")

    @app.websocket("/live")
    async def _live(websocket: WebSocket):
        global active_connections
        assert websocket.client is not None
        client = f"{websocket.client.host}:{websocket.client.port}"
        await websocket.accept()

        if active_connections >= config.max_connections:
            logger.warning(
                f"Connection from {client} immediately closed: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            await websocket.close(code=1013, reason="Server too busy, try again later")
            return

        active_connections += 1
        logger.info(f"{client} connected ({active_connections=:,})")
        last_active = time.monotonic()

        async def idle_checker():
            nonlocal last_active
            while True:
                await asyncio.sleep(min(5, config.max_idle_time))
                if time.monotonic() - last_active <= config.max_idle_time:
                    continue
                msg = f"Connection closed due to inactivity after {config.max_idle_time:,} seconds"
                logger.info(f"{client}: {msg}")
                await websocket.close(code=1013, reason=msg)
                break

        idle_task = asyncio.create_task(idle_checker())

        try:
            while True:
                data = await websocket.receive_json()
                last_active = time.monotonic()
                try:
                    request = Request(**data)
                except Exception:
                    logger.error(
                        f"Invalid request from {client}:\n{json.dumps(data, indent=2)}"
                    )
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                logger.info(f"Processing request from {client}: {request.question}")

                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
                stop_event = threading.Event()

                def run_generate() -> None:
                    try:
                        generator = generate(
                            skeleton_model,
                            skeleton_tokenizer,
                            config,
                            request.question,
                            manager,
                            parser,
                            logger,
                            selection_model,
                            selection_tokenizer,
                            yield_output=True,
                        )

                        for output in generator:
                            if stop_event.is_set():
                                break
                            asyncio.run_coroutine_threadsafe(
                                queue.put(("data", output)),
                                loop,
                            ).result()

                    except Exception as exc:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("error", exc)),
                            loop,
                        ).result()
                    finally:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("done", None)),
                            loop,
                        ).result()

                producer = asyncio.create_task(asyncio.to_thread(run_generate))
                start_time = time.monotonic()

                try:
                    while True:
                        kind, payload = await queue.get()

                        if kind == "data":
                            current_time = time.monotonic()
                            if current_time - start_time > config.max_generation_time:
                                msg = f"Generation hit time limit of {config.max_generation_time:,} seconds"
                                logger.warning(msg)
                                stop_event.set()
                                await websocket.send_json({"error": msg})
                                break

                            if (
                                payload.get("type") == "output"
                                and output_logger is not None
                            ):
                                output_logger.info(json.dumps(payload))

                            await websocket.send_json(payload)
                            ack = await websocket.receive_json()
                            last_active = time.monotonic()

                            if ack.get("cancel", False):
                                logger.info(f"Generation cancelled by {client}")
                                stop_event.set()
                                await websocket.send_json({"cancelled": True})
                                break

                        elif kind == "error":
                            stop_event.set()
                            logger.error(
                                f"Unexpected error while generating for {client}:\n{payload}"
                            )
                            await websocket.send_json(
                                {"error": f"Failed to handle request:\n{payload}"}
                            )
                            break

                        elif kind == "done":
                            break

                finally:
                    stop_event.set()
                    try:
                        await producer
                    except Exception as exc:
                        logger.error(f"Generator worker for {client} failed:\n{exc}")

        except WebSocketDisconnect:
            pass

        except Exception as e:
            logger.error(f"Unexpected error with {client}:\n{e}")
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            idle_task.cancel()
            active_connections -= 1
            logger.info(f"{client} disconnected ({active_connections=:,})")

    uvicorn.run(app, host="0.0.0.0", port=config.port)


def main(args: argparse.Namespace) -> None:
    config = GRISPServerConfig(**load_config(args.config))
    serve(config, args.log_level)


if __name__ == "__main__":
    main(parse_args())
