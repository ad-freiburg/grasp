import asyncio
import json
import os
import random
import string
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from logging import INFO, FileHandler, Formatter, Logger
from math import ceil
from typing import Any

import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi import Request as HTTPRequest
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, conlist
from search_rdf.model import SentenceTransformerModel
from universal_ml_utils.io import dump_json, load_json
from universal_ml_utils.logging import get_logger
from universal_ml_utils.ops import partition_by
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from grasp.configs import ServerConfig
from grasp.core import generate, load_notes, setup
from grasp.model import Message
from grasp.tasks import Task
from grasp.examples import load_example_indices


class Past(BaseModel):
    messages: conlist(Message, min_length=1)  # type: ignore
    known: set[str]


class Request(BaseModel):
    task: Task
    input: Any
    knowledge_graphs: conlist(str, min_length=1)  # type: ignore
    past: Past | None = None


class State(BaseModel):
    task: Task
    selected_kgs: conlist(str, min_length=1) = Field(alias="selectedKgs")  # type: ignore
    last_input: Any | None = Field(default=None, alias="lastInput")
    last_output: dict[str, Any] | None = Field(default=None, alias="lastOutput")


ALPHABET = string.ascii_letters + string.digits


def generate_id(length: int = 6) -> str:
    return "".join(random.sample(ALPHABET, length))


class RateLimiter:
    def __init__(self, limit: int, window: int):
        self.limit = limit
        self.window = window
        self.requests: defaultdict[str, deque[float]] = defaultdict(deque)

    def check(self, ip: str) -> float | None:
        now = time.monotonic()
        timestamps = self.requests[ip]
        while timestamps and now - timestamps[0] >= self.window:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            return self.window - (now - timestamps[0])
        timestamps.append(now)
        return None


def serve(config: ServerConfig, log_level: int | str | None = None) -> None:
    # create a fast api websocket server to serve the generate_sparql function
    app = FastAPI()
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="*")  # type: ignore
    logger = get_logger("GRASP SERVER", log_level)

    # keep track of connections and limit to 10 concurrent connections
    active_connections = 0
    rate_limiter: RateLimiter | None = None
    if config.rate_limit is not None:
        rate_limiter = RateLimiter(config.rate_limit, config.rate_limit_window)
        logger.info(
            f"Rate limiting enabled: {config.rate_limit} requests "
            f"per {config.rate_limit_window}s per IP"
        )

    if config.log_file is not None:
        log_dir = os.path.dirname(config.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        logger.info(f"Logging to file: {config.log_file}")
        file_handler = FileHandler(config.log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    if config.log_outputs is not None:
        log_dir = os.path.dirname(config.log_outputs)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        output_logger = Logger("GRASP JSONL OUTPUTS")
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

    managers, models, _ = setup(config)

    examples_model = models.get(f"sentence-transformer/{config.embedding_model}")
    if examples_model is None:
        examples_model = config.embedding_model
    else:
        assert isinstance(examples_model, SentenceTransformerModel), (
            f"Expected examples embedding model to be a SentenceTransformerModel, got {type(examples_model)}"
        )

    kgs = [manager.kg for manager in managers]

    notes = {}
    kg_notes = {}
    example_indices = {}
    for task in Task:
        general_notes, kg_specific_notes = load_notes(config)
        notes[task.value] = general_notes
        kg_notes[task.value] = kg_specific_notes

        task_indices = load_example_indices(task.value, config, examples_model)
        example_indices[task.value] = task_indices

    if config.share is not None:
        share_dir = config.share

        def generate_and_check_share_id(max_retries: int = 3) -> str | None:
            share_id = generate_id()
            if not os.path.exists(os.path.join(share_dir, f"{share_id}.json")):
                return share_id
            if max_retries <= 0:
                return None
            return generate_and_check_share_id(max_retries - 1)

        @app.post("/save")
        async def save_state(request: State):
            share_id = generate_and_check_share_id()
            if share_id is None:
                logger.error(
                    "Failed to generate unique share ID after multiple attempts"
                )
                raise HTTPException(status_code=500, detail="Failed to share state")

            try:
                data = request.model_dump(by_alias=True)
                timestamp = datetime.now(timezone.utc).isoformat()
                dump_json(
                    {
                        "timestamp": timestamp,
                        "state": data,
                    },
                    os.path.join(share_dir, f"{share_id}.json"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Failed to persist state: {exc}")
                raise HTTPException(
                    status_code=500, detail="Failed to save state"
                ) from exc

            logger.info(f"Saved state {share_id}")
            return {"id": share_id}

        @app.get("/load/{share_id}")
        async def load_state(share_id: str):
            try:
                path = os.path.join(share_dir, f"{share_id}.json")
                data = load_json(path)
                return State(**data["state"])
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="State not found")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Unexpected error loading state {share_id}: {exc}")
                raise HTTPException(status_code=500, detail="Failed to load state")

    @app.get("/knowledge_graphs")
    async def get_knowledge_graphs():
        return kgs

    @app.get("/config")
    async def get_config():
        return config.model_dump()

    @app.post("/run")
    async def run_task(request: Request, http_request: HTTPRequest):
        nonlocal active_connections
        nonlocal rate_limiter

        client_ip = http_request.client.host if http_request.client else "unknown"
        prefix = f"[{client_ip}] [/run]"

        if rate_limiter is not None:
            retry_after = rate_limiter.check(client_ip)
            if retry_after is not None:
                logger.warning(
                    f"{prefix} Rate limit exceeded, try again in {ceil(retry_after):,}s"
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many requests, try again in {ceil(retry_after):,}s",
                    headers={"Retry-After": str(ceil(retry_after))},
                )

        if active_connections >= config.max_connections:
            logger.warning(
                f"{prefix} Request refused: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            raise HTTPException(
                status_code=503,
                detail="Server too busy, try again later",
            )

        active_connections += 1
        logger.info(f"{prefix} Request started ({active_connections=:,})")

        stop_event = threading.Event()

        try:
            sel = request.knowledge_graphs
            if not sel or not all(kg in kgs for kg in sel):
                logger.error(
                    f"{prefix} Unsupported knowledge graph selection:\n"
                    f"{request.model_dump_json(indent=2)}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported knowledge graph selection",
                )

            sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)

            past_messages = request.past.messages if request.past else None
            past_known = request.past.known if request.past else None

            def run_generate() -> dict:
                generator = generate(
                    request.task,
                    request.input,
                    config,
                    sel_managers,
                    kg_notes[request.task],
                    notes[request.task],
                    example_indices[request.task],
                    past_messages,
                    past_known,
                    logger,
                    yield_output=True,
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
                        logger.info(
                            f"{prefix} Client disconnected, stopping generation"
                        )
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
                    f"{prefix} Generation hit time limit of {config.max_generation_time:,} seconds"
                )
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Generation hit time limit of {config.max_generation_time:,} seconds"
                    ),
                )
            except HTTPException as e:
                raise e
            except Exception as exc:
                logger.error(f"{prefix} Unexpected error:\n{exc}")
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
            logger.info(f"{prefix} Request finished ({active_connections=:,})")

    @app.websocket("/live")
    async def live_socket(websocket: WebSocket):
        nonlocal active_connections
        nonlocal rate_limiter

        assert websocket.client is not None
        client_ip = websocket.client.host
        prefix = f"[{client_ip}] [/live]"
        await websocket.accept()

        # Check if we've reached the maximum number of connections
        if active_connections >= config.max_connections:
            logger.warning(
                f"{prefix} Connection refused: "
                f"maximum of {config.max_connections:,} active connections reached"
            )
            await websocket.close(code=1013, reason="Server too busy, try again later")
            return

        active_connections += 1
        logger.info(f"{prefix} Connected ({active_connections=:,})")
        last_active = time.monotonic()

        async def idle_checker():
            nonlocal last_active
            while True:
                await asyncio.sleep(min(5, config.max_idle_time))

                if time.monotonic() - last_active <= config.max_idle_time:
                    continue

                msg = f"Connection closed due to inactivity after {config.max_idle_time:,} seconds"
                logger.info(f"{prefix} {msg}")
                await websocket.close(code=1013, reason=msg)  # Try Again Later
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
                        f"{prefix} Invalid request:\n{json.dumps(data, indent=2)}"
                    )
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                logger.info(
                    f"{prefix} Got request:\n{request.model_dump_json(indent=2, exclude={'past'})}"
                )

                if rate_limiter is not None:
                    retry_after = rate_limiter.check(client_ip)
                    if retry_after is not None:
                        logger.warning(
                            f"{prefix} Rate limit exceeded, try again in {ceil(retry_after):,}s"
                        )
                        await websocket.close(
                            code=1013,
                            reason=f"Too many requests, try again in {ceil(retry_after):,}s",
                        )
                        break

                sel = request.knowledge_graphs
                if not sel or not all(kg in kgs for kg in sel):
                    logger.error(
                        f"{prefix} Unsupported knowledge graph selection:\n"
                        f"{request.model_dump_json(indent=2)}"
                    )
                    await websocket.send_json(
                        {"error": "Unsupported knowledge graph selection"}
                    )
                    continue

                sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)

                past_messages = None
                past_known = None
                if request.past is not None:
                    # set past
                    past_messages = request.past.messages
                    past_known = request.past.known

                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()
                stop_event = threading.Event()

                def run_generate() -> None:
                    try:
                        generator = generate(
                            request.task,
                            request.input,
                            config,
                            sel_managers,
                            kg_notes[request.task],
                            notes[request.task],
                            example_indices[request.task],
                            past_messages,
                            past_known,
                            logger,
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
                #
                # Track start time for timeout
                start_time = time.monotonic()

                try:
                    while True:
                        kind, payload = await queue.get()

                        if kind == "data":
                            # Check if we've exceeded the time limit
                            current_time = time.monotonic()
                            if current_time - start_time > config.max_generation_time:
                                msg = f"Generation hit time limit of {config.max_generation_time:,} seconds"
                                logger.warning(f"{prefix} {msg}")
                                stop_event.set()
                                await websocket.send_json({"error": msg})
                                break

                            output = payload
                            if output["type"] == "output" and output_logger is not None:
                                output_logger.info(json.dumps(output))

                            await websocket.send_json(output)
                            data = await websocket.receive_json()
                            last_active = time.monotonic()

                            if data.get("cancel", False):
                                logger.info(f"{prefix} Generation cancelled")
                                stop_event.set()
                                await websocket.send_json({"cancelled": True})
                                break

                        elif kind == "error":
                            exc = payload
                            stop_event.set()
                            logger.error(
                                f"{prefix} Unexpected error while generating:\n{exc}"
                            )
                            await websocket.send_json(
                                {"error": f"Failed to handle request:\n{exc}"}
                            )
                            break

                        elif kind == "done":
                            break

                finally:
                    stop_event.set()
                    try:
                        await producer
                    except Exception as exc:
                        logger.error(f"{prefix} Generator worker failed:\n{exc}")

        except WebSocketDisconnect:
            pass

        except Exception as e:
            logger.error(f"{prefix} Unexpected error:\n{e}")
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            idle_task.cancel()
            active_connections -= 1
            logger.info(f"{prefix} Disconnected ({active_connections=:,})")

    if config.speech_to_text is not None:
        stt_config = config.speech_to_text
        stt_client = OpenAI(
            base_url=stt_config.model_endpoint,
            api_key=stt_config.model_api_key,
            timeout=stt_config.model_timeout,
            max_retries=stt_config.num_retries,
        )
        logger.info(
            f"Speech-to-text enabled (model={stt_config.model}, "
            f"max_audio_bytes={stt_config.max_audio_bytes:,})"
        )

        stt_rate_limiter: RateLimiter | None = None
        if stt_config.rate_limit is not None:
            stt_rate_limiter = RateLimiter(
                stt_config.rate_limit, stt_config.rate_limit_window
            )
            logger.info(
                f"STT rate limiting enabled: {stt_config.rate_limit} requests "
                f"per {stt_config.rate_limit_window}s per IP"
            )

        @app.post("/transcribe")
        async def transcribe_audio(
            http_request: HTTPRequest,
            file: UploadFile = File(...),
        ):
            client_ip = http_request.client.host if http_request.client else "unknown"
            prefix = f"[{client_ip}] [/transcribe]"

            if stt_rate_limiter is not None:
                retry_after = stt_rate_limiter.check(client_ip)
                if retry_after is not None:
                    logger.warning(
                        f"{prefix} Rate limit exceeded, try again in {ceil(retry_after):,}s"
                    )
                    raise HTTPException(
                        status_code=429,
                        detail=f"Too many transcription requests, try again in {ceil(retry_after):,}s",
                        headers={"Retry-After": str(ceil(retry_after))},
                    )

            audio_bytes = await file.read()
            if not audio_bytes:
                raise HTTPException(status_code=400, detail="Empty audio file")
            if len(audio_bytes) > stt_config.max_audio_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Audio too large (max {stt_config.max_audio_bytes:,} bytes)"
                    ),
                )

            filename = file.filename or "Unknown"
            content_type = file.content_type or "Unknown"

            logger.info(
                f"{prefix} Transcribing {len(audio_bytes):,} bytes "
                f"({filename=}, {content_type=}) with model {stt_config.model}"
            )

            def run_transcription() -> str:
                result = stt_client.audio.transcriptions.create(
                    model=stt_config.model,
                    file=(filename, audio_bytes, content_type),
                    language=stt_config.language,  # type: ignore
                )
                return result.text

            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(run_transcription),
                    timeout=stt_config.model_timeout + 3.0,
                )
                logger.info(f"{prefix} Transcription successful: {text}")
            except asyncio.TimeoutError:
                logger.warning(f"{prefix} Transcription timed out")
                raise HTTPException(status_code=504, detail="Transcription timed out")
            except OpenAIError as exc:
                logger.error(f"{prefix} Transcription failed: {exc}")
                raise HTTPException(status_code=502, detail="Transcription failed")
            except Exception as exc:  # noqa: BLE001
                logger.error(f"{prefix} Unexpected transcription error: {exc}")
                raise HTTPException(
                    status_code=500, detail="Failed to transcribe audio"
                )

            return {"text": text}

    uvicorn.run(app, host="0.0.0.0", port=config.port)
