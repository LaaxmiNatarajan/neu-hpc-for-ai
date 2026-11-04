import asyncio
import json
import queue
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse


class EngineWorker:
    """
    Runs one shared inference engine in a background thread.

    HTTP requests submit work through a thread-safe command queue.
    Engine token events are forwarded to the correct asyncio queue.
    """

    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

        self.commands = queue.Queue()
        self.event_queues = {}

        self.loop = None
        self.thread = None
        self.running = False

        self.lock = threading.Lock()

        # Connect engine events to this worker.
        self.engine.event_callback = self.on_engine_event

    def start(self):
        self.loop = asyncio.get_running_loop()
        self.running = True

        self.thread = threading.Thread(
            target=self.run,
            daemon=True,
        )

        self.thread.start()

        print("[WORKER] Engine worker started.")

    def stop(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=5)

        print("[WORKER] Engine worker stopped.")

    def register_request(self, request_id):
        """
        Create and register the event queue.

        IMPORTANT:
        The HTTP handler and engine worker must use
        the same queue for each request.
        """
        event_queue = asyncio.Queue()

        with self.lock:
            self.event_queues[request_id] = event_queue

        return event_queue

    def unregister_request(self, request_id):
        with self.lock:
            self.event_queues.pop(request_id, None)

    def on_engine_event(self, event):
        request_id = event.get("request_id")

        with self.lock:
            event_queue = self.event_queues.get(request_id)

        if event_queue is None or self.loop is None:
            return

        try:
            self.loop.call_soon_threadsafe(
                event_queue.put_nowait,
                event,
            )
        except RuntimeError:
            # The event loop may already be shutting down.
            pass

    def submit(self, request_id, prompt, max_new_tokens):
        """
        Submit a request from an HTTP handler thread.
        Wait until the engine accepts the request.
        """
        done = threading.Event()
        result = {}

        self.commands.put({
            "request_id": request_id,
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "done": done,
            "result": result,
        })

        if not done.wait(timeout=120):
            raise TimeoutError(
                "Timed out waiting for the inference engine to accept the request."
            )

        if "error" in result:
            raise RuntimeError(result["error"])

    def _submit_command(self, command):
        try:
            self.engine.submit(
                request_id=command["request_id"],
                prompt=command["prompt"],
                max_new_tokens=command["max_new_tokens"],
            )

        except Exception as exc:
            command["result"]["error"] = str(exc)

        finally:
            command["done"].set()

    def _notify_all_errors(self, message):
        with self.lock:
            queues = list(self.event_queues.values())

        for event_queue in queues:
            if self.loop is not None:
                try:
                    self.loop.call_soon_threadsafe(
                        event_queue.put_nowait,
                        {
                            "event": "error",
                            "message": message,
                        },
                    )
                except RuntimeError:
                    pass

    def _fail_pending_commands(self, message):
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                break

            command["result"]["error"] = message
            command["done"].set()

    def run(self):
        print("[WORKER] Background inference loop running.")

        while self.running:

            # 1. Accept newly submitted HTTP requests.
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                command = None

            if command is not None:
                self._submit_command(command)

            # 2. Execute continuous batching steps.
            try:
                if self.engine.waiting or self.engine.active:
                    self.engine.step()

                elif command is None:
                    # Avoid busy-waiting when the engine is idle.
                    try:
                        command = self.commands.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    self._submit_command(command)

            except Exception as exc:
                message = str(exc)

                print("[ENGINE ERROR]", repr(exc))

                self.running = False

                self._notify_all_errors(message)
                self._fail_pending_commands(message)

                break

        print("[WORKER] Background inference loop exited.")


def create_app(engine, tokenizer):

    app = FastAPI(title="Qwen3 Inference Server")

    worker = EngineWorker(engine, tokenizer)

    MODEL_NAME = "Qwen/Qwen3-4B"

    @app.on_event("startup")
    async def startup():
        worker.start()

    @app.on_event("shutdown")
    async def shutdown():
        worker.stop()

    # --------------------------------------------------
    # Health check
    # --------------------------------------------------

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_NAME,
            "engine": "VectorizedContinuousBatchEngine",
        }

    # --------------------------------------------------
    # Engine metrics
    # --------------------------------------------------

    @app.get("/metrics")
    async def metrics():
        return engine.get_metrics()

    # --------------------------------------------------
    # Chat completions
    # --------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict):

        messages = payload.get("messages", [])

        if not messages:
            raise HTTPException(
                status_code=400,
                detail="messages must not be empty",
            )

        request_id = None

        try:
            # ------------------------------------------
            # 1. Convert chat messages into a prompt
            # ------------------------------------------

            if hasattr(tokenizer, "apply_chat_template"):

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            else:

                prompt = "\n".join(
                    f"{message.get('role', 'user')}: "
                    f"{message.get('content', '')}"
                    for message in messages
                )

            # ------------------------------------------
            # 2. Read generation parameters
            # ------------------------------------------

            max_new_tokens = int(
                payload.get(
                    "max_completion_tokens",
                    payload.get("max_tokens", 128),
                )
            )

            if max_new_tokens < 1:
                raise ValueError(
                    "max_tokens must be positive"
                )

            # ------------------------------------------
            # 3. Create request and shared event queue
            # ------------------------------------------

            request_id = "chatcmpl-" + str(uuid.uuid4())

            # IMPORTANT: Use the queue returned by the worker.
            event_queue = worker.register_request(request_id)

            print(
                f"[API] Request received: {request_id}, "
                f"max_new_tokens={max_new_tokens}"
            )

            # ------------------------------------------
            # 4. Submit request to inference engine
            # ------------------------------------------

            await asyncio.to_thread(
                worker.submit,
                request_id,
                prompt,
                max_new_tokens,
            )

        except TimeoutError as exc:

            if request_id is not None:
                worker.unregister_request(request_id)

            raise HTTPException(
                status_code=504,
                detail=str(exc),
            )

        except Exception as exc:

            if request_id is not None:
                worker.unregister_request(request_id)

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        # --------------------------------------------------
        # 5. Streaming response
        # --------------------------------------------------

        if payload.get("stream", False):

            async def stream_events():

                previous_text = ""

                try:

                    # Send initial assistant role.
                    initial_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.get("model", MODEL_NAME),
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                            },
                            "finish_reason": None,
                        }],
                    }

                    yield (
                        "data: "
                        + json.dumps(initial_chunk)
                        + "\n\n"
                    )

                    while True:

                        event = await asyncio.wait_for(
                            event_queue.get(),
                            timeout=600,
                        )

                        if event["event"] == "error":

                            yield (
                                "data: "
                                + json.dumps({
                                    "error": event.get(
                                        "message",
                                        "Inference engine error",
                                    )
                                })
                                + "\n\n"
                            )

                            break

                        # ----------------------------------
                        # Generated token
                        # ----------------------------------

                        if event["event"] == "token":

                            cumulative_text = tokenizer.decode(
                                event["generated_tokens"],
                                skip_special_tokens=True,
                            )

                            if cumulative_text.startswith(previous_text):

                                delta = cumulative_text[
                                    len(previous_text):
                                ]

                            else:
                                delta = cumulative_text

                            previous_text = cumulative_text

                            if delta:

                                chunk = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": payload.get(
                                        "model",
                                        MODEL_NAME,
                                    ),
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": delta
                                        },
                                        "finish_reason": None,
                                    }],
                                }

                                yield (
                                    "data: "
                                    + json.dumps(chunk)
                                    + "\n\n"
                                )

                        # ----------------------------------
                        # Request completed
                        # ----------------------------------

                        elif event["event"] == "finished":

                            finish_reason = (
                                "stop"
                                if event.get("finish_reason") == "eos"
                                else "length"
                            )

                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": payload.get(
                                    "model",
                                    MODEL_NAME,
                                ),
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }],
                            }

                            yield (
                                "data: "
                                + json.dumps(chunk)
                                + "\n\n"
                            )

                            yield "data: [DONE]\n\n"

                            break

                except asyncio.TimeoutError:

                    yield (
                        "data: "
                        + json.dumps({
                            "error": "Generation timed out."
                        })
                        + "\n\n"
                    )

                finally:
                    worker.unregister_request(request_id)

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # --------------------------------------------------
        # 6. Non-streaming response
        # --------------------------------------------------

        try:

            final_event = None

            while True:

                event = await asyncio.wait_for(
                    event_queue.get(),
                    timeout=600,
                )

                if event["event"] == "error":

                    raise RuntimeError(
                        event.get(
                            "message",
                            "Inference engine error",
                        )
                    )

                if event["event"] == "finished":
                    final_event = event
                    break

            # ----------------------------------------------
            # Decode generated tokens
            # ----------------------------------------------

            generated_tokens = final_event.get(
                "generated_tokens",
                [],
            )

            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            # ----------------------------------------------
            # Token usage
            # ----------------------------------------------

            prompt_tokens = len(
                tokenizer(
                    prompt,
                    add_special_tokens=False,
                )["input_ids"]
            )

            completion_tokens = len(generated_tokens)

            finish_reason = (
                "stop"
                if final_event.get("finish_reason") == "eos"
                else "length"
            )

            # ----------------------------------------------
            # OpenAI-compatible response
            # ----------------------------------------------

            response = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model", MODEL_NAME),
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": (
                        prompt_tokens + completion_tokens
                    ),
                },
            }

            print(
                f"[API] Request completed: {request_id}, "
                f"prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}"
            )

            return response

        except asyncio.TimeoutError:

            raise HTTPException(
                status_code=504,
                detail="Generation timed out.",
            )

        except Exception as exc:

            print(
                "[API RESPONSE ERROR]",
                repr(exc),
            )

            raise HTTPException(
                status_code=500,
                detail=str(exc),
            )

        finally:
            worker.unregister_request(request_id)

    return app
