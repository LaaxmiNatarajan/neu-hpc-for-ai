
import subprocess
import modal


# --------------------------------------------------
# Configuration
# --------------------------------------------------

APP_NAME = "qwen3-vllm-baseline"
MODEL_NAME = "Qwen/Qwen3-4B"

GPU = "H100:1"
PORT = 8000

app = modal.App(APP_NAME)


# --------------------------------------------------
# Container image
# --------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_LOG_STATS_INTERVAL": "1",
    })
)


# --------------------------------------------------
# Persistent model and compilation caches
# --------------------------------------------------

hf_cache = modal.Volume.from_name(
    "huggingface-cache",
    create_if_missing=True,
)

vllm_cache = modal.Volume.from_name(
    "vllm-cache",
    create_if_missing=True,
)


# --------------------------------------------------
# vLLM server
# --------------------------------------------------

@app.server(
    image=image,
    gpu=GPU,
    scaledown_window=15 * 60,
    startup_timeout=10 * 60,
    target_concurrency=8,
    port=PORT,
    routing_region="us-east",
    unauthenticated=True,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
class VLLMServer:

    @modal.enter()
    def start(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,

            "--host",
            "0.0.0.0",

            "--port",
            str(PORT),

            "--served-model-name",
            MODEL_NAME,

            "--dtype",
            "bfloat16",

            "--max-model-len",
            "4096",

            "--max-num-seqs",
            "64",

            "--gpu-memory-utilization",
            "0.90",

            "--disable-prefix-caching",

            "--reasoning-parser",
            "qwen3",
        ]

        print("Starting vLLM:")
        print(" ".join(cmd))

        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
