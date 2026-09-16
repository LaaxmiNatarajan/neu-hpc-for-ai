import modal
import subprocess
from pathlib import Path

app = modal.App("c-matrix-mult-benchmark")

image = (
    modal.Image.debian_slim()
    .apt_install("gcc", "make", "build-essential")
    .add_local_dir(Path("."), remote_path="/root/project")
)

@app.function(
    image=image,
    cpu=64,           # Modal max is 64 vCPUs per container
    memory=16384,     # 16 GB RAM
    timeout=600
)
def run_all():
    work_dir = "/root/project"
    
    print(">>> Compiling targets...")
    subprocess.run(["make", "clean"], cwd=work_dir, check=True)
    subprocess.run(["make", "all"], cwd=work_dir, check=True)

    print("\n>>> Running Unit Tests...")
    test_run = subprocess.run(
        ["./bin/test_matrix"], cwd=work_dir, capture_output=True, text=True, check=True
    )
    print(test_run.stdout)

    print(">>> Running Scaling Benchmark (1536x1536)...")
    bench_run = subprocess.run(
        ["./bin/bench_matrix", "1536"], cwd=work_dir, capture_output=True, text=True, check=True
    )
    print(bench_run.stdout)

@app.local_entrypoint()
def main():
    run_all.remote()