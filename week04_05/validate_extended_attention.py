import io
import json
import sqlite3
from pathlib import Path

import modal

app = modal.App("validate-extended-attention")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "numpy",
        "cuda-tile[tileiras]",
    )
    .add_local_file(
        "attention_kernel.py",
        "/root/attention_kernel.py",
    )
)

DB_NAME = "attention_tests.sqlite3"


def load_rows():
    if not Path(DB_NAME).exists():
        raise FileNotFoundError(DB_NAME)

    conn = sqlite3.connect(DB_NAME)

    try:
        rows = conn.execute(
            """
            SELECT
                id,
                metadata_json,
                q_npy,
                k_npy,
                v_npy,
                reference_output_npy
            FROM attention_test_cases
            ORDER BY id
            """
        ).fetchall()

        return rows
    finally:
        conn.close()


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
)
def validate_remote(rows):
    import numpy as np
    import torch
    from attention_kernel import cutile_attention

    def restore(blob, dtype_name):
        array = np.load(
            io.BytesIO(blob),
            allow_pickle=False,
        )

        if "bfloat16" in dtype_name:
            tensor = torch.from_numpy(
                array.copy()
            ).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(
                array.copy()
            ).to(torch.float32)

        return tensor.to("cuda").contiguous()

    results = []

    for row in rows:
        (
            test_id,
            metadata_json,
            q_blob,
            k_blob,
            v_blob,
            ref_blob,
        ) = row

        metadata = json.loads(metadata_json)
        dtype_name = metadata["dtype"]

        q = restore(q_blob, dtype_name)
        k = restore(k_blob, dtype_name)
        v = restore(v_blob, dtype_name)
        reference = restore(ref_blob, dtype_name)

        B, Hq, Sq, D = q.shape
        Sk = k.shape[-2]

        print(
            f"\nTest {test_id}: "
            f"B={B}, Sq={Sq}, Sk={Sk}, "
            f"Hq={Hq}, D={D}, dtype={dtype_name}"
        )

        try:
            with torch.inference_mode():
                actual = cutile_attention(
                    q,
                    k,
                    v,
                    tile_n=1024,
                )

            torch.cuda.synchronize()

            if actual.shape != reference.shape:
                raise AssertionError(
                    f"Shape mismatch: "
                    f"{actual.shape} vs {reference.shape}"
                )

            diff = (
                actual.float() - reference.float()
            ).abs()

            max_error = diff.max().item()
            mean_error = diff.mean().item()

            # Explicit test tolerances.
            if q.dtype == torch.bfloat16:
                atol = 2e-2
                rtol = 2e-2
            else:
                atol = 1e-4
                rtol = 1e-4

            passed = torch.allclose(
                actual.float(),
                reference.float(),
                atol=atol,
                rtol=rtol,
            )

            print("Max error:", max_error)
            print("Mean error:", mean_error)
            print("Status:", "PASS" if passed else "FAIL")

            results.append(
                {
                    "id": test_id,
                    "passed": passed,
                    "max_error": max_error,
                    "mean_error": mean_error,
                    "error": None,
                }
            )

        except Exception as exc:
            print(
                "ERROR:",
                type(exc).__name__,
                str(exc),
            )

            results.append(
                {
                    "id": test_id,
                    "passed": False,
                    "max_error": None,
                    "mean_error": None,
                    "error": str(exc),
                }
            )

    return results


@app.local_entrypoint()
def main():
    rows = load_rows()

    print("Loaded test cases:", len(rows))

    results = validate_remote.remote(rows)

    passed = sum(
        result["passed"] for result in results
    )

    failed = len(results) - passed

    print("\n=== Final validation summary ===")
    print("Total:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)

    for result in results:
        if not result["passed"]:
            print(
                "Failed test:",
                result["id"],
                result["error"] or (
                    f"max_error={result['max_error']}"
                ),
            )

    if failed:
        raise RuntimeError(
            f"{failed} attention test(s) failed."
        )

    print("PASS: All attention tests passed.")
