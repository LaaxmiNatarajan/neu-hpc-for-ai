
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("expand-attention-db")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy")
)

DB_NAME = "attention_tests.sqlite3"

TEST_CASES = [
    # B, Hq, Hkv, Sq, Sk, D, dtype
    (1, 8, 2, 512, 512, 32, "float32"),
    (2, 32, 8, 512, 512, 128, "bfloat16"),
    (1, 32, 8, 1024, 1024, 128, "bfloat16"),
    (2, 32, 8, 1, 512, 128, "bfloat16"),
    (2, 32, 8, 1, 1024, 128, "bfloat16"),
    (2, 32, 8, 64, 512, 128, "bfloat16"),
]


def to_npy_bytes(tensor):
    import numpy as np
    import torch

    tensor = tensor.detach().cpu().contiguous()

    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()

    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
)
def generate_cases():
    import torch
    import torch.nn.functional as F

    rows = []

    for index, config in enumerate(TEST_CASES):
        B, Hq, Hkv, Sq, Sk, D, dtype_name = config

        dtype = (
            torch.bfloat16
            if dtype_name == "bfloat16"
            else torch.float32
        )

        seed = 20260 + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        q = torch.randn(
            B, Hq, Sq, D,
            device="cuda",
            dtype=dtype,
        )
        k = torch.randn(
            B, Hkv, Sk, D,
            device="cuda",
            dtype=dtype,
        )
        v = torch.randn(
            B, Hkv, Sk, D,
            device="cuda",
            dtype=dtype,
        )

        # Right-aligned causal mask.
        # Supports both prefill and cached decoding.
        query_positions = (
            torch.arange(Sq, device="cuda") + Sk - Sq
        )
        key_positions = torch.arange(Sk, device="cuda")

        allowed = (
            key_positions[None, :]
            <= query_positions[:, None]
        )

        causal_mask = allowed.view(1, 1, Sq, Sk)

        with torch.inference_mode():
            reference = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )

        metadata = {
            "test_type": "synthetic_grouped_query_attention",
            "batch_size": B,
            "num_query_heads": Hq,
            "num_kv_heads": Hkv,
            "seq_len": Sq,
            "query_seq_len": Sq,
            "kv_seq_len": Sk,
            "head_dim": D,
            "dtype": str(dtype),
            "causal": True,
            "scale": D ** -0.5,
            "seed": seed,
            "pytorch_version": torch.__version__,
            "device_used_to_generate": "cuda",
        }

        rows.append(
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata),
                to_npy_bytes(q),
                to_npy_bytes(k),
                to_npy_bytes(v),
                to_npy_bytes(reference),
            )
        )

        print(
            f"Generated case {index + 1}: "
            f"B={B}, Sq={Sq}, Sk={Sk}, "
            f"Hq={Hq}, Hkv={Hkv}, "
            f"D={D}, dtype={dtype_name}"
        )

    return rows


@app.local_entrypoint()
def main():
    db_path = Path(DB_NAME)

    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. "
            "Run your original create_attention_db.py first."
        )

    rows = generate_cases.remote()

    conn = sqlite3.connect(db_path)

    try:
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(attention_test_cases)"
            )
        }

        required = {
            "created_at_utc",
            "metadata_json",
            "q_npy",
            "k_npy",
            "v_npy",
            "reference_output_npy",
        }

        missing = required - columns

        if missing:
            raise RuntimeError(
                f"Database schema missing columns: {missing}"
            )

        existing_rows = conn.execute(
            "SELECT metadata_json FROM attention_test_cases"
        ).fetchall()

        existing_signatures = set()

        def signature(metadata):
            return (
                metadata.get("batch_size"),
                metadata.get("num_query_heads"),
                metadata.get("num_kv_heads"),
                metadata.get(
                    "query_seq_len",
                    metadata.get("seq_len"),
                ),
                metadata.get(
                    "kv_seq_len",
                    metadata.get("seq_len"),
                ),
                metadata.get("head_dim"),
                metadata.get("dtype"),
            )

        for (metadata_json,) in existing_rows:
            existing_signatures.add(
                signature(json.loads(metadata_json))
            )

        inserted = 0

        for row in rows:
            metadata = json.loads(row[1])
            sig = signature(metadata)

            if sig in existing_signatures:
                print("Skipping existing:", sig)
                continue

            conn.execute(
                """
                INSERT INTO attention_test_cases (
                    created_at_utc,
                    metadata_json,
                    q_npy,
                    k_npy,
                    v_npy,
                    reference_output_npy
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                row,
            )

            existing_signatures.add(sig)
            inserted += 1

        conn.commit()

        total = conn.execute(
            "SELECT COUNT(*) FROM attention_test_cases"
        ).fetchone()[0]

        print("\n=== Database summary ===")
        print("New cases inserted:", inserted)
        print("Total cases:", total)
        print("Database:", db_path.resolve())

    finally:
        conn.close()
