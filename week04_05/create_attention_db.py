
import io
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone

import modal

app = modal.App("qwen3-attention-test-database")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy")
)


def tensor_to_bytes(tensor):
    import numpy as np
    import torch

    buffer = io.BytesIO()
    tensor = tensor.detach().cpu().contiguous()

    # NumPy does not directly support torch.bfloat16.
    # Preserve BF16 bits using uint16 representation.
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()

    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def bytes_to_tensor(blob, dtype):
    import numpy as np
    import torch

    array = np.load(io.BytesIO(blob), allow_pickle=False)
    tensor = torch.from_numpy(array.copy())

    if dtype == "torch.bfloat16":
        if tensor.dtype != torch.uint16:
            raise ValueError("Invalid BF16 storage format.")
        return tensor.view(torch.bfloat16)

    expected_dtype = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
    }.get(dtype)

    if expected_dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"Stored dtype mismatch: {tensor.dtype} != {expected_dtype}"
        )

    return tensor


def create_database(db_path="attention_tests.sqlite3"):
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable.")

    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))

    # B, Sq, Sk, Hq, Hkv, D, dtype
    #
    # Sq = query sequence length
    # Sk = key/value sequence length
    #
    # Cached decoding uses Sq < Sk.
    test_cases = [
        # Original FP32 cases
        (2, 16, 16, 4, 2, 16, torch.float32),
        (1, 32, 32, 4, 2, 16, torch.float32),
        (2, 64, 64, 4, 2, 16, torch.float32),
        (1, 128, 128, 8, 2, 32, torch.float32),
        (2, 128, 128, 8, 2, 32, torch.float32),
        (4, 256, 256, 8, 2, 32, torch.float32),

        # Qwen3-4B full-sequence BF16 cases
        (1, 16, 16, 32, 8, 128, torch.bfloat16),
        (2, 128, 128, 32, 8, 128, torch.bfloat16),

        # Qwen3-4B cached decoding BF16 cases
        (1, 1, 128, 32, 8, 128, torch.bfloat16),
        (2, 1, 256, 32, 8, 128, torch.bfloat16),
    ]

    conn = sqlite3.connect(db_path)

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attention_test_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                q_npy BLOB NOT NULL,
                k_npy BLOB NOT NULL,
                v_npy BLOB NOT NULL,
                reference_output_npy BLOB NOT NULL
            )
        """)

        # Regenerate a clean test suite.
        conn.execute("DELETE FROM attention_test_cases")
        conn.execute(
            "DELETE FROM sqlite_sequence "
            "WHERE name = 'attention_test_cases'"
        )
        conn.commit()

        for case_index, (
            B, Sq, Sk, Hq, Hkv, D, dtype
        ) in enumerate(test_cases):

            seed = 1234 + case_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            q = torch.randn(
                (B, Hq, Sq, D),
                dtype=dtype,
                device=device,
            )
            k = torch.randn(
                (B, Hkv, Sk, D),
                dtype=dtype,
                device=device,
            )
            v = torch.randn(
                (B, Hkv, Sk, D),
                dtype=dtype,
                device=device,
            )

            scale = 1.0 / (D ** 0.5)
            group_size = Hq // Hkv

            # Expand KV heads for an independent reference.
            k_expanded = k.repeat_interleave(
                group_size, dim=1
            )
            v_expanded = v.repeat_interleave(
                group_size, dim=1
            )

            # FP32 intermediate arithmetic for reference.
            scores = torch.matmul(
                q.float(),
                k_expanded.float().transpose(-2, -1),
            ) * scale

            # Right-aligned causal mask.
            # For Sq=1, query is at position Sk-1.
            query_positions = torch.arange(
                Sk - Sq, Sk, device=device
            )
            key_positions = torch.arange(
                Sk, device=device
            )

            causal_mask = (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                causal_mask,
                float("-inf"),
            )

            probabilities = torch.softmax(
                scores, dim=-1
            )

            reference = torch.matmul(
                probabilities,
                v_expanded.float(),
            ).to(dtype)

            # Explicit additive mask ensures right-aligned
            # causal semantics for Sq != Sk.
            additive_mask = torch.zeros(
                (Sq, Sk),
                device=device,
                dtype=torch.float32,
            )
            additive_mask = additive_mask.masked_fill(
                causal_mask, float("-inf")
            )

            sdpa = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask.to(dtype),
                is_causal=False,
                enable_gqa=True,
                dropout_p=0.0,
                scale=scale,
            )

            atol = (
                1e-4 if dtype == torch.float32
                else 5e-2
            )
            rtol = (
                1e-4 if dtype == torch.float32
                else 5e-2
            )

            torch.testing.assert_close(
                reference,
                sdpa,
                atol=atol,
                rtol=rtol,
            )

            max_error = (
                reference.float() - sdpa.float()
            ).abs().max().item()

            metadata = {
                "seed": seed,
                "batch_size": B,
                "seq_len": Sq,
                "query_seq_len": Sq,
                "key_seq_len": Sk,
                "num_query_heads": Hq,
                "num_kv_heads": Hkv,
                "head_dim": D,
                "dtype": str(dtype),
                "causal": True,
                "causal_alignment": "bottom_right",
                "scale": scale,
                "device_used_to_generate": device,
                "pytorch_version": torch.__version__,
                "test_type": (
                    "synthetic_grouped_query_attention"
                ),
                "sdpa_max_absolute_error": max_error,
            }

            conn.execute(
                """
                INSERT INTO attention_test_cases (
                    created_at_utc,
                    metadata_json,
                    q_npy,
                    k_npy,
                    v_npy,
                    reference_output_npy
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(
                        timezone.utc
                    ).isoformat(),
                    json.dumps(metadata),
                    tensor_to_bytes(q),
                    tensor_to_bytes(k),
                    tensor_to_bytes(v),
                    tensor_to_bytes(reference),
                ),
            )

            print(
                f"Created case {case_index + 1}/"
                f"{len(test_cases)}: "
                f"B={B}, Sq={Sq}, Sk={Sk}, "
                f"Hq={Hq}, Hkv={Hkv}, D={D}, "
                f"dtype={dtype}, "
                f"SDPA max error={max_error:.6g}"
            )

        conn.commit()

        # Read back and verify all saved tensors.
        rows = conn.execute(
            """
            SELECT id, metadata_json, q_npy,
                   k_npy, v_npy, reference_output_npy
            FROM attention_test_cases
            ORDER BY id
            """
        ).fetchall()

        assert len(rows) == len(test_cases)

        for row in rows:
            (
                case_id,
                metadata_json,
                q_blob,
                k_blob,
                v_blob,
                ref_blob,
            ) = row

            metadata = json.loads(metadata_json)
            dtype_string = metadata["dtype"]

            q_saved = bytes_to_tensor(
                q_blob, dtype_string
            )
            k_saved = bytes_to_tensor(
                k_blob, dtype_string
            )
            v_saved = bytes_to_tensor(
                v_blob, dtype_string
            )
            ref_saved = bytes_to_tensor(
                ref_blob, dtype_string
            )

            B = metadata["batch_size"]
            Sq = metadata["query_seq_len"]
            Sk = metadata["key_seq_len"]
            Hq = metadata["num_query_heads"]
            Hkv = metadata["num_kv_heads"]
            D = metadata["head_dim"]

            assert tuple(q_saved.shape) == (
                B, Hq, Sq, D
            )
            assert tuple(k_saved.shape) == (
                B, Hkv, Sk, D
            )
            assert tuple(v_saved.shape) == (
                B, Hkv, Sk, D
            )
            assert tuple(ref_saved.shape) == (
                B, Hq, Sq, D
            )

            assert torch.isfinite(
                ref_saved.float()
            ).all()

            print(
                f"Verified SQLite row {case_id}: "
                f"Q={tuple(q_saved.shape)}, "
                f"K={tuple(k_saved.shape)}, "
                f"V={tuple(v_saved.shape)}"
            )

        print(
            f"\nPASS: Generated and verified "
            f"{len(rows)} test cases."
        )

    finally:
        conn.close()


@app.function(
    image=image,
    gpu="L40S",
    timeout=1200,
)
def run_database_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(
            temp_dir, "attention_tests.sqlite3"
        )

        create_database(db_path)

        with open(db_path, "rb") as f:
            return f.read()


@app.local_entrypoint()
def main():
    db_bytes = run_database_test.remote()

    output_path = "attention_tests.sqlite3"

    with open(output_path, "wb") as f:
        f.write(db_bytes)

    print(
        f"Saved database locally: "
        f"{os.path.abspath(output_path)}"
    )
    print(
        f"Database size: {len(db_bytes):,} bytes"
    )
