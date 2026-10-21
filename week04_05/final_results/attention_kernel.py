
import math

import torch
import cuda.tile as ct


@ct.kernel
def _attention_kernel(
    Q,
    K,
    V,
    O,
    TILE_N: ct.Constant[int],
    HEAD_DIM: ct.Constant[int],
    SCALE: ct.Constant[float],
):
    pid = ct.bid(0)

    Hq = Q.shape[1]
    Sq = Q.shape[2]
    Hkv = K.shape[1]
    Sk = K.shape[2]

    q_idx = pid % Sq
    hq = (pid // Sq) % Hq
    b = pid // (Sq * Hq)

    group_size = Hq // Hkv
    hkv = hq // group_size

    # Right-aligned query position.
    q_position = Sk - Sq + q_idx

    q4 = ct.load(
        Q,
        index=(b, hq, q_idx, 0),
        shape=(1, 1, 1, HEAD_DIM),
        padding_mode=ct.PaddingMode.ZERO,
    )

    q = ct.reshape(
        q4,
        (1, HEAD_DIM),
    ).astype(ct.float32)

    offsets = ct.arange(
        TILE_N,
        dtype=ct.int32,
    )

    num_blocks = ct.cdiv(Sk, TILE_N)

    # ---------------------------------------
    # PASS 1: Global maximum attention score
    # ---------------------------------------
    global_max = ct.full(
        (1, 1),
        float("-inf"),
        dtype=ct.float32,
    )

    for block in range(num_blocks):
        start = block * TILE_N

        if start <= q_position:
            k4 = ct.load(
                K,
                index=(b, hkv, start, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                padding_mode=ct.PaddingMode.ZERO,
            )

            kt = ct.reshape(
                k4,
                (TILE_N, HEAD_DIM),
            ).astype(ct.float32)

            scores = ct.matmul(
                q,
                ct.permute(kt, (1, 0)),
            ) * SCALE

            key_pos = start + offsets

            valid = (
                (key_pos < Sk)
                & (key_pos <= q_position)
            )

            scores = ct.where(
                ct.reshape(valid, (1, TILE_N)),
                scores,
                float("-inf"),
            )

            block_max = ct.max(
                scores,
                axis=1,
                keepdims=True,
            )

            global_max = ct.maximum(
                global_max,
                block_max,
            )

    # ---------------------------------------
    # PASS 2: Softmax denominator and output
    # ---------------------------------------
    denominator = ct.zeros(
        (1, 1),
        dtype=ct.float32,
    )

    accumulator = ct.zeros(
        (1, HEAD_DIM),
        dtype=ct.float32,
    )

    for block in range(num_blocks):
        start = block * TILE_N

        if start <= q_position:
            k4 = ct.load(
                K,
                index=(b, hkv, start, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                padding_mode=ct.PaddingMode.ZERO,
            )

            v4 = ct.load(
                V,
                index=(b, hkv, start, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                padding_mode=ct.PaddingMode.ZERO,
            )

            kt = ct.reshape(
                k4,
                (TILE_N, HEAD_DIM),
            ).astype(ct.float32)

            vt = ct.reshape(
                v4,
                (TILE_N, HEAD_DIM),
            ).astype(ct.float32)

            scores = ct.matmul(
                q,
                ct.permute(kt, (1, 0)),
            ) * SCALE

            key_pos = start + offsets

            valid = (
                (key_pos < Sk)
                & (key_pos <= q_position)
            )

            scores = ct.where(
                ct.reshape(valid, (1, TILE_N)),
                scores,
                float("-inf"),
            )

            probabilities = ct.exp(
                scores - global_max
            )

            block_denominator = ct.sum(
                probabilities,
                axis=1,
                keepdims=True,
            )

            block_output = ct.matmul(
                probabilities,
                vt,
            )

            denominator = (
                denominator + block_denominator
            )

            accumulator = (
                accumulator + block_output
            )

    result = accumulator / denominator

    result = result.astype(O.dtype)

    ct.store(
        O,
        index=(b, hq, q_idx, 0),
        tile=ct.reshape(
            result,
            (1, 1, 1, HEAD_DIM),
        ),
    )


def cutile_attention(q, k, v, tile_n=256):
    """
    CUDA Tile grouped-query causal attention.

    Q: [B, Hq, Sq, D]
    K: [B, Hkv, Sk, D]
    V: [B, Hkv, Sk, D]

    Supports right-aligned causal attention,
    including cached decoding.
    """

    if not (
        q.is_cuda
        and k.is_cuda
        and v.is_cuda
    ):
        raise ValueError(
            "Q, K and V must be CUDA tensors."
        )

    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
    ):
        raise ValueError(
            "Expected 4D Q, K and V tensors."
        )

    if k.shape != v.shape:
        raise ValueError(
            "K and V shapes must match."
        )

    B, Hq, Sq, D = q.shape
    Bk, Hkv, Sk, Dk = k.shape

    if B != Bk or D != Dk:
        raise ValueError(
            "Batch size or head dimension mismatch."
        )

    if Sq <= 0 or Sk < Sq:
        raise ValueError(
            "Require 0 < Sq <= Sk."
        )

    if Hkv <= 0 or Hq % Hkv != 0:
        raise ValueError(
            "Hq must be divisible by Hkv."
        )

    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        raise ValueError(
            "Q, K and V must have matching dtypes."
        )

    if q.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            f"Unsupported dtype: {q.dtype}"
        )

    if tile_n <= 0 or (
        tile_n & (tile_n - 1)
    ):
        raise ValueError(
            "tile_n must be a positive power of two."
        )

    if D <= 0 or (
        D & (D - 1)
    ):
        raise ValueError(
            "Head dimension must be a power of two."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    scale = 1.0 / math.sqrt(D)

    ct.launch(
        torch.cuda.current_stream(),
        (B * Hq * Sq,),
        _attention_kernel,
        (
            q,
            k,
            v,
            out,
            tile_n,
            D,
            scale,
        ),
    )

    return out
