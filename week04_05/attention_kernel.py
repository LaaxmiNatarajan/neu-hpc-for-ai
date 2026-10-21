"""Single-pass online-softmax causal grouped-query attention using CUDA Tile."""
import math
import torch
import cuda.tile as ct


@ct.kernel
def _attention_kernel(Q, K, V, O,
                      TILE_N: ct.Constant[int],
                      HEAD_DIM: ct.Constant[int],
                      SCALE: ct.Constant[float]):
    pid = ct.bid(0)
    Hq = Q.shape[1]
    Sq = Q.shape[2]
    Hkv = K.shape[1]
    Sk = K.shape[2]
    q_idx = pid % Sq
    hq = (pid // Sq) % Hq
    b = pid // (Sq * Hq)
    hkv = hq // (Hq // Hkv)
    q_position = Sk - Sq + q_idx

    q4 = ct.load(Q, index=(b, hq, q_idx, 0),
                 shape=(1, 1, 1, HEAD_DIM),
                 padding_mode=ct.PaddingMode.ZERO)
    q = ct.reshape(q4, (1, HEAD_DIM)).astype(ct.float32)
    offsets = ct.arange(TILE_N, dtype=ct.int32)

    running_max = ct.full((1, 1), float('-inf'), dtype=ct.float32)
    running_sum = ct.zeros((1, 1), dtype=ct.float32)
    accumulator = ct.zeros((1, HEAD_DIM), dtype=ct.float32)

    for block in range(ct.cdiv(Sk, TILE_N)):
        start = block * TILE_N
        if start <= q_position:
            # CUDA Tile indices count TILES, not individual elements.
            k4 = ct.load(K, index=(b, hkv, block, 0),
                         shape=(1, 1, TILE_N, HEAD_DIM),
                         padding_mode=ct.PaddingMode.ZERO)
            v4 = ct.load(V, index=(b, hkv, block, 0),
                         shape=(1, 1, TILE_N, HEAD_DIM),
                         padding_mode=ct.PaddingMode.ZERO)
            kt = ct.reshape(k4, (TILE_N, HEAD_DIM)).astype(ct.float32)
            vt = ct.reshape(v4, (TILE_N, HEAD_DIM)).astype(ct.float32)
            scores = ct.matmul(q, ct.permute(kt, (1, 0))) * SCALE
            key_pos = start + offsets
            valid = (key_pos < Sk) & (key_pos <= q_position)
            scores = ct.where(ct.reshape(valid, (1, TILE_N)),
                              scores, float('-inf'))

            block_max = ct.max(scores, axis=1, keepdims=True)
            new_max = ct.maximum(running_max, block_max)
            alpha = ct.exp(running_max - new_max)
            weights = ct.exp(scores - new_max)
            block_sum = ct.sum(weights, axis=1, keepdims=True)
            block_output = ct.matmul(weights, vt)

            running_sum = alpha * running_sum + block_sum
            accumulator = alpha * accumulator + block_output
            running_max = new_max

    result = (accumulator / running_sum).astype(O.dtype)
    ct.store(O, index=(b, hq, q_idx, 0),
             tile=ct.reshape(result, (1, 1, 1, HEAD_DIM)))


def cutile_attention(q, k, v, tile_n=256):
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError('Q, K and V must be CUDA tensors')
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError('Expected 4D Q, K and V tensors')
    if k.shape != v.shape:
        raise ValueError('K and V shapes must match')
    B, Hq, Sq, D = q.shape
    Bk, Hkv, Sk, Dk = k.shape
    if B != Bk or D != Dk:
        raise ValueError('Batch size or head dimension mismatch')
    if Sq <= 0 or Sk < Sq:
        raise ValueError('Require 0 < Sq <= Sk')
    if Hkv <= 0 or Hq % Hkv != 0:
        raise ValueError('Hq must be divisible by Hkv')
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError('Q, K and V must have matching dtypes')
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f'Unsupported dtype: {q.dtype}')
    if tile_n <= 0 or (tile_n & (tile_n - 1)):
        raise ValueError('tile_n must be a positive power of two')
    if D <= 0 or (D & (D - 1)):
        raise ValueError('Head dimension must be a power of two')
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    ct.launch(torch.cuda.current_stream(), (B * Hq * Sq,),
              _attention_kernel, (q, k, v, out, tile_n, D, 1.0 / math.sqrt(D)))
    return out
