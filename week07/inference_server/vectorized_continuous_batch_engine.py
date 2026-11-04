import time
from collections import deque

import torch
import torch.nn.functional as F

from paged_kv_cache import PagedKVCache
from batched_paged_cache import BatchedPagedCache
from request_state import RequestState


class DynamicVectorizedPagedCache:
    """
    Dynamic batch view over a shared paged KV-cache manager.

    Request IDs and append lengths can change between model forwards.
    Each request retains its own physical KV blocks.
    """

    def __init__(
        self,
        config,
        max_batch_size,
        max_cache_len,
        block_size=16,
        dtype=torch.bfloat16,
        device="cuda",

    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_cache_len = max_cache_len
        self.max_batch_size = max_batch_size

        self.num_layers = config.num_hidden_layers
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )

        blocks_per_request = (
            max_cache_len + block_size - 1
        ) // block_size

        self.manager = PagedKVCache(
            num_layers=self.num_layers,
            num_blocks=max_batch_size * blocks_per_request,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        self.paged = BatchedPagedCache(self.manager)

        self.request_ids = []
        self.append_lengths = []

    def activate_request(self, request_id):
        request_id = str(request_id)

        if self.paged.has_request(request_id):
            raise ValueError(f"Request already active: {request_id}")

        if len(self.paged.active_sequences) >= self.max_batch_size:
            raise RuntimeError("Paged cache request capacity exhausted.")

        self.paged.create_request(request_id)

    def activate_batch(self, request_ids):
        request_ids = [str(x) for x in request_ids]

        if len(request_ids) > self.max_batch_size:
            raise ValueError("Batch exceeds max_batch_size.")

        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Duplicate request IDs in batch.")

        for request_id in request_ids:
            if not self.paged.has_request(request_id):
                self.activate_request(request_id)

    def set_batch(self, request_ids, append_lengths):
        request_ids = [str(x) for x in request_ids]

        if not request_ids:
            raise ValueError("Cannot set an empty model batch.")

        if len(request_ids) != len(append_lengths):
            raise ValueError("Request IDs and append lengths must match.")

        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Request IDs must be unique.")

        if any(not self.paged.has_request(x) for x in request_ids):
            raise KeyError("Every batch request must have an active cache.")

        if any(int(x) < 1 for x in append_lengths):
            raise ValueError("Append lengths must be positive.")

        self.request_ids = request_ids
        self.append_lengths = [int(x) for x in append_lengths]

    def get_lengths(self, layer_idx=0):
        return [
            self.paged.get_seq_length(request_id, layer_idx)
            for request_id in self.request_ids
        ]

    def get_seq_length(self, layer_idx=0):
        return max(self.get_lengths(layer_idx), default=0)

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_max_length(self):
        return self.max_cache_len

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_cache_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape != value_states.shape:
            raise ValueError("Key and value shapes must match.")

        if key_states.shape[0] != len(self.request_ids):
            raise ValueError("Model batch size does not match cache batch.")

        if key_states.shape[2] != max(self.append_lengths):
            raise ValueError("Query length does not match append lengths.")

        # Append each request's valid tokens independently.
        for batch_idx, request_id in enumerate(self.request_ids):
            count = self.append_lengths[batch_idx]

            self.paged.append(
                request_id,
                layer_idx,
                key_states[batch_idx:batch_idx + 1, :, :count, :],
                value_states[batch_idx:batch_idx + 1, :, :count, :],
            )

        # Gather each sequence and pad to the longest sequence.
        lengths = self.get_lengths(layer_idx)
        max_length = max(lengths)

        keys = []
        values = []

        for request_id, length in zip(self.request_ids, lengths):
            key, value = self.paged.get_kv(request_id, layer_idx)

            if length < max_length:
                padding = max_length - length
                key = F.pad(key, (0, 0, 0, padding))
                value = F.pad(value, (0, 0, 0, padding))

            keys.append(key)
            values.append(value)

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)

    def release_request(self, request_id):
        request_id = str(request_id)

        if self.paged.has_request(request_id):
            self.paged.release_request(request_id)

    def has_request(self, request_id):
        return self.paged.has_request(str(request_id))

    def memory_mib(self):
        return self.manager.memory_mib()


class VectorizedContinuousBatchEngine:
    """
    Continuous admission with batched variable-length prefill and decode.

    A model forward processes an entire prefill or decode batch.
    Requests may finish independently and release their cache blocks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size=4,
        max_seq_len=256,
        block_size=16,
        eos_enabled=False,
        event_callback=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.eos_enabled = eos_enabled
        self.event_callback = event_callback
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        self.cache = DynamicVectorizedPagedCache(
            config=self.config,
            max_batch_size=max_batch_size,
            max_cache_len=max_seq_len,
            block_size=block_size,
            dtype=self.dtype,
            device=self.device,
        )

        self.waiting = deque()
        self.active = {}
        self.completed = []
        self.requests = {}

        self.max_active_seen = 0

    def submit(self, request_id, prompt, max_new_tokens):
        request_id = str(request_id)

        if request_id in self.requests:
            raise ValueError(f"Duplicate request ID: {request_id}")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"][0].tolist()

        if len(input_ids) + max_new_tokens - 1 > self.max_seq_len:
            raise ValueError(
                f"Request {request_id} exceeds max_seq_len={self.max_seq_len}"
            )

        request = RequestState(
            request_id,
            prompt,
            input_ids,
            max_new_tokens,
        )

        request.sequence_id = request_id
        request.created_at = time.monotonic()
        request.status = "waiting"

        self.requests[request_id] = request
        self.waiting.append(request)

        return request

    def _admit(self):
        admitted = []

        while self.waiting and len(self.active) < self.max_batch_size:
            request = self.waiting.popleft()
            request.status = "active"
            self.active[request.request_id] = request

            self.cache.activate_request(request.request_id)
            admitted.append(request)

        self.max_active_seen = max(
            self.max_active_seen,
            len(self.active),
        )

        return admitted

    def _make_mask(self, valid_lengths, query_length, key_length):
        """
        Additive 4D causal mask.

        Shape: [batch, 1, query_length, key_length].
        Valid tokens can attend only to valid causal keys.
        Padding positions are excluded.
        """
        batch_size = len(valid_lengths)

        mask = torch.full(
            (batch_size, 1, query_length, key_length),
            torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=self.device,
        )

        for batch_idx, valid_length in enumerate(valid_lengths):
            for query_idx in range(query_length):
                if query_length == 1:
                    allowed = valid_length
                elif query_idx < valid_length:
                    allowed = min(query_idx + 1, valid_length)
                else:
                    # Padding query: its output is ignored.
                    allowed = valid_length

                if allowed > 0:
                    mask[batch_idx, 0, query_idx, :allowed] = 0

        return mask

    def _mark_finished(self, request, reason):
        if request.finished:
            return

        request.finished = True
        request.status = "finished"
        request.finish_reason = reason
        request.completed_at = time.monotonic()
        self._emit_event(request, "finished")
        print(
            f"[COMPLETE] {request.request_id} "
            f"generated={len(request.generated_tokens)} "
            f"reason={reason}"
        )

    def _emit_event(self, request, event):
        if self.event_callback is not None:
            self.event_callback({
                "event": event,
                "request_id": str(request.request_id),
                "generated_tokens": list(request.generated_tokens),
                "token_count": len(request.generated_tokens),
                "timestamp": time.monotonic(),
                "finish_reason": request.finish_reason,
            })

    def _record_token(self, request, token_id):
        now = time.monotonic()

        request.generated_tokens.append(token_id)
        request.token_timestamps.append(now)

        if request.first_token_at is None:
            request.first_token_at = now

        self._emit_event(request, "token")


    def _check_completion(self, request, token_id):
        if (
            self.eos_enabled
            and self.tokenizer.eos_token_id is not None
            and token_id == self.tokenizer.eos_token_id
        ):
            self._mark_finished(request, "eos")
        elif len(request.generated_tokens) >= request.max_new_tokens:
            self._mark_finished(request, "length")

    @torch.inference_mode()
    def _run_prefill_batch(self, requests):
        if not requests:
            return

        request_ids = [str(r.request_id) for r in requests]
        prompt_lengths = [len(r.input_tokens) for r in requests]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        max_prompt_length = max(prompt_lengths)
        batch_size = len(requests)

        input_ids = torch.full(
            (batch_size, max_prompt_length),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )

        position_ids = torch.zeros_like(input_ids)

        for idx, request in enumerate(requests):
            length = prompt_lengths[idx]
            tokens = torch.tensor(
                request.input_tokens,
                dtype=torch.long,
                device=self.device,
            )
            input_ids[idx, :length] = tokens
            position_ids[idx, :length] = torch.arange(
                length,
                dtype=torch.long,
                device=self.device,
            )

        self.cache.set_batch(request_ids, prompt_lengths)

        attention_mask = self._make_mask(
            prompt_lengths,
            query_length=max_prompt_length,
            key_length=max_prompt_length,
        )

        print(
            f"[PREFILL BATCH] requests={request_ids} "
            f"prompt_lengths={prompt_lengths}"
        )

        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=0,
        )

        # Select each prompt's final real token, not its padding position.
        batch_logits = torch.stack([
            output.logits[idx, prompt_lengths[idx] - 1, :]
            for idx in range(batch_size)
        ])

        next_tokens = batch_logits.argmax(dim=-1)

        for idx, request in enumerate(requests):
            token_id = int(next_tokens[idx].item())
            self._record_token(request, token_id)
            request.prefill_complete = True
            request.prefill_time = time.monotonic()
            request.decode_steps = 0

            print(
                f"[PREFILL] {request.request_id}: "
                f"generated=1/{request.max_new_tokens}"
            )

            self._check_completion(request, token_id)

    @torch.inference_mode()
    def _run_decode_batch(self, requests):
        if not requests:
            return

        request_ids = [str(r.request_id) for r in requests]

        # Current cache lengths are the positions of the next input tokens.
        current_lengths = [
            self.cache.paged.get_seq_length(request_id, 0)
            for request_id in request_ids
        ]

        input_ids = torch.tensor(
            [[r.generated_tokens[-1]] for r in requests],
            dtype=torch.long,
            device=self.device,
        )

        position_ids = torch.tensor(
            current_lengths,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        new_lengths = [length + 1 for length in current_lengths]

        self.cache.set_batch(
            request_ids,
            [1] * len(requests),
        )

        attention_mask = self._make_mask(
            new_lengths,
            query_length=1,
            key_length=max(new_lengths),
        )

        print(
            f"[DECODE BATCH] requests={request_ids} "
            f"cache_lengths={current_lengths}"
        )

        output = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

        next_tokens = output.logits[:, -1, :].argmax(dim=-1)

        for idx, request in enumerate(requests):
            token_id = int(next_tokens[idx].item())
            self._record_token(request, token_id)
            request.decode_steps += 1

            print(
                f"[DECODE] {request.request_id}: "
                f"generated={len(request.generated_tokens)}"
            )

            self._check_completion(request, token_id)

    def _finish_completed(self):
        finished_ids = [
            request_id
            for request_id, request in self.active.items()
            if request.finished
        ]

        for request_id in finished_ids:
            request = self.active.pop(request_id)
            self.cache.release_request(request_id)

            request.status = "completed"
            self.completed.append(request)

        return finished_ids

    def step(self):
        admitted = self._admit()

        active_requests = list(self.active.values())

        # Snapshot decode-ready requests before running new prefills.
        to_decode = [
            r for r in active_requests
            if r.prefill_complete and not r.finished
        ]

        to_prefill = [
            r for r in active_requests
            if not r.prefill_complete and not r.finished
        ]

        # Both operations are vectorized batches, but use separate forwards.
        self._run_decode_batch(to_decode)
        self._run_prefill_batch(to_prefill)

        finished = self._finish_completed()

        return {
            "admitted": [r.request_id for r in admitted],
            "prefilled": [r.request_id for r in to_prefill],
            "decoded": [r.request_id for r in to_decode],
            "finished": finished,
            "active": list(self.active.keys()),
            "waiting": len(self.waiting),
        }

    def run_until_complete(self, max_iterations=1000):
        iteration = 0

        while self.waiting or self.active:
            if iteration >= max_iterations:
                raise TimeoutError(
                    f"Engine exceeded {max_iterations} iterations. "
                    f"Metrics: {self.get_metrics()}"
                )

            self.step()
            iteration += 1

        print(
            f"[ENGINE] Completed all requests in {iteration} iterations"
        )

        return self.completed


    def get_metrics(self):
        completed_metrics = []

        for request in self.completed:
            completed_metrics.append({
                "request_id": str(request.request_id),
                "prompt_tokens": request.prompt_length,
                "generated_tokens": len(request.generated_tokens),
                "ttft_ms": (
                    request.ttft_seconds * 1000
                    if request.ttft_seconds is not None
                    else None
                ),
                "latency_ms": (
                    request.latency_seconds * 1000
                    if request.latency_seconds is not None
                    else None
                ),
                "average_itl_ms": (
                    request.average_itl_seconds * 1000
                    if request.average_itl_seconds is not None
                    else None
                ),
                "finish_reason": request.finish_reason,
            })

        return {
            "waiting": len(self.waiting),
            "active": len(self.active),
            "completed": len(self.completed),
            "max_active_batch_size": self.max_active_seen,
            "paged_cache_memory_mib": self.cache.memory_mib(),
            "requests": completed_metrics,
        }
