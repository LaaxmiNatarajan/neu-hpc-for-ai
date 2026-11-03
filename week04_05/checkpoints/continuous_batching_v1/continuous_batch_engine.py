
import time
from collections import deque

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from paged_kv_cache import PagedKVCache
from batched_paged_cache import BatchedPagedCache
from request_state import RequestState


class RequestCacheLayer(CacheLayerMixin):
    """One request's view of a shared paged KV-cache manager."""

    is_sliding = False
    is_compileable = False
    is_croppable = False

    def __init__(self, paged_cache, request_id, layer_idx):
        super().__init__()
        self.paged_cache = paged_cache
        self.request_id = str(request_id)
        self.layer_idx = layer_idx
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        self.paged_cache.append(
            self.request_id,
            self.layer_idx,
            key_states,
            value_states,
        )
        return self.paged_cache.get_kv(
            self.request_id,
            self.layer_idx,
        )

    def get_seq_length(self):
        return self.paged_cache.get_seq_length(
            self.request_id,
            self.layer_idx,
        )

    def get_max_length(self):
        return self.paged_cache.manager.num_blocks * (
            self.paged_cache.manager.block_size
        )

    def get_mask_sizes(self, query_length):
        current_length = self.get_seq_length()
        return current_length + query_length, 0

    def reset(self):
        # Request lifecycle is managed by the engine.
        pass


class RequestTransformersCache(Cache):
    """Transformers Cache adapter for one request."""

    def __init__(self, paged_cache, request_id, num_layers, max_cache_len):
        self.request_id = str(request_id)
        self.max_cache_len = max_cache_len

        layers = [
            RequestCacheLayer(paged_cache, self.request_id, layer_idx)
            for layer_idx in range(num_layers)
        ]

        super().__init__(layers=layers)

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()

    def get_max_cache_shape(self, layer_idx=0):
        return self.max_cache_len

    def get_max_length(self, layer_idx=0):
        return self.max_cache_len


class SimpleContinuousScheduler:
    """
    Small continuous-admission scheduler.

    Requests are admitted as capacity becomes available.
    Finished requests are removed and their KV blocks released.
    """

    def __init__(self, max_batch_size, cache_manager):
        self.max_batch_size = max_batch_size
        self.cache_manager = cache_manager

        self.waiting = deque()
        self.active = {}
        self.completed = []

        self.max_active_seen = 0

    def submit(self, request):
        request.status = "waiting"
        self.waiting.append(request)

    def admit_requests(self):
        admitted = []

        while (
            self.waiting
            and len(self.active) < self.max_batch_size
        ):
            request = self.waiting.popleft()
            request.status = "active"
            self.active[request.request_id] = request
            admitted.append(request)

        self.max_active_seen = max(
            self.max_active_seen,
            len(self.active),
        )

        return admitted

    def get_active_batch(self):
        return list(self.active.values())

    def finish_requests(self):
        finished_ids = [
            request_id
            for request_id, request in self.active.items()
            if request.finished
        ]

        for request_id in finished_ids:
            request = self.active.pop(request_id)

            if self.cache_manager.has_request(request.sequence_id):
                self.cache_manager.release_request(
                    request.sequence_id
                )

            request.status = "completed"
            self.completed.append(request)

        # Immediately admit waiting requests into newly available slots.
        return self.admit_requests()

    def metrics(self):
        return {
            "waiting": len(self.waiting),
            "active": len(self.active),
            "completed": len(self.completed),
            "max_batch_size": self.max_active_seen,
        }


class ContinuousBatchEngine:
    def __init__(
        self,
        model,
        tokenizer,
        max_batch_size=4,
        max_seq_len=256,
        block_size=16,
        eos_enabled=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.eos_enabled = eos_enabled

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        num_layers = self.config.num_hidden_layers
        num_kv_heads = self.config.num_key_value_heads
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )

        blocks_per_request = (
            max_seq_len + block_size - 1
        ) // block_size

        total_blocks = max_batch_size * blocks_per_request

        self.manager = PagedKVCache(
            num_layers,
            total_blocks,
            num_kv_heads,
            head_dim,
            block_size,
            self.dtype,
            self.device,
        )

        self.paged_cache = BatchedPagedCache(self.manager)

        self.scheduler = SimpleContinuousScheduler(
            max_batch_size,
            self.paged_cache,
        )

        self.request_caches = {}
        self.requests = {}

    def submit(self, request_id, prompt, max_new_tokens):
        request_id = str(request_id)

        if request_id in self.requests:
            raise ValueError(
                f"Duplicate request ID: {request_id}"
            )

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)

        prompt_tokens = input_ids[0].tolist()

        if len(prompt_tokens) + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"Request {request_id} exceeds max_seq_len="
                f"{self.max_seq_len}"
            )

        request = RequestState(
            request_id,
            prompt,
            prompt_tokens,
            max_new_tokens,
        )

        request.sequence_id = request_id
        request.created_at = time.monotonic()

        self.requests[request_id] = request
        self.scheduler.submit(request)

        return request

    def _get_request_cache(self, request):
        request_id = str(request.request_id)

        if request_id not in self.request_caches:
            self.paged_cache.create_request(request_id)

            self.request_caches[request_id] = (
                RequestTransformersCache(
                    self.paged_cache,
                    request_id,
                    self.config.num_hidden_layers,
                    self.max_seq_len,
                )
            )

        return self.request_caches[request_id]

    def _mark_finished(self, request, reason):
        if request.finished:
            return

        request.finished = True
        request.status = "finished"
        request.finish_reason = reason
        request.completed_at = time.monotonic()

        print(
            f"[COMPLETE] {request.request_id} "
            f"reason={reason}"
        )

    @torch.inference_mode()
    def _prefill(self, request):
        request_id = str(request.request_id)

        cache = self._get_request_cache(request)

        input_ids = torch.tensor(
            [request.input_tokens],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.ones_like(input_ids)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        next_token = output.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = int(next_token.item())
        request.generated_tokens.append(token_id)

        request.prefill_complete = True
        request.prefill_time = time.monotonic()
        request.decode_steps = 0

        print(
            f"[PREFILL] {request_id}: "
            f"generated=1/{request.max_new_tokens}"
        )

        if (
            self.eos_enabled
            and self.tokenizer.eos_token_id is not None
            and token_id == self.tokenizer.eos_token_id
        ):
            self._mark_finished(request, "eos")
            return

        if len(request.generated_tokens) >= request.max_new_tokens:
            self._mark_finished(request, "length")

    @torch.inference_mode()
    def _decode(self, request):
        request_id = str(request.request_id)

        if not request.generated_tokens:
            raise RuntimeError(
                f"{request_id} has no generated token to decode"
            )

        cache = self._get_request_cache(request)

        last_token = request.generated_tokens[-1]

        input_ids = torch.tensor(
            [[last_token]],
            dtype=torch.long,
            device=self.device,
        )

        total_length = (
            len(request.input_tokens)
            + len(request.generated_tokens)
        )

        attention_mask = torch.ones(
            (1, total_length),
            dtype=torch.long,
            device=self.device,
        )

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

        next_token = output.logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        token_id = int(next_token.item())
        request.generated_tokens.append(token_id)
        request.decode_steps += 1

        print(
            f"[DECODE] {request_id}: "
            f"generated={len(request.generated_tokens)}"
        )

        if (
            self.eos_enabled
            and self.tokenizer.eos_token_id is not None
            and token_id == self.tokenizer.eos_token_id
        ):
            self._mark_finished(request, "eos")
            return

        if len(request.generated_tokens) >= request.max_new_tokens:
            self._mark_finished(request, "length")

    def step(self):
        # Admit requests waiting for capacity.
        newly_admitted = self.scheduler.admit_requests()

        # IMPORTANT: inspect ALL active requests.
        # A request admitted at the end of the previous step
        # must also be prefetched.
        active_requests = self.scheduler.get_active_batch()

        to_prefill = [
            request
            for request in active_requests
            if not request.prefill_complete
            and not request.finished
        ]

        # Capture decode-ready requests before prefill.
        to_decode = [
            request
            for request in active_requests
            if request.prefill_complete
            and not request.finished
        ]

        for request in to_prefill:
            self._prefill(request)

        for request in to_decode:
            if not request.finished:
                self._decode(request)

        # Release finished requests and admit waiting requests.
        admitted_after_completion = (
            self.scheduler.finish_requests()
        )

        return {
            "newly_admitted": [
                r.request_id for r in newly_admitted
            ],
            "prefilled": [
                r.request_id for r in to_prefill
            ],
            "decoded": [
                r.request_id for r in to_decode
            ],
            "admitted_after_completion": [
                r.request_id for r in admitted_after_completion
            ],
            "active": [
                r.request_id
                for r in self.scheduler.get_active_batch()
            ],
        }

    def run_until_complete(self, max_iterations=1000):
        iteration = 0

        while (
            self.scheduler.waiting
            or self.scheduler.active
        ):
            if iteration >= max_iterations:
                print("[TIMEOUT] Scheduler state:")
                print(self.scheduler.metrics())

                print(
                    "Active requests:",
                    [
                        {
                            "id": r.request_id,
                            "prefill_complete": r.prefill_complete,
                            "finished": r.finished,
                            "generated": len(r.generated_tokens),
                        }
                        for r in self.scheduler.get_active_batch()
                    ],
                )

                raise TimeoutError(
                    "Continuous batching exceeded max_iterations"
                )

            self.step()
            iteration += 1

        print(
            f"[ENGINE] Completed all requests in "
            f"{iteration} iterations"
        )

        return self.scheduler.completed

    def get_metrics(self):
        return self.scheduler.metrics()
