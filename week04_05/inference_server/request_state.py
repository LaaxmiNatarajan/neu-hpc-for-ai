
import time


class RequestState:
    def __init__(
        self,
        request_id,
        prompt,
        input_tokens,
        max_new_tokens,
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.input_tokens = input_tokens

        self.max_new_tokens = max_new_tokens
        self.generated_tokens = []
        self.token_timestamps = []

        self.sequence_id = request_id

        self.finished = False
        self.status = "waiting"
        self.finish_reason = None

        self.prefill_complete = False
        self.prompt_length = len(input_tokens)

        self.created_at = time.monotonic()
        self.prefill_time = None
        self.first_token_at = None
        self.completed_at = None

        self.decode_steps = 0

    @property
    def ttft_seconds(self):
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.created_at

    @property
    def latency_seconds(self):
        if self.completed_at is None:
            return None
        return self.completed_at - self.created_at

    @property
    def average_itl_seconds(self):
        if len(self.token_timestamps) < 2:
            return None

        intervals = [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]

        return sum(intervals) / len(intervals)
