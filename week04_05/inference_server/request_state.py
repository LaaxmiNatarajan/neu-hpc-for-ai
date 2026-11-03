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

        self.sequence_id = request_id

        self.finished = False
        self.status = "waiting"

        self.prefill_complete = False
        self.prompt_length = len(input_tokens)

        self.created_at = None
        self.prefill_time = None
        self.decode_steps = 0
