
from collections import deque
from request_state import RequestState


class ContinuousBatchScheduler:
    def __init__(
        self,
        max_batch_size,
        cache_manager,
        max_waiting_requests=100,
    ):
        self.max_batch_size = max_batch_size
        self.max_waiting_requests = max_waiting_requests

        self.cache_manager = cache_manager

        self.waiting_queue = deque()
        self.active_requests = {}

        self.completed_requests = []

    def submit(self, request):
        if not isinstance(request, RequestState):
            raise TypeError("Expected RequestState")

        if request.request_id in self.active_requests:
            raise ValueError("Duplicate request ID")

        if any(
            r.request_id == request.request_id
            for r in self.waiting_queue
        ):
            raise ValueError("Duplicate request ID")

        if len(self.waiting_queue) >= self.max_waiting_requests:
            raise RuntimeError("Waiting queue is full")

        self.waiting_queue.append(request)
        print(f"[SUBMIT] {request.request_id}")

    def admit_requests(self):
        admitted = []

        while (
            self.waiting_queue
            and len(self.active_requests) < self.max_batch_size
        ):
            request = self.waiting_queue.popleft()

            try:
                self.cache_manager.create_request(
                    request.sequence_id
                )

                request.status = "active"
                request.admitted_at = __import__("time").monotonic()

                self.active_requests[request.request_id] = request
                admitted.append(request)

                print(f"[ADMIT] {request.request_id}")

            except Exception:
                request.status = "waiting"
                self.waiting_queue.appendleft(request)
                raise

        return admitted

    def get_active_batch(self):
        return list(self.active_requests.values())

    def complete_request(self, request_id):
        request = self.active_requests.pop(request_id, None)

        if request is None:
            return

        try:
            self.cache_manager.release_request(
                request.sequence_id
            )
        finally:
            if not request.finished:
                request.finish("completed")

            self.completed_requests.append(request)

        print(f"[COMPLETE] {request_id}")

    def finish_requests(self):
        finished_ids = [
            request_id
            for request_id, request in self.active_requests.items()
            if request.finished
        ]

        for request_id in finished_ids:
            self.complete_request(request_id)

        # Admit new requests immediately after completions.
        return self.admit_requests()

    def metrics(self):
        return {
            "waiting": len(self.waiting_queue),
            "active": len(self.active_requests),
            "completed": len(self.completed_requests),
            "max_batch_size": self.max_batch_size,
        }
