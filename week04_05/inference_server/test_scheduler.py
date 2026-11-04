
from request_state import RequestState
from scheduler import ContinuousBatchScheduler
import time

class MockPagedCache:
    def __init__(self):
        self.sequences = set()

    def create_request(self, request_id):
        if request_id in self.sequences:
            raise ValueError("Sequence already allocated")
        self.sequences.add(request_id)

    def release_request(self, request_id):
        self.sequences.discard(request_id)

    def has_request(self, request_id):
        return request_id in self.sequences


def make_request(request_id, prompt_length=16, max_tokens=10):
    return RequestState(
        request_id=request_id,
        prompt=f"Test prompt for request {request_id}",
        input_tokens=list(range(prompt_length)),
        max_new_tokens=max_tokens,
    )


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main():
    cache = MockPagedCache()

    scheduler = ContinuousBatchScheduler(
        max_batch_size=3,
        cache_manager=cache,
    )

    # Submit A, B and C.
    for request_id in ["A", "B", "C"]:
        scheduler.submit(make_request(request_id))

    check(
        len(scheduler.waiting_queue) == 3,
        "Request submission",
    )

    # Admit all three.
    admitted = scheduler.admit_requests()

    check(
        len(admitted) == 3,
        "Initial batch admission",
    )

    check(
        len(scheduler.active_requests) == 3,
        "Three active requests",
    )

    # Complete B.
    request_b = scheduler.active_requests["B"]
    request_b.finished = True
    request_b.finish_reason = "length" 
    request_b.completed_at = time.monotonic()

    # Submit D while A and C remain active.
    scheduler.submit(make_request("D"))

    # Remove B and admit D.
    newly_admitted = scheduler.finish_requests()

    check(
        "B" not in scheduler.active_requests,
        "Finished request removed",
    )

    check(
        "D" in scheduler.active_requests,
        "New request admitted while others remain active",
    )

    check(
        "A" in scheduler.active_requests
        and "C" in scheduler.active_requests,
        "Existing requests remain active",
    )

    check(
        not cache.has_request("B"),
        "Finished request cache released",
    )

    # Finish all remaining requests.
    for request in list(scheduler.active_requests.values()):
        request.finished = True
        request.finish_reason = "length"
        request.completed_at = time.monotonic()

    scheduler.finish_requests()

    check(
        len(scheduler.active_requests) == 0,
        "All requests completed",
    )

    check(
        len(cache.sequences) == 0,
        "All cache sequences released",
    )

    print("\nAll scheduler tests passed!")
    print(scheduler.metrics())


if __name__ == "__main__":
    main()
