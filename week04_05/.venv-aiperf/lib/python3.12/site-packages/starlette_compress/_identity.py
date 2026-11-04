from __future__ import annotations

from starlette_compress._utils import (
    classify_start_message,
    mutable_response_headers,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send


class IdentityResponder:
    __slots__ = (
        'app',
        'minimum_size',
    )

    def __init__(self, app: ASGIApp, minimum_size: int) -> None:
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False
        start_message: Message | None = None
        streaming = False
        pathsend_capable = 'http.response.pathsend' in scope.get('extensions', ())

        async def wrapper(message: Message) -> None:
            nonlocal started, start_message, streaming

            message_type: str = message['type']

            if message_type == 'http.response.start':
                if started:
                    raise RuntimeError('http.response.start has already been received')

                verdict = classify_start_message(message)
                if verdict == 'skip':
                    started = True
                    await send(message)
                    return

                streaming = verdict == 'streaming'

                # Streaming types without pathsend: send start immediately so
                # clients see headers before the first body; pass messages through.
                if streaming and not pathsend_capable:
                    headers = mutable_response_headers(message)
                    headers.add_vary_header('Accept-Encoding')
                    # Content-Length is preserved for identity
                    started = True
                    await send(message)
                    return

                start_message = message
                started = True
                return

            if start_message is not None:
                if message_type != 'http.response.body':
                    pending = start_message
                    start_message = None
                    await send(pending)
                    await send(message)
                    return

                body: bytes = message.get('body', b'')
                more_body: bool = message.get('more_body', False)

                if not streaming and not more_body and len(body) < self.minimum_size:
                    pending = start_message
                    start_message = None
                    await send(pending)
                    await send(message)
                    return

                headers = mutable_response_headers(start_message)
                headers.add_vary_header('Accept-Encoding')
                pending = start_message
                start_message = None
                await send(pending)
                await send(message)
                return

            # Committed identity: pure passthrough, including empty bodies.
            await send(message)

        await self.app(scope, receive, wrapper)
