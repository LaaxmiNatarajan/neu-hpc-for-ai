from __future__ import annotations

from starlette_compress._utils import (
    classify_start_message,
    mutable_response_headers,
)

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Protocol

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    class StreamEncoder(Protocol):
        def feed(self, data: bytes, flush: bool) -> Iterable[bytes]: ...
        def finish(self) -> Iterable[bytes]: ...

    OneshotCompress = Callable[[bytes], bytes]
    EncoderFactory = Callable[[int], StreamEncoder]


def _content_length_from_raw(headers: Iterable[tuple[bytes, bytes]]) -> int:
    """Read Content-Length without mutating the header list."""
    for name, value in headers:
        if (name if name.islower() else name.lower()) == b'content-length':
            try:
                return int(value)
            except ValueError:
                return -1
    return -1


class CompressionResponder:
    """Shared ASGI control loop for encoded compression responders."""

    __slots__ = (
        'app',
        'create_encoder',
        'encoding',
        'minimum_size',
        'oneshot',
    )

    def __init__(
        self,
        app: ASGIApp,
        minimum_size: int,
        encoding: str,
        oneshot: OneshotCompress,
        create_encoder: EncoderFactory,
    ) -> None:
        self.app = app
        self.minimum_size = minimum_size
        self.encoding = encoding
        self.oneshot = oneshot
        self.create_encoder = create_encoder

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False
        start_message: Message | None = None
        encoder: StreamEncoder | None = None
        streaming = False
        extensions: dict[str, object] = scope.get('extensions', {})
        pathsend_capable = 'http.response.pathsend' in extensions

        # Encoded path: never advertise zerocopysend — after Content-Encoding
        # is committed, a raw file would corrupt the response. Shallow-copy
        # only; do not mutate the caller's scope.
        if 'http.response.zerocopysend' in extensions:
            extensions = dict(extensions)
            del extensions['http.response.zerocopysend']
            scope = {**scope, 'extensions': extensions}

        async def wrapper(message: Message) -> None:
            nonlocal started, start_message, encoder, streaming

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

                # Streaming types without pathsend: commit at start so clients
                # (e.g. EventSource) see headers before the first body event.
                if streaming and not pathsend_capable:
                    # Streaming: never pledge Content-Length to the encoder
                    # (SSE body size may not match the header).
                    encoder = self.create_encoder(-1)
                    headers = mutable_response_headers(message)
                    headers['Content-Encoding'] = self.encoding
                    headers.add_vary_header('Accept-Encoding')
                    del headers['Content-Length']
                    started = True
                    await send(message)
                    return

                start_message = message
                started = True
                return

            if encoder is not None:
                if message_type != 'http.response.body':
                    # Trailers / extensions after start — do not re-send start
                    await send(message)
                    return

                body: bytes = message.get('body', b'')
                more_body: bool = message.get('more_body', False)

            elif start_message is not None:
                if message_type != 'http.response.body':
                    # pathsend / trailers / unknown: forward original start raw
                    pending = start_message
                    start_message = None
                    await send(pending)
                    await send(message)
                    return

                body = message.get('body', b'')
                more_body = message.get('more_body', False)

                if not streaming and not more_body and len(body) < self.minimum_size:
                    pending = start_message
                    start_message = None
                    await send(pending)
                    await send(message)
                    return

                if not more_body:
                    # Terminal commit: fallible oneshot before any header mutation
                    # so a raised codec error leaves the app's start message intact.
                    compressed_body = self.oneshot(body)
                    headers = mutable_response_headers(start_message)
                    headers['Content-Encoding'] = self.encoding
                    headers.add_vary_header('Accept-Encoding')
                    headers['Content-Length'] = str(len(compressed_body))
                    message['body'] = compressed_body
                    del body
                    pending = start_message
                    start_message = None
                    await send(pending)
                    await send(message)
                    return

                # Stream-begin: fallible encoder create before any header mutation.
                # Only buffered (non-streaming) paths may pledge Content-Length;
                # streaming always uses -1. CL is read via a non-mutating raw scan.
                if streaming:
                    content_length = -1
                else:
                    content_length = _content_length_from_raw(start_message['headers'])
                encoder = self.create_encoder(content_length)
                headers = mutable_response_headers(start_message)
                headers['Content-Encoding'] = self.encoding
                headers.add_vary_header('Accept-Encoding')
                del headers['Content-Length']
                pending = start_message
                start_message = None
                await send(pending)

            else:
                # Passthrough, finished, or a pre-start extension message.
                await send(message)
                return

            assert encoder is not None

            if more_body:
                if not body:
                    return
                for chunk in encoder.feed(body, streaming):
                    if chunk:
                        await send({
                            'type': 'http.response.body',
                            'body': chunk,
                            'more_body': True,
                        })
                return

            if body:
                for chunk in encoder.feed(body, False):
                    if chunk:
                        await send({
                            'type': 'http.response.body',
                            'body': chunk,
                            'more_body': True,
                        })

            final_chunk = b''
            for chunk in encoder.finish():
                if chunk:
                    if final_chunk:
                        await send({
                            'type': 'http.response.body',
                            'body': final_chunk,
                            'more_body': True,
                        })
                    final_chunk = chunk
            await send({'type': 'http.response.body', 'body': final_chunk})
            encoder = None

        await self.app(scope, receive, wrapper)
