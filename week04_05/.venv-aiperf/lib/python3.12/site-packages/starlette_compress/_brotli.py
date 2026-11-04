from __future__ import annotations

import brotli

from starlette_compress._responder import CompressionResponder

TYPE_CHECKING = False

if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.types import ASGIApp


class _BrotliStreamEncoder:
    __slots__ = ('_compressor',)

    def __init__(self, quality: int) -> None:
        self._compressor = brotli.Compressor(quality=quality)

    def feed(self, data: bytes, flush: bool) -> Iterable[bytes]:
        out = self._compressor.process(data)
        if flush:
            out += self._compressor.flush()
        return (out,) if out else ()

    def finish(self) -> Iterable[bytes]:
        out = self._compressor.finish()
        return (out,) if out else ()


class BrotliResponder(CompressionResponder):
    __slots__ = ()

    def __init__(self, app: ASGIApp, minimum_size: int, quality: int) -> None:
        def oneshot(body: bytes) -> bytes:
            return brotli.compress(body, quality=quality)

        def create_encoder(content_length: int) -> _BrotliStreamEncoder:
            return _BrotliStreamEncoder(quality)

        super().__init__(app, minimum_size, 'br', oneshot, create_encoder)
