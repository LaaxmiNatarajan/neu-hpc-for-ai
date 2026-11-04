from __future__ import annotations

from zstandard import ZstdCompressor

from starlette_compress._responder import CompressionResponder

TYPE_CHECKING = False
if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.types import ASGIApp
    from zstandard import ZstdCompressionChunker


class _ZstdLegacyStreamEncoder:
    """Streaming encoder backed by zstandard's chunker (bounded output).

    The size pledge applies only when a known Content-Length was supplied
    (buffered streams); streaming always passes ``-1``.

    Per-message flush uses ``chunker.flush()`` with no argument. Never pass
    module-level ``zstandard.FLUSH_BLOCK`` (it equals COMPRESSOBJ_FLUSH_FINISH
    and ends the frame).
    """

    __slots__ = ('_chunker',)

    def __init__(self, chunker: ZstdCompressionChunker) -> None:
        self._chunker = chunker

    def feed(self, data: bytes, flush: bool) -> Iterable[bytes]:
        yield from self._chunker.compress(data)
        if flush:
            yield from self._chunker.flush()

    def finish(self) -> Iterable[bytes]:
        return self._chunker.finish()


class ZstdResponder(CompressionResponder):
    __slots__ = ()

    def __init__(self, app: ASGIApp, minimum_size: int, level: int) -> None:
        def oneshot(body: bytes) -> bytes:
            return ZstdCompressor(level=level).compress(body)

        def create_encoder(content_length: int) -> _ZstdLegacyStreamEncoder:
            chunker = ZstdCompressor(level=level).chunker(content_length)
            return _ZstdLegacyStreamEncoder(chunker)

        super().__init__(app, minimum_size, 'zstd', oneshot, create_encoder)
