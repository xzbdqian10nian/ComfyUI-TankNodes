"""Cancellable async HTTP work behind ComfyUI's synchronous node interface."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import logging
import queue
import threading


def check_interrupted() -> None:
    # Keep this on the node's execution thread. ComfyUI consumes the interrupt
    # flag when raising its exception; a worker must not consume it separately.
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        return  # Standalone tests and scripts need not load ComfyUI.
    throw_exception_if_processing_interrupted()


class APIResponse:
    """Pull async response chunks while checking the native stop flag every 50 ms.

    The worker owns its event loop, HTTP client and response. Cancellation closes
    those resources; it does not leave a blocking HTTP request running in a
    detached thread. Progress and result handling stay on the node thread.
    """

    def __init__(self, source):
        self._source = source
        self._items = queue.Queue(maxsize=32)
        self._ready = threading.Event()
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self._loop = None
        self._task = None
        self._error = None
        self._thread = None

    async def _consume(self):
        if self._cancelled.is_set():
            return
        async with aclosing(self._source()) as source:
            async for item in source:
                while True:
                    try:
                        self._items.put_nowait(item)
                        break
                    except queue.Full:
                        await asyncio.sleep(0.01)

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._task = loop.create_task(self._consume())
        self._ready.set()
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._error = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                # Do not wait for a system DNS lookup in the default executor.
                # The cancelled HTTP coroutine cannot send a request afterward.
                loop.close()
                self._done.set()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            check_interrupted()
            if self._cancelled.is_set():
                raise StopIteration
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="TankNodes-API", daemon=True)
                self._thread.start()
            while True:
                check_interrupted()
                try:
                    item = self._items.get(timeout=0.05)
                except queue.Empty:
                    if self._done.is_set() and self._items.empty():
                        if self._error is not None:
                            raise self._error
                        raise StopIteration
                else:
                    check_interrupted()
                    return item
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._cancelled.is_set():
            return
        self._cancelled.set()
        if self._thread is None:
            return
        if self._ready.wait(timeout=1.0) and not self._done.is_set():
            try:
                self._loop.call_soon_threadsafe(self._task.cancel)
            except RuntimeError:
                pass  # The worker finished and closed its loop concurrently.
        self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            logging.warning("[TankNodes] API connection cleanup is still finishing.")
