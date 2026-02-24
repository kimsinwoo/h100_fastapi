"""
WorkerQueue: bounded asyncio queue, max N parallel GPU jobs, timeout, OOM retry.
One controller; blocking inference in ThreadPoolExecutor; FIFO wait.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from app.sdxl_prod.config import get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            s = get_settings()
            max_workers = max(1, s.max_parallel_gpu_jobs)
            _executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sdxl_gpu")
        return _executor


def _run_with_oom_retry(fn: Callable[[], T]) -> T:
    import torch
    try:
        return fn()
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "OOM" in str(e).upper():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                return fn()
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower() or "OOM" in str(e2).upper():
                    raise RuntimeError("CUDA OOM after retry") from e2
                raise
        raise


QueueItemT = tuple[asyncio.Future[T], Callable[[], T], asyncio.Event]


class WorkerQueue:
    """Bounded queue; up to max_parallel_gpu_jobs run at once; FIFO; timeout and cancellation safe."""

    def __init__(
        self,
        max_queue_size: int | None = None,
        max_parallel: int | None = None,
        request_timeout: float | None = None,
        inference_timeout: float | None = None,
    ) -> None:
        s = get_settings()
        self._max_queue_size = max_queue_size if max_queue_size is not None else s.max_queue_size
        self._max_parallel = max_parallel if max_parallel is not None else s.max_parallel_gpu_jobs
        self._request_timeout = request_timeout if request_timeout is not None else s.request_timeout_seconds
        self._inference_timeout = inference_timeout if inference_timeout is not None else s.inference_timeout_seconds
        self._queue: asyncio.Queue[QueueItemT | None] = asyncio.Queue(maxsize=self._max_queue_size)
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._controller_task: asyncio.Task[None] | None = None
        self._shutdown = asyncio.Event()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop

    async def submit(self, fn: Callable[[], T]) -> T:
        """Submit a blocking inference call; wait for result with timeout. Raises TimeoutError on timeout."""
        loop = self._ensure_loop()
        if self._queue.qsize() >= self._max_queue_size:
            raise RuntimeError("Queue full, try again later")
        future: asyncio.Future[T] = loop.create_future()
        done_ev = asyncio.Event()
        await self._queue.put((future, fn, done_ev))
        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        except asyncio.CancelledError:
            if not future.done():
                future.cancel()
            raise
        finally:
            done_ev.set()

    async def _run_worker(self) -> None:
        loop = self._ensure_loop()
        executor = _get_executor()
        while not self._shutdown.is_set():
            try:
                get_fut = asyncio.ensure_future(self._queue.get())
                done, _ = await asyncio.wait(
                    [get_fut, asyncio.ensure_future(self._shutdown.wait())],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if self._shutdown.is_set():
                    break
                if get_fut not in done:
                    get_fut.cancel()
                    continue
                try:
                    item = get_fut.result()
                except asyncio.CancelledError:
                    continue
                if item is None:
                    break
                future, fn, done_ev = item
                async with self._semaphore:
                    if future.done():
                        continue
                    def _run() -> T:
                        return _run_with_oom_retry(fn)
                    try:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(executor, _run),
                            timeout=self._inference_timeout,
                        )
                        if not future.done():
                            future.set_result(result)
                    except asyncio.TimeoutError:
                        if not future.done():
                            future.set_exception(TimeoutError("Inference timeout"))
                    except Exception as e:
                        if not future.done():
                            future.set_exception(e)
                    finally:
                        done_ev.set()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Worker loop error: %s", e)

    def start(self) -> None:
        if self._controller_task is not None and not self._controller_task.done():
            return
        self._shutdown.clear()
        self._controller_task = asyncio.ensure_future(self._run_worker())
        logger.info(
            "WorkerQueue started: max_queue=%s max_parallel=%s",
            self._max_queue_size,
            self._max_parallel,
        )

    async def shutdown(self) -> None:
        self._shutdown.set()
        await self._queue.put(None)
        if self._controller_task is not None:
            try:
                await asyncio.wait_for(self._controller_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._controller_task.cancel()
            self._controller_task = None
