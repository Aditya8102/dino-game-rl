"""Parallel real-game collection and lossless one-step replay repartitioning."""

import logging
import signal
from collections.abc import Callable
from functools import partial
from multiprocessing import current_process
from multiprocessing.util import Finalize

import numpy as np
import psutil
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .env import DinoEnv

logger = logging.getLogger(__name__)
WORKER_CLOSE_TIMEOUT_SECONDS = 5


class ManagedSubprocVecEnv(SubprocVecEnv):
    """Close every owned worker, including after partial startup or a broken pipe."""

    def __init__(self, factories: list[Callable]) -> None:
        self.processes = []
        self.remotes = []
        self.closed = False
        try:
            super().__init__(factories, start_method="spawn")
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Worker cleanup failed after environment startup failed")
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError) as error:
                logger.debug("Worker pipe was already closed: %s", error)
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
            if process.is_alive():
                logger.warning("Stopping unresponsive environment worker %s", process.pid)
                try:
                    children = psutil.Process(process.pid).children(recursive=True)
                    for child in reversed(children):
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            logger.debug("Owned browser process %s already exited", child.pid)
                        except psutil.AccessDenied:
                            logger.warning(
                                "Could not terminate owned browser process %s", child.pid
                            )
                except psutil.NoSuchProcess:
                    logger.debug("Environment worker %s already exited", process.pid)
                except psutil.AccessDenied:
                    logger.warning("Could not inspect environment worker %s", process.pid)
                process.terminate()
                process.join(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=WORKER_CLOSE_TIMEOUT_SECONDS)
        for remote in self.remotes:
            remote.close()


def worker_env(max_seconds: float, show_browser: bool) -> Monitor:
    """Create one native game with child-process shutdown cleanup."""
    if current_process().name != "MainProcess":
        # The parent handles Ctrl+C and writes the final checkpoint.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    environment = Monitor(
        DinoEnv(max_seconds=max_seconds, show_browser=show_browser, realtime=True, cpu_cores=None)
    )
    if current_process().name != "MainProcess":
        Finalize(environment, environment.close, exitpriority=10)
    return environment


def training_env(workers: int, max_seconds: float, show_browser: bool = False) -> VecEnv:
    """Own one in-process game or a fixed set of spawned game workers."""
    factories = [partial(worker_env, max_seconds, show_browser) for _ in range(workers)]
    if workers == 1:
        return DummyVecEnv(factories)
    return ManagedSubprocVecEnv(factories)


def repartition_replay(old: ReplayBuffer, workers: int) -> tuple[ReplayBuffer, dict]:
    """Preserve real transitions, including terminal/timeout flags, across worker counts.

    A vector replay row needs one entry per worker. Repeat at most workers-1
    existing experiences to fill the final row; no synthetic transition is made.
    """
    if type(old) is not ReplayBuffer or old.optimize_memory_usage:
        raise ValueError("Only the project's standard one-step replay format can be migrated.")
    count = old.size() * old.n_envs
    if workers == old.n_envs:
        return old, {
            "from_workers": old.n_envs,
            "to_workers": workers,
            "retained_transitions": count,
            "repeated_padding_transitions": 0,
        }
    capacity = old.buffer_size * old.n_envs
    if capacity % workers:
        raise ValueError("Replay capacity must divide evenly by the new worker count.")
    new = ReplayBuffer(
        capacity,
        old.observation_space,
        old.action_space,
        device=old.device,
        n_envs=workers,
        handle_timeout_termination=old.handle_timeout_termination,
    )
    padding = (-count) % workers
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        source = getattr(old, name)[: old.size()]
        if old.full:
            source = np.concatenate((source[old.pos :], source[: old.pos]), axis=0)
        source = source.reshape((count, *source.shape[2:]))
        target = getattr(new, name).reshape((capacity, *source.shape[1:]))
        target[:count] = source
        if padding:
            target[count : count + padding] = source[np.arange(padding) % count]
    rows = (count + padding) // workers
    new.full = rows == new.buffer_size
    new.pos = 0 if new.full else rows
    return new, {
        "from_workers": old.n_envs,
        "to_workers": workers,
        "retained_transitions": count,
        "repeated_padding_transitions": padding,
    }
