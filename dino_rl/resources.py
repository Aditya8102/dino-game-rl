"""Sample process-tree resource use without delaying the game."""

import time

import psutil


class ResourceMonitor:
    """Sample CPU and memory use; short-lived child processes may be missed."""

    def __init__(self) -> None:
        self.root = psutil.Process()
        self.started = self.sampled = time.monotonic()
        self.previous, self.rss = self._sample()
        self.cpu_seconds = 0.0
        self.peak_cores = 0.0

    def _sample(self):
        usage, rss = {}, 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                with process.oneshot():
                    times = process.cpu_times()
                    usage[(process.pid, process.create_time())] = times.user + times.system
                    rss += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # Browser subprocesses can exit between enumeration and sampling.
                continue
        return usage, rss

    def tick(self, *, force: bool = False) -> None:
        """Update the usage sample at most once a second unless forced."""
        now = time.monotonic()
        if not force and now - self.sampled < 1.0:
            return
        current, self.rss = self._sample()
        consumed = sum(
            max(0.0, value - self.previous.get(key, value)) for key, value in current.items()
        )
        self.cpu_seconds += consumed
        self.peak_cores = max(self.peak_cores, consumed / max(now - self.sampled, 1e-6))
        self.previous, self.sampled = current, now

    def summary(self) -> dict:
        """Return sampled estimates; short-lived children may be missed."""
        self.tick(force=True)
        elapsed = time.monotonic() - self.started
        mean_cores = self.cpu_seconds / max(elapsed, 1e-6)
        return {
            "elapsed_wall_seconds": round(elapsed, 2),
            "mean_cpu_cores": round(mean_cores, 3),
            "mean_machine_cpu_percent": round(100 * mean_cores / psutil.cpu_count(), 2),
            "peak_sample_cpu_cores": round(self.peak_cores, 3),
            "process_group_rss_mb": round(self.rss / 1024**2, 1),
        }
