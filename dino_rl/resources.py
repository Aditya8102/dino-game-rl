"""Process-tree resource measurements and optional controlled-clock pacing."""

import math
import time

import psutil


class CpuBudget:
    """Sample process-tree CPU; optionally pace a controlled-clock caller."""

    def __init__(self, cores: float | None = None) -> None:
        if cores is not None and (not math.isfinite(cores) or cores <= 0):
            raise ValueError("CPU budget must be a positive number of cores.")
        self.cores = cores
        self.root = psutil.Process()
        self.started = self.sampled = time.monotonic()
        self.previous, self.rss = self._sample()
        self.cpu_seconds = 0.0
        self.sleep_seconds = 0.0
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

    def tick(self, *, force: bool = False, throttle: bool = True) -> None:
        """Update sampled usage and optionally pace an explicit controlled clock."""
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
        if throttle and self.cores is not None:
            delay = min(2.0, max(0.0, self.cpu_seconds / self.cores - (now - self.started)))
            if delay:
                time.sleep(delay)
                self.sleep_seconds += delay

    def summary(self) -> dict:
        """Return sampled estimates; short-lived children may be missed."""
        self.tick(force=True, throttle=False)
        elapsed = time.monotonic() - self.started
        mean_cores = self.cpu_seconds / max(elapsed, 1e-6)
        return {
            "cpu_budget_cores": self.cores,
            "elapsed_wall_seconds": round(elapsed, 2),
            "mean_cpu_cores": round(mean_cores, 3),
            "mean_machine_cpu_percent": round(100 * mean_cores / psutil.cpu_count(), 2),
            "peak_sample_cpu_cores": round(self.peak_cores, 3),
            "pacing_sleep_seconds": round(self.sleep_seconds, 2),
            "process_group_rss_mb": round(self.rss / 1024**2, 1),
        }
