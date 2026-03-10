from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


def _interactive_stream() -> bool:
    return bool(sys.stdout.isatty() and os.environ.get("TERM"))


def stage_logger(stage: str, root_dir: Path, verbose: bool = False) -> tuple[logging.Logger, bool]:
    log_dir = Path(root_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"patch_select.{stage}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger, _interactive_stream()


def progress(
    iterable: Iterable,
    *,
    interactive: bool,
    **kwargs,
):
    return tqdm(
        iterable,
        disable=not interactive,
        dynamic_ncols=interactive,
        **kwargs,
    )


class PeriodicProgress:
    def __init__(self, logger: logging.Logger, prefix: str, total: int, every: int = 50):
        self.logger = logger
        self.prefix = prefix
        self.total = max(0, int(total))
        self.every = max(1, int(every))
        self.start = time.time()
        self.last_emit = 0

    def update(self, count: int, **stats) -> None:
        count = int(count)
        if count < self.total and (count - self.last_emit) < self.every:
            return
        self.last_emit = count
        elapsed = max(0.0, time.time() - self.start)
        rate = (count / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.total - count)
        eta_sec = (remaining / rate) if rate > 0 else 0.0
        stats_txt = " ".join(f"{k}={v}" for k, v in stats.items())
        self.logger.info(
            f"[{self.prefix}] processed={count}/{self.total} elapsed={_fmt_sec(elapsed)} eta={_fmt_sec(eta_sec)} {stats_txt}".rstrip()
        )


def _fmt_sec(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"
