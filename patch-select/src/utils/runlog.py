from __future__ import annotations

import logging
import os
import sys
import time
import traceback
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
    def __init__(
        self,
        logger: logging.Logger,
        prefix: str,
        total: int,
        every: int = 50,
        every_secs: float = 300.0,
    ):
        self.logger = logger
        self.prefix = prefix
        self.total = max(0, int(total))
        self.every = max(1, int(every))
        self.every_secs = max(1.0, float(every_secs))
        self.start = time.time()
        self.last_emit = 0
        self.last_emit_ts = self.start

    def update(self, count: int, **stats) -> None:
        count = int(count)
        now = time.time()
        due_by_count = (count - self.last_emit) >= self.every
        due_by_time = (now - self.last_emit_ts) >= self.every_secs
        is_final = count >= self.total
        if not is_final and not due_by_count and not due_by_time:
            return
        self.last_emit = count
        self.last_emit_ts = now
        elapsed = max(0.0, now - self.start)
        rate = (count / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, self.total - count)
        eta_sec = (remaining / rate) if rate > 0 else 0.0
        stats_txt = " ".join(f"{k}={v}" for k, v in stats.items())
        self.logger.info(
            f"[{self.prefix}] processed={count}/{self.total} elapsed={_fmt_sec(elapsed)} eta={_fmt_sec(eta_sec)} {stats_txt}".rstrip()
        )


def log_debug_traceback(logger: logging.Logger, prefix: str = "") -> None:
    tb = traceback.format_exc().rstrip()
    if not tb:
        return
    if prefix:
        logger.debug(f"{prefix}\n{tb}")
    else:
        logger.debug(tb)


def _fmt_sec(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"
