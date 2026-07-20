import json
import logging
import os
import sys
import time

_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
EVENT_PREFIX = "EVENT:"


class _JsonlHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fp = open(path, 'a', buffering=1, encoding='utf-8')

    def emit(self, record):
        data = getattr(record, 'event', None)
        if data is None:
            return
        try:
            self._fp.write(json.dumps(data, default=str) + '\n')
        except Exception:
            self.handleError(record)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    if getattr(logger, '_The_White_Lotus_Academy_configured', False):
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Force UTF-8 so box-drawing characters don't blow up on Windows cp1252 consoles.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    log_dir = log_dir or _LOG_DIR
    ts = time.strftime('%Y%m%d-%H%M%S')
    path = os.path.join(log_dir, f"{name}-{ts}.jsonl")
    logger.addHandler(_JsonlHandler(path))
    logger._The_White_Lotus_Academy_configured = True
    logger._jsonl_path = path
    return logger


def log_event(logger, event, **fields):
    """Emit a structured event to JSONL and an EVENT:{...} line to stdout."""
    data = {"ts": time.time(), "event": event, **fields}
    line = EVENT_PREFIX + json.dumps(data, default=str)
    logger.info(line, extra={"event": data})


def parse_event_line(line):
    s = line.strip()
    if not s.startswith(EVENT_PREFIX):
        return None
    try:
        return json.loads(s[len(EVENT_PREFIX):])
    except Exception:
        return None
