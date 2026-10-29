"""Atomic JSON replacement with bounded retries for Windows reader locks.

Never remove the destination to force replacement. Persistent failures retain
the old checkpoint and a uniquely named new temporary file for diagnosis.
"""
import json
import os
from pathlib import Path
import time
import uuid


def atomic_json(path,value,attempts=40,delay_seconds=.025):
    if attempts<1 or delay_seconds<0:raise ValueError('Positive attempts and nonnegative retry delay required')
    path=Path(path)
    encoded=json.dumps(value,indent=2,allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    with temporary.open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==attempts-1:raise
            time.sleep(delay_seconds)
