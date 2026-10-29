"""Fail-closed exact-script GPU guard and user-pause checks for new studies."""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from pc_tests.model_pipeline.paths import PIPELINE


def check(script):
    pause=PIPELINE/'reports/experiment_pause.json'
    if pause.exists() and json.loads(pause.read_text())['status']=='paused_by_user':
        raise RuntimeError('Experiments paused by user')
    path=PIPELINE/'artifacts/pc_routers/thermal_guard.json'
    for attempt in range(20):
        try:
            state=json.loads(path.read_text(encoding='utf-8-sig'));break
        except (OSError,json.JSONDecodeError):
            if attempt==19: raise RuntimeError('Thermal telemetry unreadable')
            time.sleep(.025)
    if state['status']!='monitoring' or state.get('boundary_start_count') is not None:
        raise RuntimeError('Thermal latch; never restart automatically')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(state['utc'].replace('Z','+00:00'))).total_seconds()
    if not 0<=age<=130 or state['expected_script']!=Path(script).name or os.getpid() not in state['tracked_process_ids']:
        raise RuntimeError('Fresh exact worker thermal guard required')
    return state


def wait(script, seconds=120):
    deadline=time.monotonic()+seconds
    while True:
        try: return check(script)
        except RuntimeError:
            # Only an unlatched old guard may be replaced during startup.
            state=json.loads((PIPELINE/'artifacts/pc_routers/thermal_guard.json').read_text(encoding='utf-8-sig'))
            if state['status']!='monitoring' or state.get('boundary_start_count') is not None or time.monotonic()>deadline:
                raise
            pause=json.loads((PIPELINE/'reports/experiment_pause.json').read_text())
            if pause['status']=='paused_by_user': raise
            time.sleep(1)
