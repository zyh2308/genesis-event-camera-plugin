"""Nested host-wall timers for isolated cost measurements, not a simulator patch."""
from __future__ import annotations
from contextlib import contextmanager
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


class WallLedger:
    """Exclusive times are additive; inclusive times are diagnostic only.

    This intentionally measures host elapsed time, including waits. It does not
    infer GPU cumulative kernel time. The optional sync callback is reserved for
    a separately labelled detailed-profile pass, not baseline throughput.
    """
    def __init__(self, clock=time.perf_counter_ns, cpu_clock=time.process_time_ns, sync=None):
        self.clock, self.cpu_clock, self.sync = clock, cpu_clock, sync
        self.stack = []
        self.records = []
        self.active = False
        self.hot_start_ns = self.hot_stop_ns = None
        self.hot_cpu_start_ns = self.hot_cpu_stop_ns = None

    def start_hot(self):
        if self.hot_start_ns is None:
            self.hot_start_ns = self.clock()
            self.hot_cpu_start_ns = self.cpu_clock()
            self.active = True

    def stop_hot(self):
        if self.hot_start_ns is None:
            raise RuntimeError('No measured interval started')
        if self.stack:
            raise RuntimeError('Stop the measured interval after all spans have closed')
        self.hot_stop_ns, self.hot_cpu_stop_ns = self.clock(), self.cpu_clock()
        self.active = False

    @contextmanager
    def span(self, name, simulated_time_s=None):
        if self.sync is not None:
            self.sync()
        frame = dict(name=name, begin=self.clock(), cpu_begin=self.cpu_clock(), children=0,
                     hot=self.active, simulated_time_s=simulated_time_s)
        self.stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            if self.sync is not None:
                self.sync()
            elapsed = self.clock()-frame['begin']
            cpu = self.cpu_clock()-frame['cpu_begin']
            assert self.stack.pop() is frame
            if self.stack:
                self.stack[-1]['children'] += elapsed
            self.records.append(dict(name=name, hot=frame['hot'], wall_ns=elapsed,
                exclusive_wall_ns=max(0, elapsed-frame['children']), process_cpu_ns=cpu,
                simulated_time_s=simulated_time_s, failed=failed))

    def inside(self, name):
        return any(frame['name'] == name for frame in self.stack)

    def aggregate(self, hot_only=True):
        result = defaultdict(lambda: dict(calls=0, inclusive_wall_s=0., exclusive_wall_s=0.))
        for row in self.records:
            if hot_only and not row['hot']:
                continue
            a = result[row['name']]
            a['calls'] += 1
            a['inclusive_wall_s'] += row['wall_ns']/1e9
            a['exclusive_wall_s'] += row['exclusive_wall_ns']/1e9
        return dict(result)

    def summary(self):
        hot_wall = ((self.hot_stop_ns-self.hot_start_ns)/1e9
                    if self.hot_stop_ns is not None else None)
        hot_cpu = ((self.hot_cpu_stop_ns-self.hot_cpu_start_ns)/1e9
                   if self.hot_cpu_stop_ns is not None else None)
        stages = self.aggregate()
        attributed = sum(row['exclusive_wall_s'] for row in stages.values())
        return dict(hot_wall_s=hot_wall, parent_hot_process_cpu_s=hot_cpu, hot_stages=stages,
            measured_parent_exclusive_sum_s=attributed,
            unattributed_parent_hot_wall_s=max(0., hot_wall-attributed) if hot_wall is not None else None,
            gpu_cumulative_time_s=None,
            gpu_time_status='unknown: no device-wide kernel/activity profiler collected',
            timing_kind='host wall elapsed; parent RPC wait already includes worker compute, never add it again',
            per_stage_cuda_barriers=self.sync is not None)

    def write(self, path):
        Path(path).write_text(json.dumps(dict(summary=self.summary(), spans=self.records), indent=2), encoding='utf-8')
