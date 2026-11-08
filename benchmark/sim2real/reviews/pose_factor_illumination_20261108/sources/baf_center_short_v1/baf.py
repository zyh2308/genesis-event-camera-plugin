"""Fixed 2-ms, 8-neighbour causal BAF; continuous float64-microsecond extension.

Official DV uses integral microseconds. We retain the original fractional-second
timestamps and only multiply by 1e6 for decisions. No polarity discrimination,
no own-pixel support, zero time surface, and every input updates the surface.
"""
from __future__ import annotations
import numpy as np


class BackgroundActivityFilter:
    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 1:
            raise ValueError('Positive integral sensor dimensions required')
        self.width, self.height = width, height
        self.surface = np.zeros(width * height, dtype=np.float64)
        self.last_us = -np.inf

    def process(self, timestamp_s, x, y, polarity=None):
        """Process one chronological packet, retaining state for the next packet.

        Search previous INPUT indices, not accepted indices. Vectorized grouped
        predecessor lookup is exactly the scalar update-after-decision rule.
        """
        t = np.asarray(timestamp_s)
        x, y = np.asarray(x), np.asarray(y)
        if t.ndim != 1 or x.shape != t.shape or y.shape != t.shape:
            raise ValueError('Aligned one-dimensional arrays required')
        if polarity is not None and np.asarray(polarity).shape != t.shape:
            raise ValueError('Polarity length differs')
        if not (np.issubdtype(x.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)):
            raise ValueError('Integral coordinates required')
        us = np.asarray(t, dtype=np.float64) * 1e6
        if not np.isfinite(us).all() or (us < 0).any() or (np.diff(us) < 0).any():
            raise ValueError('Finite, nonnegative chronological timestamps required')
        if len(us) and us[0] < self.last_us:
            raise ValueError('Packet moved backwards in time')
        if ((x < 0) | (x >= self.width) | (y < 0) | (y >= self.height)).any():
            raise ValueError('Out-of-sensor coordinate')
        n = len(us)
        if not n:
            return np.zeros(0, dtype=bool)
        stride = n + 1
        if self.width * self.height > np.iinfo(np.int64).max // stride:
            raise ValueError('Input too large for exact predecessor indices')
        pixel = y.astype(np.int64) * self.width + x
        index = np.arange(n, dtype=np.int64)
        codes = pixel * stride + index
        order = np.argsort(codes)
        ordered = codes[order]
        latest = np.full(n, -np.inf, dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                valid = (x + dx >= 0) & (x + dx < self.width) & (y + dy >= 0) & (y + dy < self.height)
                loc = np.flatnonzero(valid)
                neighbour = (y[loc] + dy) * self.width + x[loc] + dx
                previous = np.searchsorted(ordered, neighbour * stride + index[loc], side='left') - 1
                safe = np.maximum(previous, 0)
                found = (previous >= 0) & (ordered[safe] // stride == neighbour)
                values = self.surface[neighbour].copy()
                values[found] = us[order[safe[found]]]
                latest[loc] = np.maximum(latest[loc], values)
        keep = us - latest < 2000.0
        # Surface records all inputs, including rejected events, after decisions.
        np.maximum.at(self.surface, pixel, us)
        self.last_us = float(us[-1])
        return keep


class EpisodeEncoder:
    """Side-channel filtered encodings; the adapter still receives raw images.

    Strong references prevent identity reuse. One fresh instance per adapter
    call bounds memory; a filter is reset only when a new episode is encountered.
    """
    def __init__(self, encoder):
        self.encoder = encoder
        self.episodes = {}
        self.windows = []

    def __call__(self, t, x, y, p, start, end, width, height, count_scale):
        key = tuple(id(a) for a in (t, x, y, p))
        if key not in self.episodes:
            keep = BackgroundActivityFilter(width, height).process(t, x, y, p)
            self.episodes[key] = dict(refs=(t, x, y, p), keep=keep,
                                      filtered=tuple(a[keep] for a in (t, x, y, p)), size=(width, height))
        episode = self.episodes[key]
        assert all(a is b for a, b in zip(episode['refs'], (t, x, y, p)))
        assert episode['size'] == (width, height)
        raw = self.encoder(t, x, y, p, start, end, width, height, count_scale)
        filtered = self.encoder(*episode['filtered'], start, end, width, height, count_scale)
        inside = (t > start) & (t <= end)
        self.windows.append(dict(raw_image=raw, filtered_image=filtered, start_s=float(start), end_s=float(end),
                                 raw_events=int(inside.sum()), retained_events=int((inside & episode['keep']).sum())))
        return raw
