"""
事件数据读取器：支持 Genesis HDF5、Prophesee .raw、numpy
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Tuple

from .event_schema import canonicalize_events


def read_genesis_hdf5(
    path: str,
    max_events: Optional[int] = None,
) -> np.ndarray:
    """
    Read events from Genesis plugin HDF5 output.
    
    Expected HDF5 structure:
      /events  — (N, 4) legacy uint32 [t_us, x, y, p_01]
    
    Returns:
        (N, 4) float64 [x, y, p, t_s] (canonical evaluation convention)
    """
    with h5py.File(path, 'r') as f:
        if '/aer/events' in f:
            dataset = f['/aer/events']
            events = dataset[:]
        elif '/events' in f:
            dataset = f['/events']
            events = dataset[:]
        else:
            # Try to find any dataset
            def find_events(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.shape[-1] == 4:
                    return name
            found = None
            f.visititems(lambda n, o: ...)  # not great
            # Just list available datasets
            keys = []
            f.visit(lambda k: keys.append(k) if isinstance(f[k], h5py.Dataset) else None)
            raise KeyError(f"No /aer/events found. Available: {keys}")
    
        layout = dataset.attrs.get('column_order', 't_us,x,y,p_01')
        if isinstance(layout, bytes):
            layout = layout.decode('utf-8')
        timestamp_unit = dataset.attrs.get('timestamp_unit', 'us')
        if isinstance(timestamp_unit, bytes):
            timestamp_unit = timestamp_unit.decode('utf-8')

    # Current and legacy recorder files both store time first. Metadata makes
    # units/polarity explicit; absent metadata uses the documented legacy form.
    if not str(layout).startswith('t'):
        raise ValueError(f"Unsupported Genesis HDF5 column order: {layout!r}")
    events = canonicalize_events(
        events, layout='txyp', timestamp_unit=str(timestamp_unit), sort=True
    )
    
    if max_events and len(events) > max_events:
        idx = np.linspace(0, len(events)-1, max_events, dtype=int)
        events = events[idx]
    
    return events


def read_genesis_event_frames(
    path: str,
    rgb_path: Optional[str] = None,
    episode: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read event/RGB frames from current and legacy Genesis layouts.

    Older experiments used top-level ``/event_frames`` and ``/rgb_frames``
    datasets.  The current recorder writes event frames and RGB frames to
    separate files, with episode groups such as ``/ep_0/frames`` and
    ``/ep_0/clean``.  ``rgb_path`` may be supplied when the files are
    separate; ``episode`` selects a specific episode group.
    """

    def episode_names(handle: h5py.File) -> list:
        if episode is not None:
            return [
                f"ep_{episode}",
                f"ep_{episode:06d}",
                f"episodes/ep_{episode:06d}",
            ]
        names = []
        for name in handle.keys():
            if isinstance(handle[name], h5py.Group) and name.startswith("ep_"):
                names.append(name)
        if "episodes" in handle and isinstance(handle["episodes"], h5py.Group):
            names.extend(f"episodes/{name}" for name in handle["episodes"].keys())
        return sorted(names)

    def read_event(handle: h5py.File) -> Optional[np.ndarray]:
        for key in ("event_frames", "frames"):
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return handle[key][:]
        for name in episode_names(handle):
            if name not in handle:
                continue
            group = handle[name]
            for key in ("frames", "event_frame", "event_frames"):
                if key in group:
                    return group[key][:]
        return None

    def read_rgb(handle: h5py.File) -> Optional[np.ndarray]:
        for key in ("rgb_frames", "rgb"):
            if key in handle and isinstance(handle[key], h5py.Dataset):
                return handle[key][:]
        for name in episode_names(handle):
            if name not in handle:
                continue
            group = handle[name]
            for key in ("clean", "rgb", "rgb_frames"):
                if key in group:
                    return group[key][:]
        return None

    with h5py.File(path, "r") as f:
        event_frames = read_event(f)
        rgb_frames = read_rgb(f)
    if rgb_path is not None:
        with h5py.File(rgb_path, "r") as f:
            rgb_frames = read_rgb(f)
    return event_frames, rgb_frames


def read_prophesee_raw(
    path: str,
    max_events: Optional[int] = None,
) -> np.ndarray:
    """
    Read Prophesee .raw events.
    
    Requires metavision_core or manual binary parsing.
    Returns (N, 4) [x, y, p, t] float32.
    """
    # Try metavision SDK first
    try:
        from metavision_core.event_io import EventsIterator
        events_list = []
        mv_iter = EventsIterator(str(path), delta_t=10000)
        for evs in mv_iter:
            events_list.append(np.stack([
                evs['x'], evs['y'], evs['p'].astype(np.float32), evs['t']
            ], axis=1))
            if max_events and sum(len(e) for e in events_list) >= max_events:
                break
        events = np.concatenate(events_list, axis=0).astype(np.float32)
        if max_events and len(events) > max_events:
            events = events[:max_events]
        return canonicalize_events(
            events, layout='xypt', timestamp_unit='us', sort=True
        )
    except ImportError:
        pass
    
    # Fallback: try RAW format manually
    # Prophesee RAW format: https://docs.prophesee.ai/stable/data/file_formats/raw.html
    # Event CD v2: 8 bytes per event
    # This is a simplified reader — full implementation needs EVT2/EVT3 decoding
    raise ImportError(
        "metavision_core not available. "
        "Install with: pip install metavision_core\n"
        "Or: sudo apt install python3-metavision-sdk-core"
    )


def read_events(
    path: str,
    max_events: Optional[int] = None,
    format: Optional[str] = None,
    layout: str = 'xypt',
    timestamp_unit: str = 's',
) -> np.ndarray:
    """
    Auto-detect format and read events.
    
    Supported formats:
      - .h5, .hdf5: Genesis HDF5
      - .raw: Prophesee RAW
      - .npy: numpy array (N,4)
    """
    path = Path(path)
    fmt = format or path.suffix.lower()
    
    if fmt in ('.h5', '.hdf5'):
        return read_genesis_hdf5(str(path), max_events)
    elif fmt == '.raw':
        return read_prophesee_raw(str(path), max_events)
    elif fmt == '.npy':
        events = np.load(str(path))
        events = canonicalize_events(
            events, layout=layout, timestamp_unit=timestamp_unit, sort=True
        )
        if max_events and len(events) > max_events:
            idx = np.linspace(0, len(events) - 1, max_events, dtype=int)
            events = events[idx]
        return events
    else:
        raise ValueError(f"Unknown format: {fmt}")
