import unittest
import numpy as np
from baf import BackgroundActivityFilter, EpisodeEncoder


def run(t_us, x, y, p=None, width=8, height=8):
    return BackgroundActivityFilter(width, height).process(np.array(t_us, np.float64) / 1e6,
                                                          np.array(x), np.array(y), p)


def scalar(t, x, y, width=8, height=8, initial=0.):
    surface = np.full((height, width), initial, dtype=np.float64)
    out = []
    for us, a, b in zip(t * 1e6, x, y):
        neighbours = [surface[ny, nx] for ny in range(max(0, b-1), min(height, b+2))
                      for nx in range(max(0, a-1), min(width, a+2)) if (nx, ny) != (a, b)]
        out.append(any(us - v < 2000 for v in neighbours))
        surface[b, a] = us
    return np.array(out), surface.ravel()


class BafTests(unittest.TestCase):
    def test_zero_start_and_strict_boundary(self):
        np.testing.assert_array_equal(run([0, 1999, 2000], [0, 4, 7], [0, 4, 7]), [True, True, False])
        np.testing.assert_array_equal(run([3000, 5000, 5000.125], [2, 3, 3], [2, 2, 3]), [False, False, True])

    def test_rejected_input_updates_surface_and_polarity_ignored(self):
        np.testing.assert_array_equal(run([3000, 3001], [2, 3], [2, 2], np.array([0, 1])), [False, True])
        np.testing.assert_array_equal(run([3000, 3001], [2, 3], [2, 2], np.array([1, 0])), [False, True])

    def test_own_pixel_excluded_but_diagonals_and_borders_work(self):
        np.testing.assert_array_equal(run([3000, 3001, 3002], [0, 0, 1], [0, 0, 1]), [False, False, True])
        np.testing.assert_array_equal(run([3000, 3001], [0, 7], [0, 7]), [False, False])

    def test_scalar_equivalence_prefix_and_packet_continuity(self):
        rng = np.random.default_rng(442)
        t = np.cumsum(rng.integers(0, 600, 1000)).astype(np.float64) / 1e6
        x, y = rng.integers(0, 8, (2, len(t)))
        expected, state = scalar(t, x, y)
        filt = BackgroundActivityFilter(8, 8)
        np.testing.assert_array_equal(filt.process(t, x, y), expected)
        np.testing.assert_array_equal(filt.surface, state)
        filt = BackgroundActivityFilter(8, 8)
        packet = np.concatenate([filt.process(t[a:b], x[a:b], y[a:b]) for a, b in ((0, 1), (1, 117), (117, 500), (500, 1000))])
        np.testing.assert_array_equal(packet, expected)
        np.testing.assert_array_equal(BackgroundActivityFilter(8, 8).process(t[:501], x[:501], y[:501]), expected[:501])

    def test_startup_zero_vs_minus_inf_agree_at_or_after_two_ms(self):
        rng = np.random.default_rng(19)
        t = np.sort(rng.uniform(0, .02, 500))
        x, y = rng.integers(0, 8, (2, len(t)))
        zero = BackgroundActivityFilter(8, 8).process(t, x, y)
        absent, _ = scalar(t, x, y, initial=-np.inf)
        np.testing.assert_array_equal(zero[t * 1e6 >= 2000], absent[t * 1e6 >= 2000])

    def test_fractional_timestamps_and_original_arrays_unchanged(self):
        t = np.array([.0030001, .0050000, .0070001]); x = np.array([2, 3, 4]); y = x.copy()
        originals = [v.copy() for v in (t, x, y)]
        expected, _ = scalar(t, x, y)
        np.testing.assert_array_equal(BackgroundActivityFilter(8, 8).process(t, x, y), expected)
        for a, b in zip((t, x, y), originals): np.testing.assert_array_equal(a, b)

    def test_empty_and_invalid_input_rejected(self):
        f = BackgroundActivityFilter(8, 8)
        self.assertEqual(len(f.process(np.array([]), np.array([], int), np.array([], int))), 0)
        for t, x, y in (([.003, .002], [0, 1], [0, 1]), ([np.nan], [0], [0]), ([.003], [8], [0]), ([-.001], [0], [0])):
            with self.assertRaises(ValueError): f.process(np.array(t), np.array(x), np.array(y))
        f.process(np.array([.01]), np.array([2]), np.array([2]))
        with self.assertRaises(ValueError): f.process(np.array([.009]), np.array([2]), np.array([2]))

    def test_episode_encoder_cache_exact_raw_and_no_window_reset_or_drop(self):
        calls = []
        def encode(t, x, y, p, start, end, w, h, scale):
            selected = (t > start) & (t <= end)
            result = np.zeros((3, h, w), np.float32)
            for a, b, polarity in zip(x[selected], y[selected], p[selected]): result[int(polarity), b, a] += 1
            calls.append(t)
            return result
        t = np.array([.010, .011, .030]); x = np.array([2, 3, 7]); y = np.array([2, 2, 7]); p = np.array([0, 1, 0])
        wrapped = EpisodeEncoder(encode)
        for start, end in ((.009, .0105), (.0105, .012), (.025, .031)):
            expected = encode(t, x, y, p, start, end, 8, 8, 8.)
            np.testing.assert_array_equal(wrapped(t, x, y, p, start, end, 8, 8, 8.), expected)
        self.assertEqual(len(wrapped.episodes), 1)
        self.assertEqual(len(wrapped.windows), 3)
        self.assertEqual([r['retained_events'] for r in wrapped.windows], [0, 1, 0])
        self.assertTrue(all(a is b for a, b in zip(next(iter(wrapped.episodes.values()))['refs'], (t, x, y, p))))
        self.assertEqual(wrapped.windows[-1]['filtered_image'].sum(), 0.)
        wrapped(t.copy(), x.copy(), y.copy(), p.copy(), .0105, .012, 8, 8, 8.)
        self.assertEqual(len(wrapped.episodes), 2)


if __name__ == '__main__': unittest.main(verbosity=2)
