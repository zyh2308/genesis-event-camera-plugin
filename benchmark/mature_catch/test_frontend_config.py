import unittest
from frontend_config import candidate_grid, resolve_frontend, validate_frontend


class FrontendConfigTests(unittest.TestCase):
    def test_symmetric_grid_is_complete_and_cannot_tune_generator(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 8)
        self.assertEqual(len({x['frontend_id'] for x in grid}), 8)
        for row in grid:
            self.assertEqual(validate_frontend(row['env']), row['env'])
        with self.assertRaises(ValueError):
            validate_frontend(dict(grid[0]['env'], CATCHIT_EVENT_THRESHOLD='0.1'))

    def test_no_selection_preserves_existing_environment(self):
        self.assertEqual(resolve_frontend({}, {}, 'ours'), {})

    def test_crossed_row_is_shared_across_sources(self):
        a, b = candidate_grid()[0]['env'], candidate_grid()[-1]['env']
        spec = dict(frontend_by_method=dict(ours=a, v2e=b))
        self.assertEqual(resolve_frontend(spec, {}, 'v2e'), b)
        for method in ['ours', 'v2e']:
            self.assertEqual(resolve_frontend(spec, dict(frontend_overrides=a), method), a)

    def test_incomplete_or_unregistered_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_frontend({'CATCHIT_EVENT_FLIGHT_FIT': '1'})
        values = dict(candidate_grid()[0]['env'])
        values['CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'] = '200'
        with self.assertRaises(ValueError):
            validate_frontend(values)


if __name__ == '__main__':
    unittest.main()
