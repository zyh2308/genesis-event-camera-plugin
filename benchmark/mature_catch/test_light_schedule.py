"""Pure illumination contract; never imports Genesis or initializes a GPU."""
import math
import unittest
from light_schedule import front_glare_intensity,front_glare_schedule,validated_peak

class PublicLightingScheduleTests(unittest.TestCase):
    def test_initialization_and_piecewise_endpoints(self):
        self.assertEqual(front_glare_intensity(0.),0.)
        self.assertEqual(front_glare_intensity(.099),0.)
        self.assertEqual(front_glare_intensity(.1),0.)
        self.assertAlmostEqual(front_glare_intensity(.15),200.)
        self.assertEqual(front_glare_intensity(.2),400.)
        self.assertEqual(front_glare_intensity(2.5),400.)

    def test_range_scaling_and_monotonicity(self):
        for peak in [0.,400.,1600.]:
            values=[front_glare_intensity(i/1000,peak) for i in range(2501)]
            self.assertTrue(all(0<=x<=peak for x in values))
            self.assertEqual(values,sorted(values))
            self.assertAlmostEqual(front_glare_intensity(.15,peak),peak/2)

    def test_reset_does_not_inherit_previous_lamp_value(self):
        self.assertEqual(front_glare_intensity(2.5,1600.),1600.)
        self.assertEqual(front_glare_intensity(0.,1600.),0.)
        self.assertEqual(front_glare_intensity(.1,1600.),0.)

    def test_invalid_inputs_rejected(self):
        for value in [-1.,math.nan,math.inf,-math.inf]:
            with self.assertRaises(ValueError):front_glare_intensity(value)
            with self.assertRaises(ValueError):validated_peak(value)
        self.assertEqual(validated_peak('1600'),1600.)

    def test_protocol_is_explicit(self):
        metadata=front_glare_schedule(1600.)
        self.assertEqual(metadata['peak_point_intensity_renderer_units'],1600.)
        self.assertEqual(metadata['point_position'],[0.,-.5,2.5])
        self.assertIn('no object state or release flags',metadata['time_source'])

if __name__=='__main__':unittest.main()
