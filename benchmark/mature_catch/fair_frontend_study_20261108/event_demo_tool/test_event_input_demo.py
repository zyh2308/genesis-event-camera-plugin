"""Causality/visual normalization checks; no synthetic demo files are produced."""
import sys,unittest
sys.dont_write_bytecode=True
import numpy as np
import make_event_input_demo as demo


def raw():
    return dict(events=np.asarray([[.041,20,30,1],[.070,21,31,-1],[.079,20,30,1]],dtype=float),
        availability=[.01,.09],times=[0.,.08],offsets=[0,0,3],cache={})


class EventDemoContracts(unittest.TestCase):
    def test_parent_arrival_not_event_or_endpoint_time_gates_display(self):
        data=raw()
        self.assertEqual(demo.latest_packet(data,0),-1)
        self.assertEqual(demo.latest_packet(data,.08),0)
        self.assertEqual(demo.latest_packet(data,.089),0)
        self.assertEqual(demo.latest_packet(data,.09),1)
    def test_window_is_same40ms_and_only_available_prefix(self):
        data=raw();before=demo.event_picture(data,0);after=demo.event_picture(data,1)
        self.assertEqual(before['event_count'],0);self.assertEqual(after['event_count'],3)
        self.assertAlmostEqual(after['window_end_s']-after['window_start_s'],.04)
        self.assertLessEqual(after['maximum_event_s'],after['parent_time_s'])
    def test_empty_prerelease_has_no_fake_events(self):
        result=demo.event_picture(raw(),-1)
        self.assertEqual(result['event_count'],0);self.assertIsNone(result['available_s'])
        values=np.asarray(result['image']);self.assertTrue((values==np.asarray([8,17,27])).all())
    def test_held_packet_reuses_exact_picture_without_sliding_window(self):
        data=raw();index=demo.latest_packet(data,9.)
        first=demo.event_picture(data,index);second=demo.event_picture(data,index)
        self.assertIs(first,second);self.assertEqual(second['window_end_s'],.08)
    def test_shared_fixed_count_normalization_and_polarity(self):
        a=raw();b=raw();one=demo.event_picture(a,1);two=demo.event_picture(b,1)
        self.assertTrue(np.array_equal(np.asarray(one['image']),np.asarray(two['image'])))
        pixels=np.asarray(one['image'])
        self.assertGreater(int(pixels[30,20,1]),int(pixels[30,20,0])) # ON cyan.
        self.assertGreater(int(pixels[31,21,0]),int(pixels[31,21,1])) # OFF coral.


if __name__=='__main__':unittest.main()
