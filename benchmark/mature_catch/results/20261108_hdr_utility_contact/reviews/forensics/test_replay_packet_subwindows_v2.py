"""Synthetic/local checks only; no actual HDF replay or physical success claim."""
from pathlib import Path
import importlib.util,json,os,sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('subwindow_diagnostic_v2',ROOT/'replay_packet_subwindows_v2.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


class SubwindowChecks(unittest.TestCase):
    def geometry(self):
        times=np.array([0.,.04]);poses=np.stack([np.eye(4),np.eye(4)])
        return times,poses,np.array([0.,.05]),np.array([-1,0])

    def test_partition_has_no_duplicate_or_dropped_boundary_events(self):
        times,poses,available,ids=self.geometry()
        ts=np.array([0.,.01-1e-12,.01,.02,.03,.04])
        raw=np.column_stack([ts,np.arange(len(ts)),np.zeros(len(ts)),np.ones(len(ts))])
        packet=dict(events=raw,end_s=.04,available_s=.05,T=poses[1])
        parts=m.split_packet(packet,times,poses,available,.01,ids)
        self.assertEqual([len(s['events']) for s in parts],[2,1,1,2])
        np.testing.assert_array_equal(np.concatenate([s['events'] for s in parts]),raw)
        self.assertTrue(all(s['available_s']==.05 and s['parent_packet_id']==0 for s in parts))

    def test_pose_interpolation_and_exact_endpoints(self):
        a=np.eye(4);b=np.eye(4);b[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();b[:3,3]=[2,4,6]
        np.testing.assert_array_equal(m.pose_between(0.,0.,1.,a,b),a)
        np.testing.assert_array_equal(m.pose_between(1.,0.,1.,a,b),b)
        mid=m.pose_between(.5,0.,1.,a,b)
        np.testing.assert_allclose(mid[:3,3],[1,2,3],atol=1e-12)
        np.testing.assert_allclose(mid[:3,:3],Rotation.from_euler('z',45,degrees=True).as_matrix(),atol=1e-12)

    def test_actual_empty_reset_packet(self):
        times,poses,available,ids=self.geometry();ids[0]=0
        packet=dict(events=np.zeros((0,4)),end_s=0.,available_s=.01,T=poses[0])
        self.assertEqual(m.split_packet(packet,times,poses,available,.01,ids),[])
        packet['events']=np.array([[0.,1,1,1]])
        with self.assertRaises(AssertionError):m.split_packet(packet,times,poses,available,.01,ids)

    def test_future_camera_endpoint_is_rejected(self):
        times,poses,available,ids=self.geometry();available[1]=.08
        packet=dict(events=np.zeros((0,4)),end_s=.04,available_s=.05,T=poses[1])
        with self.assertRaises(AssertionError):m.split_packet(packet,times,poses,available,.01,ids)

    def test_internal_multi_output_no_duplicate_append(self):
        class Fake:
            def __init__(self):self.samples=[];self.last_event_time=-np.inf
            def _detect(self,packet):
                if not len(packet['events']):return None
                return dict(t=float(np.median(packet['events'][:,0])),position=[0,0,1],
                    maximum_event_s=float(packet['events'][:,0].max()),available_s=packet['available_s'],camera_origin=packet['T'][:3,3].tolist())
        t=Fake();times,poses,available,ids=self.geometry()
        m.install_subwindows(t,Fake,times,poses,available,ids,.01,serial)
        raw=np.array([[.005,1,2,1],[.015,1,2,1],[.025,1,2,1],[.035,1,2,1]])
        self.assertIsNone(t._detect(dict(events=raw,end_s=.04,available_s=.05,T=poses[1])))
        self.assertEqual(len(t.samples),4)
        self.assertTrue(all(x['available_s']==.05 for x in t.samples))
        self.assertAlmostEqual(t.last_event_time,.035)

    def late_fixture(self):
        class Fake:
            def __init__(self):self.samples=[];self.last_event_time=-np.inf;self.filter_time=None
            def _detect(self,packet):
                if not len(packet['events']):return None
                return dict(t=float(np.median(packet['events'][:,0])),position=[0,0,1],
                    maximum_event_s=float(packet['events'][:,0].max()),available_s=packet['available_s'],camera_origin=packet['T'][:3,3].tolist())
        times=np.array([0.,.04,.08]);poses=np.stack([np.eye(4),np.eye(4),np.eye(4)])
        poses[1,0,3]=.04;poses[2,0,3]=.08
        available=np.array([0.,.05,.09]);ids=np.array([0,1,2])
        tracker=Fake();m.install_subwindows(tracker,Fake,times,poses,available,ids,.01,serial)
        def packet(ts,end,arrival):
            raw=np.column_stack([np.array(ts),np.arange(len(ts)),np.zeros(len(ts)),np.ones(len(ts))])
            return dict(events=raw,end_s=end,available_s=arrival,T=poses[int(round(end/.04))])
        return tracker,packet

    def test_true_carry_in_keeps_late_availability_and_repeated_window_end(self):
        tracker,packet=self.late_fixture()
        tracker._detect(packet([.035,.039],.04,.05))
        second=packet([.0395,.045,.055,.065,.075],.08,.09)
        tracker._detect(second)
        parent=tracker._subwindow_history[-1];carry=parent['windows'][0]
        self.assertEqual(len(tracker.samples),6)
        self.assertTrue(carry['carry_in_historical_window'])
        self.assertAlmostEqual(carry['end_s'],.04)
        self.assertEqual(carry['same_window_end_previous_parent_ids'],[1])
        self.assertEqual(carry['parent_packet_id'],2)
        self.assertEqual(carry['input_event_index_range'],[0,1])
        self.assertEqual(carry['available_s'],.09)
        self.assertEqual(carry['camera_endpoint_times_s'],[.04])
        np.testing.assert_allclose(carry['camera_to_world'][0][3],.04)
        self.assertEqual(carry['detected']['t'],.0395)
        self.assertTrue(all(s['available_s']==.09 for s in tracker.samples[1:]))
        self.assertEqual(sum(w['events'] for w in parent['windows']),len(second['events']))

    def test_truly_out_of_sequence_events_stop_without_resetting_filter(self):
        tracker,packet=self.late_fixture()
        tracker._detect(packet([.035,.039],.04,.05));tracker.filter_time=.037
        with self.assertRaisesRegex(RuntimeError,'out_of_sequence_raw_events'):
            tracker._detect(packet([.0345,.045],.08,.09))
        self.assertEqual(len(tracker.samples),1)
        self.assertEqual(tracker.filter_time,.037)

    def test_atomic_same_timestamp_across_parents_is_recorded_not_deduplicated(self):
        tracker,packet=self.late_fixture()
        tracker._detect(packet([.0395],.04,.05));tracker.filter_time=.0395
        tracker._detect(packet([.0395,.045],.08,.09))
        self.assertEqual(len(tracker.samples),3)
        carry=tracker._subwindow_history[-1]['windows'][0]
        self.assertTrue(carry['same_measurement_timestamp_as_previous'])
        self.assertEqual(tracker._subwindow_history[-1]['raw_events_equal_previous_parent_max'],1)
        self.assertEqual(carry['detected']['available_s'],.09)
        self.assertEqual(tracker.filter_time,.0395)

    def test_measurement_older_than_filter_stops_instead_of_rewinding(self):
        tracker,packet=self.late_fixture()
        tracker._detect(packet([.035,.039],.04,.05));tracker.filter_time=.042
        with self.assertRaisesRegex(RuntimeError,'measurement_older_than_already_processed'):
            tracker._detect(packet([.0395,.045],.08,.09))
        self.assertEqual(len(tracker.samples),1)
        self.assertEqual(tracker.filter_time,.042)

    def test_retrospective_early_onset_is_not_early_published_control(self):
        states=[dict(time_s=.64,position=[0,0,1],motion_onset_s=None),dict(time_s=.72,position=[0,0,1],motion_onset_s=.659)]
        score=m.scoring_only_motion_publication(states,.67)
        self.assertEqual(score['scoring_only_first_control_moving_time_s'],.72)
        self.assertFalse(score['scoring_only_pre_release_moving_control'])
        self.assertAlmostEqual(score['scoring_only_first_control_moving_delay_from_release_s'],.05)
        states[0]['motion_onset_s']=.62
        self.assertTrue(m.scoring_only_motion_publication(states,.67)['scoring_only_pre_release_moving_control'])

    def test_actual_native_detector_static_negative_and_motion_confirmation(self):
        source=Path('//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-robot-catch-hdr-utility-20261108/benchmark/mature_catch')
        sys.path.insert(0,str(source));os.environ.update(CATCHIT_EVENT_FLIGHT_FIT='0',CATCHIT_REVOCABLE_GRASP='0',CATCHIT_REJECT_PARTIAL_CONTOURS='0')
        from event_sphere_tracker import EventSphereTracker
        K=np.array([[433.,0,320.],[0,433.,240.],[0,0,1.]])
        T=np.eye(4);T[:3,:3]=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]);T[:3,3]=[0.,0.,1.4]
        angle=np.linspace(0,2*np.pi,128,endpoint=False)
        for moving in [False,True]:
            tracker=EventSphereTracker(K,(640,480),.04)
            tracker.initialize_rgb(dict(valid=True,position=[0.,2.,1.],camera_origin=[0.,0.,1.4],radius_px=8.66,range_std_m=.01,initialization_mode='single_frame_RGBD'))
            m.install_subwindows(tracker,EventSphereTracker,np.array([0.,.04]),np.stack([T,T]),np.array([0.,.05]),np.array([-1,0]),.01,serial)
            events=[]
            for j in range(4):
                ts=j*.01+np.linspace(.001,.009,len(angle))
                events.append(np.column_stack([ts,320+8.66*np.cos(angle),326.6-(12*j if moving else 0)+8.66*np.sin(angle),np.ones(len(angle))]))
            tracker._detect(dict(events=np.concatenate(events),end_s=.04,available_s=.05,T=T))
            self.assertEqual(len(tracker.samples),4)
            state=tracker._predict(.08)
            self.assertIsNotNone(state)
            self.assertTrue(all(s['available_s']==.05 for s in tracker.samples))
            if moving:self.assertIsNotNone(tracker.motion_onset_s,msg=str({'samples':tracker.samples,'suspicion':tracker.motion_suspicion,'processed':tracker.filter_processed}))
            else:
                self.assertIsNone(tracker.motion_onset_s)
                np.testing.assert_allclose(state['position'],[0,2,1],atol=1e-12)
            self.assertNotIn('genesis',sys.modules);self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SubwindowChecks))
    (ROOT/'SUBWINDOW_V2_SYNTHETIC_TEST_STATUS.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        scope='synthetic/local CPU only; native detector used but no archived raw replay, renderer, HDF, GPU or physical episode',
        actual_archived_baseline_parity='NOT_RUN_LOCAL',failures=len(result.failures),errors=len(result.errors)),indent=2),encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
