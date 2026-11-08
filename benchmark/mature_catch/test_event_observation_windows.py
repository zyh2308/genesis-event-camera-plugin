"""CPU contract tests; actual cached parity remains a separate run."""
import json
import os
import unittest
from unittest.mock import patch
import numpy as np

from event_observation_windows import EventObservationWindows, history_camera_pose
from event_sphere_tracker import EventSphereTracker
from event_sphere_tracker_rgbd_base import EventSphereTracker as Baseline


class FakeTracker:
    def __init__(self):
        self.samples=[]; self.last_event_time=-np.inf; self.filter_time=None; self.predict_calls=[]
    def detect(self, packet):
        if not len(packet['events']): return None
        return dict(t=float(np.median(packet['events'][:,0])), position=[0.,0.,1.],
            camera_origin=packet['T'][:3,3].tolist(), available_s=packet['available_s'],
            maximum_event_s=float(packet['events'][:,0].max()))
    def _predict(self,t): self.predict_calls.append((t,len(self.samples)))


class WindowTests(unittest.TestCase):
    def packet(self,ts,end,available,parent):
        T=np.eye(4); T[0,3]=end
        events=np.column_stack((np.array(ts),np.arange(len(ts)),np.zeros(len(ts)),np.ones(len(ts))))
        return dict(events=events,end_s=end,available_s=available,T=T,parent_packet_id=parent)

    def ready(self):
        windows=EventObservationWindows(10); tracker=FakeTracker()
        reset=self.packet([],0.,.01,0)
        windows.register_parent(reset)
        windows.consume(tracker,reset,tracker.detect)
        return windows,tracker

    def test_default_zero_delegates_original_methods_without_packet_rewrite(self):
        K=np.eye(3)
        for value in [None,'0']:
            env={'CATCHIT_EVENT_FLIGHT_FIT':'0','CATCHIT_REJECT_PARTIAL_CONTOURS':'0'}
            if value is not None:env['CATCHIT_EVENT_OBSERVATION_WINDOW_MS']=value
            with patch.dict(os.environ,env,clear=True): tracker=EventSphereTracker(K,(640,480),.04)
            self.assertIsNone(tracker._observation_windows)
            self.assertEqual(tracker.observation_window_audit,[])
            sentinel=object(); events=np.zeros((0,4)); T=np.eye(4)
            with patch.object(Baseline,'push',return_value=sentinel) as old:
                self.assertIs(tracker.push(events,.05,.04,T),sentinel)
                old.assert_called_once_with(events,.05,.04,T)
            packet={'events':events,'T':T}; sample={'position':[1,2,3]}
            with patch.object(Baseline,'_detect',return_value=sample) as old:
                self.assertIs(tracker._detect(packet),sample); old.assert_called_once_with(packet)
            self.assertEqual(set(packet),{'events','T'})
            with patch.object(Baseline,'observe',return_value=sentinel) as old:
                self.assertIs(tracker.observe(.08,T,[]),sentinel); old.assert_called_once_with(.08,T,[])

    def test_reset_register_then_detect_is_idempotent_and_empty(self):
        windows,tracker=self.ready()
        self.assertEqual(len(windows.endpoints),1)
        self.assertEqual(len(tracker.samples),0)
        self.assertEqual(windows.audit[0]['subwindows'],0)
        self.assertIsNone(windows.audit[0]['processing_now_s'])
        with self.assertRaises(AssertionError):windows.consume(tracker,self.packet([],0.,.01,0),tracker.detect)

    def test_carry_in_parent_ids_unique_ranges_and_duplicate_child_end(self):
        windows,tracker=self.ready()
        p1=self.packet([.035,.039],.04,.05,1)
        windows.register_parent(p1); windows.consume(tracker,p1,tracker.detect)
        p2=self.packet([.0395,.045,.055,.065,.075],.08,.09,2)
        windows.register_parent(p2); windows.consume(tracker,p2,tracker.detect)
        self.assertEqual(len(windows.endpoints),3)
        self.assertEqual(len(tracker.samples),6)
        carry=windows.audit[-1]['windows'][0]
        self.assertEqual(carry['end_s'],.04)
        self.assertEqual(carry['available_s'],.09)
        self.assertEqual(carry['same_window_end_previous_parent_ids'],[1])
        self.assertEqual(carry['detected']['t'],.0395)
        self.assertEqual(carry['camera_to_world'][0][3],.04)
        ranges=[w['input_event_index_range'] for w in windows.audit[-1]['windows']]
        self.assertEqual(ranges,[[0,1],[1,2],[2,3],[3,4],[4,5]])
        self.assertEqual(tracker.predict_calls[-5:][0],(.04,2))
        json.dumps(windows.audit)  # Sensor.close must be able to persist this.

    def test_future_camera_and_unavailable_parent_are_rejected(self):
        with self.assertRaises(AssertionError):
            history_camera_pose(.02,np.array([0.,.04]),np.stack([np.eye(4),np.eye(4)]),np.array([0.,.1]),.05)
        windows,tracker=self.ready();windows.processing_now_s=.049
        with self.assertRaises(AssertionError):windows.consume(tracker,self.packet([.035],.04,.05,1),tracker.detect)
        self.assertEqual(len(tracker.samples),0)

    def test_true_out_of_sequence_stops_without_rewinding(self):
        windows,tracker=self.ready()
        windows.consume(tracker,self.packet([.035,.039],.04,.05,1),tracker.detect)
        tracker.filter_time=.037
        with self.assertRaisesRegex(RuntimeError,'Out-of-sequence raw'):
            windows.consume(tracker,self.packet([.0345,.045],.08,.09,2),tracker.detect)
        self.assertEqual(tracker.filter_time,.037)
        self.assertEqual(len(tracker.samples),1)
        self.assertEqual(windows.audit[-1]['status'],'STOP_OUT_OF_SEQUENCE_RAW')

    def test_real_native_push_observe_availability_and_full_child_detector(self):
        env={'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10','CATCHIT_EVENT_FLIGHT_FIT':'0',
             'CATCHIT_REJECT_PARTIAL_CONTOURS':'0','CATCHIT_REVOCABLE_GRASP':'0'}
        with patch.dict(os.environ,env):
            K=np.array([[433.,0,320.],[0,433.,240.],[0,0,1.]])
            tracker=EventSphereTracker(K,(640,480),.04)
        initial=dict(valid=True,position=[0.,2.,1.],camera_origin=[0.,0.,1.4],radius_px=8.66,
                     range_std_m=.01,initialization_mode='single_frame_RGBD')
        tracker.initialize_rgb(initial)
        T=np.eye(4);T[:3,:3]=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]);T[:3,3]=[0.,0.,1.4]
        tracker.push(np.zeros((0,4)),.01,0.,T,parent_packet_id=0)
        tracker.observe(.005,np.eye(4),[])
        self.assertEqual(len(tracker.observation_window_audit),0)
        tracker.observe(.02,np.eye(4),[])
        angle=np.linspace(0,2*np.pi,128,endpoint=False);events=[]
        for j in range(4):
            ts=.01*j+np.linspace(.001,.009,len(angle))
            events.append(np.column_stack((ts,320+8.66*np.cos(angle),326.6+8.66*np.sin(angle),np.ones(len(angle)))))
        tracker.push(np.concatenate(events),.05,.04,T,parent_packet_id=1)
        tracker.observe(.049,np.eye(4),[])
        self.assertEqual(len(tracker.samples),0)
        state=tracker.observe(.08,np.eye(4),[])
        self.assertEqual(len(tracker.samples),4)
        self.assertTrue(all(s['available_s']==.05 and 'camera_origin' in s for s in tracker.samples))
        self.assertIsNone(tracker.motion_onset_s)
        np.testing.assert_array_equal(state['position'],initial['position'])
        self.assertEqual(tracker.observation_window_audit[-1]['processing_now_s'],.08)
        self.assertEqual(tracker.observation_window_audit[-1]['added_samples'],4)
        json.dumps(tracker.observation_window_audit)


if __name__=='__main__':unittest.main(verbosity=2)
