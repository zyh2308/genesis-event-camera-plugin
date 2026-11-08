"""Supplemental timing remains useful when detailed confirmation is unknown."""
from pathlib import Path
import json,sys,tempfile,types,unittest
from unittest.mock import patch
import numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent))
import extract_contact_cadence_timing as m


def fixture(run):
 sensor=run/'migration_audit/sensor_episode_000';sensor.mkdir(parents=True);(run/'episode_000').mkdir()
 samples=[dict(t=t,maximum_event_s=t+.001,available_s=.09) for t in [.055,.065]]
 children=[dict(subwindow_index=i,events=12,detected=s) for i,s in enumerate(samples)]
 doc=dict(windows=[dict(parent_packet_id=0,processing_now_s=.04,windows=[],carry_in_window_count=0),
     dict(parent_packet_id=1,processing_now_s=.12,windows=children,carry_in_window_count=0)])
 packets=[dict(time_s=0.,available_s=.01,events=0),dict(time_s=.08,available_s=.09,events=24)]
 for name,data in [('packet_audit.json',packets),('event_samples.json',samples),('event_observation_windows.json',doc),('initialization.json',{})]:
  (sensor/name).write_text(json.dumps(data))
 trace=[dict(t_before=t,event_state=None,policy_object_input=None,held_before_observation=False) for t in [0.,.04,.08]]
 trace.append(dict(t_before=.12,event_state=dict(motion_onset_s=.065),policy_object_input={'pos3d':[0,0,1]},held_before_observation=True))
 trace.append(dict(t_before=.16,event_state=dict(motion_onset_s=.065),policy_object_input={'pos3d':[0,0,1]},held_before_observation=False))
 (run/'episode_000/trace.json').write_text(json.dumps(trace))
 return dict(event_endpoint_rate_hz=12.5,sphere_radius_m=.04,timestep=.002,steps_per_policy=20)


class TimingChecks(unittest.TestCase):
 def test_unknown_confirmation_does_not_hide_actual_control_timing(self):
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);protocol=fixture(run)
   fake=types.ModuleType('h5py');fake.File=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('test unavailable detail'))
   with patch.dict(sys.modules,{'h5py':fake}):result=m.timing_episode(run,protocol,{},'x','ours')
   self.assertEqual(result['confirmation_details_status'],'UNKNOWN')
   self.assertEqual(result['first_control_published_moving_state_s'],.12)
   self.assertEqual(result['first_control_policy_adopted_moving_state_s'],.16)
   self.assertEqual(result['total_detected_subwindows'],2)
   self.assertEqual(result['original_parent_period_s'],.08)
   self.assertFalse((run/'episode_000/result.json').exists())
 def test_confirmed_same_packet_gap_does_not_change_late_arrival(self):
  class File:
   def __init__(self,*a,**k):pass
   def __enter__(self):return {'intrinsics':np.eye(3)}
   def __exit__(self,*a):pass
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);protocol=fixture(run);fake=types.ModuleType('h5py');fake.File=File
   with patch.dict(sys.modules,{'h5py':fake}),patch.object(m,'reconstruct_confirmation',return_value=dict(motion_onset_event_s=.065,confirmation_sample_indices=[0,1],source='test fixture')):
    result=m.timing_episode(run,protocol,{},'x','ours')
   self.assertEqual(result['confirmation_details_status'],'RECONSTRUCTED')
   self.assertTrue(result['confirmation_from_single_arrival_parent'])
   self.assertAlmostEqual(result['confirmation_max_adjacent_gap_s'],.01)
   self.assertEqual(result['confirmation_information_available_s'],.09)
   self.assertEqual(result['earliest_control_poll_after_confirmation_arrival_s'],.12)
   self.assertTrue(result['subwindow_confirmation_observed_with_parent_period_longer_than_gap'])


if __name__=='__main__':unittest.main(verbosity=2)
