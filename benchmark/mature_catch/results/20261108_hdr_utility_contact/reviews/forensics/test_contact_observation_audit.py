"""Artifact contract tests, using a real production audit fixture plus corruption."""
from pathlib import Path
import copy,json,sys,unittest
import numpy as np

root=Path(__file__).resolve().parent;sys.path.insert(0,str(root.parent))
sys.path.insert(0,'//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-robot-catch-hdr-utility-20261108/benchmark/mature_catch')
from event_observation_windows import EventObservationWindows
from audit_contact_observation_windows import audit_data


def fixture():
 class Tracker:
  def __init__(self):self.samples=[];self.last_event_time=-np.inf;self.filter_time=None
  def _predict(self,t):return None
  def detect(self,p):
   if not len(p['events']):return None
   return dict(t=float(np.median(p['events'][:,0])),position=[0.,0.,1.],uv=[320.,240.],
    radius_px=9.,quality=8.,camera_origin=p['T'][:3,3].tolist(),available_s=p['available_s'],
    maximum_event_s=float(p['events'][:,0].max()))
 window=EventObservationWindows(10);tracker=Tracker();packets=[];parts=[];poses=[]
 times=np.array([0.,.04,.08,.12]);event_times=[[],[.035,.039],[.0395,.045,.055,.065,.075],[.115]]
 for pid,(t,ts) in enumerate(zip(times,event_times)):
  raw=np.column_stack((np.array(ts),np.zeros(len(ts)),np.zeros(len(ts)),np.ones(len(ts))))
  T=np.eye(4);T[0,3]=t;poses.append(T)
  p=dict(events=raw,end_s=float(t),available_s=float(t+.01),T=T,parent_packet_id=pid)
  window.register_parent(p);packets.append(dict(time_s=float(t),available_s=float(t+.01),events=len(raw)));parts.append(raw)
  if pid<3:
   window.processing_now_s=float(t+.04);window.consume(tracker,p,tracker.detect)
 doc=dict(observation_window_ms=10.,windows=copy.deepcopy(window.audit))
 # Physical endpoint can be 0.15, after pending arrival 0.13, while no controller
 # poll occurs after 0.12. The final packet must still be allowed to stay pending.
 trace=[dict(t_before=0.,t=.04,event_state=None),dict(t_before=.04,t=.08,event_state=dict(maximum_event_s=None)),
        dict(t_before=.08,t=.12,event_state=dict(maximum_event_s=.039)),
        dict(t_before=.12,t=.15,event_state=dict(maximum_event_s=.075))]
 return dict(protocol=dict(event_observation_window_ms=10.),packets=packets,raw=np.concatenate(parts),
  camera_times=times,camera_poses=np.array(poses),document=doc,trace=trace,samples=copy.deepcopy(tracker.samples))


class AuditTests(unittest.TestCase):
 def test_valid_carry_and_pending_at_physical_end(self):
  result=audit_data(**fixture())
  self.assertEqual(result['pending_parents_at_final_control'],1)
  self.assertEqual(result['pending_events'],1)
  self.assertEqual(result['carry_in_events'],1)
  self.assertEqual(result['repeated_child_endpoints'],1)
  self.assertEqual(result['consumed_parents'],3)
 def test_changed_protocol_or_child_availability_rejected(self):
  d=fixture();d['protocol']['event_observation_window_ms']=0
  with self.assertRaises(AssertionError):audit_data(**d)
  d=fixture();d['document']['windows'][2]['windows'][0]['available_s']=.05
  with self.assertRaisesRegex(AssertionError,'availability changed'):audit_data(**d)
 def test_omitting_due_parent_rejected_without_requiring_future_parent(self):
  d=fixture();d['document']['windows'].pop()
  with self.assertRaisesRegex(AssertionError,'consumed parents differ'):audit_data(**d)
 def test_duplicate_input_indices_rejected(self):
  d=fixture();d['document']['windows'][2]['windows'][1]['input_event_index_range']=[0,2]
  with self.assertRaisesRegex(AssertionError,'event indices'):audit_data(**d)
 def test_camera_geometry_or_future_support_rejected(self):
  d=fixture();d['document']['windows'][2]['windows'][0]['camera_to_world'][0][3]+=.1
  with self.assertRaises(AssertionError):audit_data(**d)
  d=fixture();d['document']['windows'][1]['windows'][0]['camera_endpoint_available_s'][-1]=.09
  with self.assertRaisesRegex(AssertionError,'camera support differs'):audit_data(**d)
 def test_early_processing_or_duplicate_append_rejected(self):
  d=fixture();d['document']['windows'][2]['processing_now_s']=.08
  with self.assertRaisesRegex(AssertionError,'first available control poll'):audit_data(**d)
  d=fixture();d['samples'].append(copy.deepcopy(d['samples'][-1]))
  with self.assertRaisesRegex(AssertionError,'missing or duplicate appends'):audit_data(**d)


if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
 (root/'CONTACT_OBSERVATION_AUDIT_TEST_STATUS.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
  tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),
  scope='Local CPU artifact-contract tests; actual online actor audit not yet claimed'),indent=2),encoding='utf-8')
 sys.exit(0 if result.wasSuccessful() else 1)
