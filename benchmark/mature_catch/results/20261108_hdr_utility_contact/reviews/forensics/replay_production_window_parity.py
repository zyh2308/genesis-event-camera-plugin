"""Real-cache production extraction parity, separate from physical catch scoring."""
from pathlib import Path
import argparse,importlib.util,json,os,sys
import numpy as np

def same(a,b,label):
 assert len(a['samples'])==len(b['samples']),label
 for x,y in zip(a['samples'],b['samples']):
  for key in ['t','position','uv','radius_px','quality','available_s','maximum_event_s','camera_origin']:
   np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-9,err_msg=label+' '+key)
 assert len(a['states'])==len(b['states'])
 for x,y in zip(a['states'],b['states']):
  for key in ['time_s','position','velocity','motion_onset_s','detected_samples','suspicion_times']:
   assert (x[key] is None)==(y[key] is None),(label,key)
   if x[key] is not None:np.testing.assert_allclose(x[key],y[key],rtol=0,atol=1e-9,err_msg=label+' '+key)

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--run',type=Path,action='append',required=True);a=p.parse_args();assert len(a.run)==8;a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.source));helper=Path(__file__).with_name('replay_static_association.py');spec=importlib.util.spec_from_file_location('window_production_raw_helper',helper);common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common);reference=json.loads((a.reference/'SUMMARY.json').read_text());records=[]
 for run in a.run:
  for window,label in [(0,'production_whole_packet_fit1'),(10,'production_subwindow10ms_fit1')]:
   os.environ.update(CATCHIT_EVENT_OBSERVATION_WINDOW_MS=str(window),CATCHIT_REVOCABLE_GRASP='0',CATCHIT_REJECT_PARTIAL_CONTOURS='0')
   result=common.replay(run,'production_native_window'+str(window),a.source,a.output,1.45,flight_fit=True);got=json.loads((a.output/result['file']).read_text());ref=next(x for x in reference['results'] if x['case']==result['case'] and x['method']==result['method'] and x['mode']==label);wanted=json.loads((a.reference/ref['file']).read_text());same(got,wanted,result['case']+' '+result['method']+' '+str(window));result['cached_v2_reference_parity']='PASS';records.append(result)
 assert 'torch' not in sys.modules and 'genesis' not in sys.modules
 receipt={'status':'PASS','actual_cache_inputs':8,'native_paths_checked':16,'default0_matches_preexisting_native':True,'enabled10_matches_validated_diagnostic':True,'query_timestamps_unchanged':True,'physics_steps':0,'renderer_calls':0,'CUDA_calls':0,'results':records};(a.output/'SUMMARY.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'status':'PASS','actual_cache_inputs':8,'native_paths_checked':16}),flush=True)
if __name__=='__main__':main()
