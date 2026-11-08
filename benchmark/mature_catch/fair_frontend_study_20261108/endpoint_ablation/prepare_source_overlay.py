"""Prepare three benchmark-only overlay files; never edit original/frozen source."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace_once(text,old,new):
 if text.count(old)!=1:raise ValueError('Expected exactly one sealed source hunk: '+old[:90])
 return text.replace(old,new,1)


def prepare(source,output):
 source=Path(source).resolve();output=Path(output).resolve();here=Path(__file__).resolve().parent
 metadata=json.loads((here.parent/'BASE_RUNTIME.json').read_text());patched={};original={}
 for name in ('sparse_event_sensor.py','run_original.py'):
  if sha(source/name)!=metadata['runtime_files'][name]:raise ValueError('Original source differs from sealed4931807: '+name)
  original[name]=(source/name).read_text(encoding='utf-8')
 sensor=original['sparse_event_sensor.py']
 sensor=replace_once(sensor,"        assert self.generator_method in ('ours','v2e')",'''        assert self.generator_method in ('ours','v2e')
        self.reconstruction_mode=os.environ.get('CATCHIT_EVENT_RECONSTRUCTION_MODE','full')
        if self.reconstruction_mode not in ('full','endpoint_only'):
            raise ValueError('Unregistered event reconstruction mode')
        if self.reconstruction_mode=='endpoint_only' and self.generator_method!='ours':
            raise ValueError('Endpoint-only ablation applies only to the frozen Ours response')''')
 sensor=replace_once(sensor,"        worker_name='event_worker.py' if self.generator_method=='ours' else 'v2e_event_worker.py'",
  "        worker_name=('endpoint_event_worker.py' if self.reconstruction_mode=='endpoint_only' else 'event_worker.py') if self.generator_method=='ours' else 'v2e_event_worker.py'")
 sensor=replace_once(sensor,"            ready=decode(pickle.load(self.worker.stdout));assert ready['ready']",'''            ready=decode(pickle.load(self.worker.stdout));assert ready['ready']
            if self.reconstruction_mode=='endpoint_only':
                assert ready['reconstruction_mode']=='endpoint_only' and ready['intermediate_frames']==0''')
 sensor=replace_once(sensor,"        worker_packet=packet if self.generator_method=='ours' else dict(time_s=float(t),gray8=gray8)",'''        worker_packet=packet if self.generator_method=='ours' else dict(time_s=float(t),gray8=gray8)
        if self.reconstruction_mode=='endpoint_only':
            worker_packet={key:packet[key] for key in ('time_s','resolution_wh','intrinsics','radiance')}''')
 sensor=replace_once(sensor,"        reply=decode(pickle.load(self.worker.stdout));events=reply['events']",'''        reply=decode(pickle.load(self.worker.stdout));events=reply['events']
        if self.reconstruction_mode=='endpoint_only':
            assert reply['reconstruction_mode']=='endpoint_only' and reply['intermediate_frames']==0
            assert reply['endpoint_reads']==self.frame_index+1 and reply['dvs_update_calls']==self.frame_index
            assert reply['packet_end_s']==float(t)''')
 sensor=replace_once(sensor,"            hdr_used_only_for_audit=self.generator_method=='v2e',worker_input_keys=sorted(worker_packet)))",'''            hdr_used_only_for_audit=self.generator_method=='v2e',worker_input_keys=sorted(worker_packet),
            reconstruction_mode=reply.get('reconstruction_mode',self.reconstruction_mode),
            intermediate_frames=reply.get('intermediate_frames'),
            dvs_update_calls=reply.get('dvs_update_calls'),
            dvs_response_config_sha256=reply.get('dvs_response_config_sha256')))''')
 patched['sparse_event_sensor.py']=sensor
 runner=original['run_original.py']
 runner=replace_once(runner,"    launch_override=json.loads(args.launch_config.resolve().read_text()) if args.launch_config else None",'''    launch_override=json.loads(args.launch_config.resolve().read_text()) if args.launch_config else None
    event_reconstruction_mode=os.environ.get('CATCHIT_EVENT_RECONSTRUCTION_MODE','full')
    if event_reconstruction_mode not in ('full','endpoint_only'):
        parser.error('Unregistered event reconstruction mode')
    if event_reconstruction_mode=='endpoint_only' and os.environ.get('CATCHIT_EVENT_GENERATOR','ours')!='ours':
        parser.error('Endpoint-only ablation applies only to the frozen Ours response')''')
 runner=replace_once(runner,"    manifest=dict(event_generator=os.environ.get('CATCHIT_EVENT_GENERATOR','ours'),event_updates_disabled=",
  "    manifest=dict(event_generator=os.environ.get('CATCHIT_EVENT_GENERATOR','ours'),event_reconstruction_mode=event_reconstruction_mode,event_updates_disabled=")
 patched['run_original.py']=runner
 output.mkdir(parents=True,exist_ok=False)
 differences=[]
 for name,value in patched.items():
  (output/name).write_text(value,encoding='utf-8',newline='\n')
  differences.extend(difflib.unified_diff(original[name].splitlines(keepends=True),value.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
 shutil.copyfile(here/'endpoint_event_worker.py',output/'endpoint_event_worker.py')
 (output/'source_hook.patch').write_text(''.join(differences),encoding='utf-8',newline='\n')
 receipt=dict(status='PREPARED_ONLY',original_code_commit=metadata['code_commit'],original_source=str(source),
  original_sha256={name:sha(source/name) for name in original},
  overlay_sha256={name:sha(output/name) for name in (*patched,'endpoint_event_worker.py')},
  core_modified=False,default_mode='full',ablation_mode='endpoint_only',
  selector_environment='CATCHIT_EVENT_RECONSTRUCTION_MODE',actual_runs=0)
 (output/'SOURCE_OVERLAY.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 return receipt


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--original-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();print(json.dumps(prepare(args.original_source,args.output)))
