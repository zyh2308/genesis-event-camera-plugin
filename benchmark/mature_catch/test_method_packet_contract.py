"""Guard the native V2E input boundary and the common causal availability delay."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,io,json,pickle,tempfile
from unittest.mock import patch
import h5py,numpy as np

spec=importlib.util.spec_from_file_location('sensor_under_test',Path(__file__).with_name('sparse_event_sensor.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from event_wire import encode,decode

class Camera:
    def __init__(self):self.transform=np.eye(4);self.res=(640,480);self.calls=[]
    def render(self,**kwargs):
        self.calls.append(kwargs)
        if kwargs['radiance']:
            return np.ones((480,640,3),np.float32),np.ones((480,640),np.float32),np.ones((480,640),np.int32),None
        return np.full((480,640,3),91,np.uint8),None,None,None

def verify(method):
    with tempfile.TemporaryDirectory() as temp:
        camera=Camera();pushed=[];sensor=module.SparseEventSensor.__new__(module.SparseEventSensor)
        sensor.env=SimpleNamespace(update_sensor_pose=lambda:None,robot=object(),background_wall=None)
        sensor.camera=camera;sensor.root=Path(temp);sensor.K=np.eye(3);sensor.endpoint_hz=100.;sensor.generator_method=method
        sensor.initialization_only=sensor.initialization_probe=sensor.initialization_failed=False
        sensor.tracker=SimpleNamespace(attached_offset=None,push=lambda *args,**kw:pushed.append((args,kw)))
        sensor.chunks=[];sensor.packet_audit=[];sensor.frame_index=0
        reply=io.BytesIO();pickle.dump(encode(dict(events=np.array([[.02,5,6,1.]]),wall_s=0.,endpoint_reads=1,subframes=10)),reply);reply.seek(0)
        sensor.worker=SimpleNamespace(stdin=io.BytesIO(),stdout=reply)
        sensor.h=h5py.File(Path(temp)/'endpoints.h5','w')
        with patch.object(module,'link_transforms',lambda robot:{1:np.eye(4)}),patch.dict('os.environ',{'CATCHIT_STOP_EVENTS_AFTER_GRASP':'0'}):sensor.capture(.02)
        sensor.worker.stdin.seek(0);request=decode(pickle.load(sensor.worker.stdin));packet=request['packet']
        assert abs(pushed[0][1]['available_s']-.03)<1e-12
        assert sensor.packet_audit[0]['dense_HDR_reads']==0
        if method=='v2e':
            assert set(packet)=={'time_s','gray8'}
            assert packet['gray8'].dtype==np.uint8 and packet['gray8'].shape==(480,640)
            assert not any(k in packet for k in ['radiance','depth','segmentation','object_poses','camera_to_world_cv'])
            assert len(camera.calls)==2 and sensor.h['native_ldr_gray8'].dtype==np.dtype('uint8')
        else:
            assert 'object_poses' in packet and 'radiance' in packet and 'gray8' not in packet
            assert len(camera.calls)==1 and 'native_ldr_gray8' not in sensor.h
        sensor.h.close()

if __name__=='__main__':verify('ours');verify('v2e');print('PASS: V2E receives only native gray/time; Ours keeps frozen inputs; both delay packets by10ms')
