"""Causal endpoint adapter for the archived official V2E SuperSloMo pipeline.

The network, normalization, adaptive rule, interpolation equations and uint8
conversion are from SensorsINI/v2e (MIT). The official k/U timestamp convention
is preserved, including its (k+0.5)/U synthesis positions. Validate against that
file API before using this task-only adapter for a comparison.
"""
from pathlib import Path
import atexit
import hashlib
import json
import sys
import numpy as np


class StreamingV2E:
    def __init__(self,source,checkpoint,resolution=(640,480),minimum_upsampling=10):
        self.source=Path(source);self.checkpoint=Path(checkpoint)
        sys.path.insert(0,str(self.source))
        import torch
        from v2ecore.slomo import SuperSloMo
        from v2ecore.emulator import EventEmulator
        self.torch=torch;self.resolution=tuple(resolution)
        assert torch.cuda.is_available()
        self.slomo=SuperSloMo(model=str(self.checkpoint),auto_upsample=True,upsampling_factor=minimum_upsampling,batch_size=1,video_path=None,preview=False)
        # No video writers are opened; skip the unused GUI cleanup on headless nodes.
        atexit.unregister(self.slomo.cleanup)
        dim=tuple(int(x/32)*32 for x in self.resolution)
        self.slomo.flow_estimator,self.slomo.warper,self.slomo.interpolator=self.slomo._SuperSloMo__model(dim)
        self.slomo.model_loaded=True
        self.emulator=EventEmulator(pos_thres=.2,neg_thres=.2,sigma_thres=0.,cutoff_hz=0.,leak_rate_hz=0.,refractory_period_s=0.,shot_noise_rate_hz=0.,photoreceptor_noise=False,seed=23,device='cuda',output_width=resolution[0],output_height=resolution[1],hdr=False)
        self.previous=None;self.previous_time=None;self.intervals=[]

    def pair_frames(self,first,last):
        """Yield official quantized images and their original fractional labels."""
        from PIL import Image
        torch=self.torch;ss=self.slomo
        dim=tuple(int(x/32)*32 for x in self.resolution)
        assert first.shape==last.shape==(self.resolution[1],self.resolution[0])
        with torch.no_grad():
            I0=ss.to_tensor(Image.fromarray(first).resize(dim,Image.LANCZOS)).unsqueeze(0).to(ss.device)
            I1=ss.to_tensor(Image.fromarray(last).resize(dim,Image.LANCZOS)).unsqueeze(0).to(ss.device)
            flowOut=ss.flow_estimator(torch.cat((I0,I1),dim=1));F_0_1=flowOut[:,:2];F_1_0=flowOut[:,2:]
            vFlat=torch.flatten(flowOut,2,3)
            sp0=torch.sqrt(vFlat[:,0,:]*vFlat[:,0,:]+vFlat[:,1,:]*vFlat[:,1,:])
            sp1=torch.sqrt(vFlat[:,2,:]*vFlat[:,2,:]+vFlat[:,3,:]*vFlat[:,3,:])
            maximum=torch.max(torch.max(torch.cat((sp0,sp1),1),dim=1)[0]).cpu().item()
            upsampling=max(2,int(np.ceil(maximum)),ss.upsampling_factor)
            if upsampling>512:raise RuntimeError('Unbounded native optical-flow upsampling; retain and inspect rather than silently cap')
            for k in range(upsampling):
                t=(k+.5)/upsampling;temp=-t*(1-t)
                F_t_0=temp*F_0_1+t*t*F_1_0
                F_t_1=(1-t)*(1-t)*F_0_1+temp*F_1_0
                g_I0_F_t_0=ss.warper(I0,F_t_0);g_I1_F_t_1=ss.warper(I1,F_t_1)
                intrpOut=ss.interpolator(torch.cat((I0,I1,F_0_1,F_1_0,F_t_1,F_t_0,g_I1_F_t_1,g_I0_F_t_0),dim=1))
                F_t_0_f=intrpOut[:,:2]+F_t_0;F_t_1_f=intrpOut[:,2:4]+F_t_1
                V_t_0=torch.sigmoid(intrpOut[:,4:5]);V_t_1=1-V_t_0
                g_I0_F_t_0_f=ss.warper(I0,F_t_0_f);g_I1_F_t_1_f=ss.warper(I1,F_t_1_f)
                Ft=((1-t)*V_t_0*g_I0_F_t_0_f+t*V_t_1*g_I1_F_t_1_f)/((1-t)*V_t_0+t*V_t_1)
                image=ss.to_image(Ft[0].cpu().detach()).resize(self.resolution,Image.BILINEAR)
                yield k/upsampling,np.asarray(image).copy(),upsampling

    def push(self,gray,time_s):
        gray=np.asarray(gray)
        assert gray.dtype==np.uint8 and gray.shape==(self.resolution[1],self.resolution[0])
        if self.previous is None:
            self.previous=gray.copy();self.previous_time=float(time_s)
            return np.empty((0,4)),{'subframes':0,'packet_start_s':float(time_s)}
        assert time_s>self.previous_time
        start=self.previous_time;interval=float(time_s)-start;chunks=[];count=0
        for fraction,frame,upsampling in self.pair_frames(self.previous,gray):
            events=self.emulator.generate_events(frame,start+fraction*interval)
            if events is not None and len(events):chunks.append(events)
            count+=1
        self.previous=gray.copy();self.previous_time=float(time_s)
        result=np.concatenate(chunks) if chunks else np.empty((0,4))
        assert not len(result) or result[:,0].max()<=time_s+1e-8
        info={'subframes':count,'packet_start_s':start,'packet_end_s':float(time_s),'source':'actual renderer LDR grayscale; no depth, segmentation or object pose input','native_fractional_timestamp_convention':'k/U with midpoint synthesis (k+0.5)/U'}
        self.intervals.append(info)
        return result,info

    def receipt(self):
        return {'source':str(self.source),'source_sha256':{str(p.relative_to(self.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*.py') if '__pycache__' not in str(p)},'checkpoint_sha256':hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),'minimum_upsampling':self.slomo.upsampling_factor,'auto_upsample':True,'pixel_model':'native V2E EventEmulator; ideal matched contrast/noise profile, not default nonideal V2E','pos_thres':.2,'neg_thres':.2,'noise':False}
