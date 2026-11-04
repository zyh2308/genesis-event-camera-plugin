"""Explicit device entry point while preserving the NumPy interpolation API."""
import inspect,textwrap
import torch
import resident2

def preserve_public_interpolation(interpolator):
    gpu_interpolate=interpolator.interpolate
    def device(*args,**kwargs):
        kwargs.pop('return_torch',None)
        return gpu_interpolate(*args,return_torch=False,**kwargs)
    def public(*args,**kwargs):
        requested=kwargs.pop('return_torch',False)
        if requested:return gpu_interpolate(*args,return_torch=True,**kwargs)
        frames,times=device(*args,**kwargs)
        return list(torch.stack(frames).cpu().numpy()),times
    interpolator.interpolate_device=device
    interpolator.interpolate=public

def make_plugin_class(base,mode='exact'):
    import genesis_event_plugin.physics_plugin as module
    Parent=resident2.make_plugin_class(base,mode)
    source=textwrap.dedent(inspect.getsource(base.capture))
    old='chunks = [\n        self.dvs_emulator.generate_events(frame, ts)\n        for frame, ts in zip(sensor_frames, timestamps)\n    ]'
    assert source.count(old)==1
    source=source.replace(old,'chunks = self._generate_device_sequence(sensor_frames,timestamps)')
    source=source.replace('self.interpolator.interpolate(','self.interpolator.interpolate_device(',1)
    namespace=dict(module.__dict__);exec(source,namespace)
    class Resident(Parent):
        def attach(self,*args,**kwargs):
            result=super().attach(*args,**kwargs);preserve_public_interpolation(self.interpolator);return result
    Resident.capture=namespace['capture'];return Resident

sensor_device=resident2.sensor_device
finish_geometry=resident2.finish_geometry
DeferredDiagnostics=resident2.DeferredDiagnostics
DeferredList=resident2.DeferredList
DeviceRenderRequest=resident2.DeviceRenderRequest
def install_geometry(interpolator,mode):
    resident2.install_geometry(interpolator,mode);preserve_public_interpolation(interpolator)
