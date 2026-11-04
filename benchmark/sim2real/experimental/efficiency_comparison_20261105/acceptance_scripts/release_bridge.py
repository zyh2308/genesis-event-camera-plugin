from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
from genesis_event_plugin.gpu_interpolator import ResidentPhysicsInterpolator,DeferredDiagnostics,DeferredList,DeviceRenderRequest
def make_plugin_class(base,mode):
    assert mode=='exact'
    return OptimizedGenesisPhysicsEventPlugin
def install_geometry(interpolator,mode):
    assert mode=='exact'
    interpolator.__class__=ResidentPhysicsInterpolator
