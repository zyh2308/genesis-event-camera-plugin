import os,sys,json
sys.path.insert(0,'/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2')
import genesis as gs
gs.init(backend=gs.cpu,logging_level='debug')
from genesis.ext.pyrender.offscreen import OffscreenRenderer
from genesis.ext.pyrender.platforms import egl
egl._eglQueryDevicesEXT=None
r=OffscreenRenderer(pyopengl_platform='egl')
r.make_current()
from OpenGL.GL import glGetString,GL_RENDERER,GL_VERSION
d={'renderer':glGetString(GL_RENDERER).decode(),'version':glGetString(GL_VERSION).decode(),'software':r._is_software,'FBO_platform':r._platform.supports_framebuffers()};print('GL_PROBE_RESULT',json.dumps(d),flush=True);assert 'llvmpipe' in d['renderer'].lower();r.make_uncurrent();r.delete();gs.destroy()
