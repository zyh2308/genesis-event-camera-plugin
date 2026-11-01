import os,json,time
from pathlib import Path
import numpy as np
R=Path(os.environ['EPOSE_REPAIR_RUN'])
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform
OSMesaPlatform.supports_framebuffers=lambda self:True
from genesis.ext.pyrender import OffscreenRenderer
from OpenGL import GL as gl
r=OffscreenRenderer(pyopengl_platform='osmesa');r.make_current()
fb=gl.glGenFramebuffers(1);gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,fb)
t=gl.glGenTextures(1);gl.glBindTexture(gl.GL_TEXTURE_2D,t)
gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGBA32F,8,8,0,gl.GL_RGBA,gl.GL_FLOAT,None)
gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_COLOR_ATTACHMENT0,gl.GL_TEXTURE_2D,t,0)
assert gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)==gl.GL_FRAMEBUFFER_COMPLETE
gl.glViewport(0,0,8,8);gl.glClearColor(2.5,.5,.125,1);gl.glClear(gl.GL_COLOR_BUFFER_BIT)
a=np.asarray(gl.glReadPixels(0,0,8,8,gl.GL_RGBA,gl.GL_FLOAT),dtype=np.float32)
assert np.allclose(a.reshape(8,8,4)[0,0],[2.5,.5,.125,1],atol=1e-5)
proof={'GL_renderer':gl.glGetString(gl.GL_RENDERER).decode(),'GL_version':gl.glGetString(gl.GL_VERSION).decode(),'RGBA32F_readback':a.reshape(8,8,4)[0,0].tolist(),'compatibility_override':'process-local OSMesaPlatform.supports_framebuffers=True; float FBO verified; no installed source or frozen core changes'}
(R/'hdr_fbo_probe.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,0);gl.glDeleteTextures([t]);gl.glDeleteFramebuffers(1,[fb]);r.delete()
