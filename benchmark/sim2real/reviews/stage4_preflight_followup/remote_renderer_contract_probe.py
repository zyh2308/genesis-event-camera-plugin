from pathlib import Path
import json
root=Path('/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2/genesis')
out=[]
for name in ['ext/pyrender/renderer.py','ext/pyrender/jit_render.py','ext/pyrender/constants.py','vis/rasterizer.py','vis/camera.py']:
 p=root/name;l=p.read_text().splitlines()
 terms=['glRenderbufferStorageMultisample','GL_DEPTH_BUFFER_BIT','ANTIALIAS','antialiasing','POLYGON_OFFSET','glPolygonOffset','glBlitFramebuffer','GL_MULTISAMPLE','def _configure_main_framebuffer']
 for n,line in enumerate(l):
  if any(t in line for t in terms):out.append({'file':str(p),'line':n+1,'text':'\n'.join(str(i+1)+': '+l[i] for i in range(max(0,n-4),min(len(l),n+12)))})
print(json.dumps(out))
