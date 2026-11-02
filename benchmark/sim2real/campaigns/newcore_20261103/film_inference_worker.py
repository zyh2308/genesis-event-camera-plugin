"""TensorFlow-only process: official FILM Style, three midpoint subdivisions."""
import pickle,re,struct,subprocess,sys
from pathlib import Path
import numpy as np,tensorflow as tf
R=Path(sys.argv[1]);devices=tf.config.list_physical_devices('GPU');assert len(devices)==1
assembler=subprocess.check_output(['ptxas','--version'],text=True);version=re.search(r'release\s+(\d+)\.(\d+)',assembler);assert version and tuple(map(int,version.groups()))>=(11,1),assembler
tf.config.set_logical_device_configuration(devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
model=tf.saved_model.load(str(R/'baseline_assets/film/pretrained_models/film_net/Style/saved_model'))
def send(value):
 raw=pickle.dumps(value,protocol=5);sys.stdout.buffer.write(struct.pack('<Q',len(raw)));sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
def receive():
 size=sys.stdin.buffer.read(8)
 if not size:return None
 assert len(size)==8;n=struct.unpack('<Q',size)[0];assert n<16*1024*1024;raw=sys.stdin.buffer.read(n);assert len(raw)==n;return pickle.loads(raw)
def midpoint(x0,x1):return model({'x0':x0,'x1':x1,'time':np.array([[.5]],np.float32)},training=False)['image'].numpy()
send({'ready':True,'tensorflow_version':tf.__version__,'interpolation_factor':8,'memory_limit_MiB':6144,'ptxas_version':assembler})
while True:
 pair=receive()
 if pair is None:break
 left,right=(np.frombuffer(blob,dtype=np.uint8).reshape(260,346) for blob in pair)
 H,W=left.shape;pad=((0,0),(0,(-H)%64),(0,(-W)%64),(0,0))
 x0=np.pad(np.repeat(left[None,:,:,None].astype(np.float32)/255,3,3),pad);x1=np.pad(np.repeat(right[None,:,:,None].astype(np.float32)/255,3,3),pad)
 def recurse(p,q,depth):
  m=midpoint(p,q)
  if depth>1:yield from recurse(p,m,depth-1)
  yield m[0,:H,:W]
  if depth>1:yield from recurse(m,q,depth-1)
 frames=np.stack(list(recurse(x0,x1,3))).astype(np.float32);assert frames.shape==(7,H,W,3) and np.isfinite(frames).all();send(frames.tobytes())
