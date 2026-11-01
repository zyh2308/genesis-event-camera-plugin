"""Output-only decoder; no labels, masks, boxes, or real metadata are inputs."""
import numpy as np
import torch
def decode_classified_pose(output):
 if output.shape[0]!=1 or output.shape[1]!=20:raise ValueError('expected one 20-channel pose prediction')
 if not torch.isfinite(output).all():raise ValueError('nonfinite pose logits')
 _,_,H,W=output.shape;logits=output[0,18];index=int(logits.flatten().argmax());y,x=divmod(index,W)
 confidence=float(logits.flatten().softmax(0)[index]);uv=np.empty((9,2),np.float32)
 for k in range(9):
  a,b=output[0,2*k,y,x],output[0,2*k+1,y,x]
  if k==0:a,b=a.sigmoid(),b.sigmoid()
  uv[k]=[float((a+x)/W),float((b+y)/H)]
 return uv,confidence,(y,x)
