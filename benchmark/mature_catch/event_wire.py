"""NumPy-version-independent array transport between existing environments."""
import numpy as np

def encode(value):
    if isinstance(value,np.ndarray):
        a=np.ascontiguousarray(value)
        return dict(__event_wire_array__=True,dtype=a.dtype.str,shape=a.shape,data=a.tobytes())
    if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(encode(v) for v in value)
    if isinstance(value,list):return [encode(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def decode(value):
    if isinstance(value,dict):
        if value.get('__event_wire_array__') is True:
            return np.frombuffer(value['data'],dtype=np.dtype(value['dtype'])).reshape(value['shape']).copy()
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(decode(v) for v in value)
    if isinstance(value,list):return [decode(v) for v in value]
    return value
