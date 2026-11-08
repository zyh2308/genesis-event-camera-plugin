"""Benchmark acquisition cadence; simulator and estimator algorithms stay fixed."""
import math

def endpoint_stride(endpoint_hz,physics_dt_s):
    hz=float(endpoint_hz);dt=float(physics_dt_s)
    if not math.isfinite(hz) or hz<=0 or not math.isfinite(dt) or dt<=0:raise ValueError('Positive finite cadence and physics step required')
    stride=round(1/(hz*dt))
    if stride<1 or abs(stride*dt*hz-1)>1e-9:raise ValueError('Endpoint cadence must align exactly with existing physics ticks')
    return int(stride)

def minimum_v2e_upsampling(endpoint_hz,target_dt_s=.001):
    hz=float(endpoint_hz);target=float(target_dt_s)
    if not math.isfinite(hz) or hz<=0 or not math.isfinite(target) or target<=0:raise ValueError('Positive finite timing required')
    return max(2,int(math.ceil((1/hz)/target-1e-9)))
