"""Public reset-relative scene illumination; no target state, image or randomness."""
import math

FRONT_GLARE_START_S=.10
FRONT_GLARE_END_S=.20
FRONT_GLARE_DEFAULT_PEAK=400.
FRONT_GLARE_POINT_POSITION=(0.,-.5,2.5)

def validated_peak(value):
    value=float(value)
    if not math.isfinite(value) or value<0:
        raise ValueError('front_glare peak must be finite and nonnegative')
    return value

def front_glare_intensity(time_s,peak=FRONT_GLARE_DEFAULT_PEAK):
    time_s=float(time_s);peak=validated_peak(peak)
    if not math.isfinite(time_s) or time_s<0:
        raise ValueError('lighting time must be finite, nonnegative reset-relative simulation time')
    fraction=max(0.,min(1.,(time_s-FRONT_GLARE_START_S)/(FRONT_GLARE_END_S-FRONT_GLARE_START_S)))
    return peak*fraction

def front_glare_schedule(peak=FRONT_GLARE_DEFAULT_PEAK):
    peak=validated_peak(peak)
    return dict(schema='front_glare_public_time_v1',time_source='reset-relative _gs_time only; no object state or release flags',
                initial_point_intensity=0.,ramp_start_s=FRONT_GLARE_START_S,ramp_end_s=FRONT_GLARE_END_S,
                peak_point_intensity_renderer_units=peak,point_position=list(FRONT_GLARE_POINT_POSITION),
                point_color=[1.,1.,1.],normal_ambient_light=[.1,.1,.1],
                normal_directional_direction=[-1.,-1.,-1.],normal_directional_intensity=5.,
                law='zero until ramp_start, linear ramp to ramp_end, then constant peak',
                initialization='point intensity0 at time0; actual visual frame budget remains declared in the actor protocol',
                frame_processing='native renderer only; no postprocess exposure or image edits',
                calibration='renderer settings; not absolute illuminance or real-device calibration')
