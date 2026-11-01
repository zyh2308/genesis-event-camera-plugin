from genesis_event_plugin.noise_model import DvsNoiseConfig

def response():
    return DvsNoiseConfig(pos_thres=.2,neg_thres=.2,sigma_thres=0,cutoff_hz=0,
        leak_rate_hz=0,leak_jitter_fraction=0,shot_noise_rate_hz=0,
        noise_rate_cov_decades=0,refractory_period_s=0,log_input=False,
        radiance_input=True,radiance_scale=1,radiance_white_level=1,
        response_mode='physical_log',radiometric_floor=1e-6,
        radiometric_calibration_status='uncalibrated',readout_model='ideal_readout',seed=42)
