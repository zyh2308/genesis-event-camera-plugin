import json
from pathlib import Path
path = Path('/home/zhaoyuhan/Eventbased_WAM/.gpu_reservations/r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z.json')
with path.open(encoding='utf-8') as stream:
    value = json.load(stream)
value['active_attempt_output'] = '/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/genesis_event_camera_plugin/r1_5_localization_recovery_20261101a/strict_direct_pair_r1_5_localization_recovery_20261101b_gpu0_20261101T073805Z/attempt_03_robot_env_hdr'
value['active_code_commit'] = '03c5b0a548bf5c71e49cf7f340a8ebfe48ba42b5'
value['runner_sha256'] = '457e94c1705ac116f8016819b9baefc25b24a3632f2b2399111406ac8e5aee2b'
value['attempt'] = 3
with path.open('w', encoding='utf-8') as stream:
    json.dump(value, stream, indent=2)
    stream.write('\n')
