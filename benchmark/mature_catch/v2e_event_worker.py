"""Task-only V2E worker; accepts LDR grayscale and endpoint time, no scene GT."""
import hashlib,json,os,pickle,sys,time
from pathlib import Path

def main():
    wire=sys.stdout.buffer;sys.stdout=sys.stderr
    from event_wire import encode,decode
    from streaming_v2e import StreamingV2E
    proof=Path(os.environ['CATCHIT_V2E_PARITY_RECEIPT']);parity=json.loads(proof.read_text())
    assert parity['status']=='PASS'
    assert hashlib.sha256(Path(__file__).with_name('streaming_v2e.py').read_bytes()).hexdigest()==parity['adapter_sha256']
    source=Path(os.environ['CATCHIT_V2E_SOURCE']);checkpoint=source/'input/SuperSloMo39.ckpt'
    from cadence_config import minimum_v2e_upsampling
    minimum=minimum_v2e_upsampling(float(os.environ.get('CATCHIT_EVENT_ENDPOINT_HZ','100')))
    assert minimum==parity['minimum_upsampling'],(minimum,parity)
    generator=StreamingV2E(source,checkpoint,minimum_upsampling=minimum)
    receipt=generator.receipt()
    assert receipt['checkpoint_sha256']==parity['checkpoint_sha256']
    assert receipt['source_sha256']==parity['source_sha256']
    def reply(x):pickle.dump(encode(x),wire,protocol=5);wire.flush()
    reply({'ready':True,'method':'V2E streaming native SuperSloMo plus native EventEmulator','receipt':receipt,'parity_receipt':str(proof),'privileged_scene_inputs':False})
    access=0
    while True:
        try:request=decode(pickle.load(sys.stdin.buffer))
        except EOFError:break
        if request['kind']=='close':break
        packet=request['packet'];assert set(packet)=={'time_s','gray8'},set(packet)
        start=time.monotonic();events,info=generator.push(packet['gray8'],float(packet['time_s']));access+=1
        reply({'events':events,'packet_end_s':float(packet['time_s']),'endpoint_reads':access,'dense_HDR_reads':0,'renderer_calls_in_reconstruction':0,'wall_s':time.monotonic()-start,**info})

if __name__=='__main__':main()
