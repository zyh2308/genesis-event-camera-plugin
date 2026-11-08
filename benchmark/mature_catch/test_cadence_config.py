from cadence_config import endpoint_stride,minimum_v2e_upsampling
for hz,stride,up in [(100,5,10),(50,10,20),(25,20,40)]:
    assert endpoint_stride(hz,.002)==stride
    assert minimum_v2e_upsampling(hz)==up
    captures=[i*.002 for i in range(1,501) if i%stride==0]
    assert len(captures)==hz and abs(captures[-1]-1)<1e-12
    assert abs((1/hz)/up-.001)<1e-12
for hz in [0,-1,75,float('nan')]:
    try:endpoint_stride(hz,.002)
    except ValueError:pass
    else:raise AssertionError(hz)
print('PASS: identical physics step, exact capture ticks, fixed V2E1ms subframe budget across100/50/25Hz')
