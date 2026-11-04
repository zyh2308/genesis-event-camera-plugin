import gc, hashlib, json, sys, time
from pathlib import Path
import torch
root = Path(__file__).resolve().parent
campaign = root.parent
sys.path.insert(0,str(campaign/'sources/benchmark/sim2real/scripts'))
from run_steflow_stage2_pilot import pure_torch_spatial_correlation as ref
from native_adapter import spatial_correlation_sample as native
torch.set_num_threads(2)
torch.manual_seed(20261104)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
report = dict(torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
              reference='frozen memory-bounded PyTorch',upstream_commit='4aa4431bf2f6057c9d40d4990227e3e655699ed3',
              tests=[],benchmarks=[],real_data_used=False)
def save(): (root/'operator_results.json').write_text(json.dumps(report,indent=2)+'\n')
def difference(a,b,atol,rtol):
    diff=(a-b).abs(); denom=max(float(a.abs().max()),1e-30)
    return dict(max_abs=float(diff.max()),scaled_max_abs=float(diff.max())/denom,
                relative_l2=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(a).clamp_min(1e-30)),
                bitwise_equal=bool(torch.equal(a,b)), passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),atol=atol,rtol=rtol)
def test(shape,dtype=torch.float32,kind='random',noncontiguous=False):
    a=torch.randn(shape,device='cuda',dtype=dtype); b=torch.randn_like(a)
    if kind=='zero': a.zero_()
    if kind=='onehot':
        a.zero_();b.zero_();a[:,:,0,0]=1;b[:,:,-1,-1]=1
    if noncontiguous: a=a.transpose(-1,-2);b=b.transpose(-1,-2)
    a.requires_grad_();b.requires_grad_()
    y=ref(a,b,kernel_size=1,patch_size=9,stride=1,padding=0,dilation_patch=1)
    z=native(a,b)
    upstream=torch.randn_like(y).transpose(-1,-2).contiguous().transpose(-1,-2)
    gy=torch.autograd.grad(y,(a,b),upstream)
    gz=torch.autograd.grad(z,(a,b),upstream)
    atol,rtol=(1e-10,1e-10) if dtype==torch.float64 else (3e-4,3e-5)
    row=dict(shape=list(a.shape),dtype=str(dtype),kind=kind,noncontiguous=noncontiguous,
             forward=difference(y,z,atol,rtol),grad1=difference(gy[0],gz[0],atol,rtol),grad2=difference(gy[1],gz[1],atol,rtol))
    report['tests'].append(row);save();print('TEST',row,flush=True)
    assert all(row[k]['passed'] for k in ['forward','grad1','grad2']),row
for s in [(1,1,1,1),(2,7,5,9),(2,33,11,13),(8,64,128,128),(8,128,64,64),(8,256,32,32),(8,512,16,16)]:test(s)
test((2,7,5,9),noncontiguous=True)
test((1,3,7,11),kind='zero');test((1,3,7,11),kind='onehot')
test((1,7,5,6),dtype=torch.float64)
a=torch.randn(1,2,2,3,device='cuda',dtype=torch.float64,requires_grad=True)
b=torch.randn_like(a,requires_grad=True)
report['gradcheck']=bool(torch.autograd.gradcheck(native,(a,b),eps=1e-6,atol=1e-5,rtol=1e-3));save()
assert report['gradcheck']
gc.collect();torch.cuda.empty_cache()
def bench(fn,shape,backward):
    a=torch.randn(shape,device='cuda',requires_grad=backward); b=torch.randn_like(a,requires_grad=backward)
    g=torch.randn(shape[0],9,9,shape[2],shape[3],device='cuda')
    def step():
        y=fn(a,b,kernel_size=1,patch_size=9,stride=1,padding=0,dilation_patch=1)
        if backward:torch.autograd.grad(y,(a,b),g)
    for _ in range(3):step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    t=time.perf_counter();start.record()
    for _ in range(10):step()
    end.record();torch.cuda.synchronize()
    return dict(cuda_ms=start.elapsed_time(end)/10,wall_ms=(time.perf_counter()-t)*100,
                peak_bytes=torch.cuda.max_memory_allocated())
for shape in [(8,64,128,128),(8,128,64,64),(8,256,32,32),(8,512,16,16)]:
    for backward in (False,True):
        r=bench(ref,shape,backward);n=bench(native,shape,backward)
        row=dict(shape=list(shape),forward_backward=backward,reference=r,native=n,
                 wall_speedup=r['wall_ms']/n['wall_ms'],cuda_speedup=r['cuda_ms']/n['cuda_ms'])
        report['benchmarks'].append(row);save();print('BENCH',row,flush=True)
report['passed']=True;save();print('OPERATOR_AUDIT_PASSED',flush=True)
