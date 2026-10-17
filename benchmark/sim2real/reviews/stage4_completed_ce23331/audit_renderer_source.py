from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
source='''from pathlib import Path
import json
root=Path('/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2/genesis')
hits=[]
for p in root.rglob('*.py'):
    if not any(s in str(p) for s in ['vis','render','camera']):continue
    lines=p.read_text(errors='replace').splitlines()
    for i,line in enumerate(lines):
        if any(term in line for term in ['seg_idx','segmentation_idx','link_idx + 1','link.idx + 1','link_idx+1','link.idx+1','seg_node_map']):
            hits.append({'file':str(p),'line':i+1,'context':'\\n'.join(lines[max(0,i-3):min(len(lines),i+4)])})
excerpts=[]
paths=[(root/'vis/rasterizer_context.py',[(1,33),(420,460)]),(root/'vis/visualizer.py',[(315,323)]),(Path('/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/scripts/generate_stage4_flow_bank.py'),[(165,200),(245,265)]),(Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/STE-FlowNet/multiscaleloss.py'),[(19,57)])]
for p,ranges in paths:
    lines=p.read_text().splitlines()
    for a,b in ranges:excerpts.append({'file':str(p),'start':a,'text':'\\n'.join(str(i+1)+': '+lines[i] for i in range(a-1,min(b,len(lines))))})
print(json.dumps({'root':str(root),'exists':root.exists(),'hits':hits,'excerpts':excerpts},ensure_ascii=False))
'''
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode(source.encode()).decode(),capture_output=True,text=True,timeout=90)
(O/'renderer_source_verification.json').write_text(r.stdout,encoding='utf-8')
if r.returncode:raise RuntimeError(r.stderr)
d=json.loads(r.stdout)
print(json.dumps({'excerpts':d['excerpts'],'creation_hits':[h for h in d['hits'] if 'create_node_seg(' in h['context']]},ensure_ascii=False))
