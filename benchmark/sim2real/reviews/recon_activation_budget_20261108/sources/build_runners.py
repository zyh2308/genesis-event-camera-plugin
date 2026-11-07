import ast,hashlib,json,pathlib,shutil
p=pathlib.Path(__file__).parent;old=p.parent/'reconstruction_repair_20261107'
shutil.copyfile(old/'recon_short_diagnostic.py',p/'recon_short_diagnostic.py')
s=(old/'run_recon_epoch5_eventgenesis.py').read_text();a=s.index('    specs=[');b=s.index('    completed = []',a)
for method in ['eventgenesis','v2e']:
    specs=f'''    specs=[('smoke','activation_budget_diagnostic.py',['--campaign',str(campaign),'--out',str(root/'smoke'),'--method','{method}','--smoke-only']),
        ('study','activation_budget_diagnostic.py',['--campaign',str(campaign),'--out',str(root/'study'),'--method','{method}'])]
'''
    src=s[:a]+specs+s[b:]
    src=src.replace('time.monotonic()+3*3600','time.monotonic()+2*3600')
    src=src.replace('bounded synthetic reconstruction root-cause probes and paired fixed-batch short training','synthetic-only activation x budget four-arm bounded diagnosis; no formal queue or real scoring')
    ast.parse(src);(p/f'run_activation_{method}.py').write_text(src)
for f in p.glob('*.py'):ast.parse(f.read_text())
previous=json.loads((p.parent/'hqf_repair_20261108/hqf_result_summary.json').read_text())
binding=previous['protocol']['evaluation_sources']['evaluate_synthetic_reconstruction_fixture.py']
assert len(binding['sha256'])==64
(p/'metric_source_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
names=['activation_budget_diagnostic.py','recon_short_diagnostic.py','run_activation_eventgenesis.py','run_activation_v2e.py','metric_source_binding.json']
(p/'source_manifest.json').write_text(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
print('Prepared',names)
