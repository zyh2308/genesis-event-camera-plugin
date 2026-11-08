import ast,base64,gzip,hashlib,json,pathlib,shutil
p=pathlib.Path(__file__).parent;old=p.parent/'recon_activation_budget_20261108'
for name in ['recon_short_diagnostic.py','activation_budget_diagnostic.py','metric_source_binding.json']:
    shutil.copyfile(old/name,p/name)
for method in ['eventgenesis','v2e']:
    text=(old/f'run_activation_{method}.py').read_text()
    text=text.replace('activation_budget_diagnostic.py','photometry_detail_diagnostic.py')
    text=text.replace("len(completion['results'])==2","len(completion['results'])==3")
    needle="            completed.append(label)"
    barrier=f'''            if label=='smoke':
                counterpart='{('v2e' if method=='eventgenesis' else 'eventgenesis')}'
                sibling=root.parent/root.name.replace('{method}',counterpart)
                wait_until=time.monotonic()+600
                while not (sibling/'smoke/completion.json').exists():
                    assert not (sibling/'failure.json').exists(), 'Paired smoke failed'
                    assert time.monotonic()<wait_until, 'Paired smoke barrier timeout'
                    time.sleep(2)
                other=json.loads((sibling/'smoke/completion.json').read_text())
                assert other['status']=='complete'
                assert [r['update']['labels'] for r in completion['results']]==[r['update']['labels'] for r in other['results']], 'Cross-generator label/augmentation pairing failed'
                save(root/'cross_generator_smoke_pairing.json',dict(passed=True,other=str(sibling),labels=[r['update']['labels'] for r in completion['results']]))
'''
    assert text.count(needle)==1;text=text.replace(needle,barrier+needle)
    text=text.replace('synthetic-only activation x budget four-arm bounded diagnosis; no formal queue or real scoring','synthetic-only matched photometry/detail loss continuation; 300 additional updates per arm; no formal queue or real scoring')
    ast.parse(text);(p/f'run_photometry_{method}.py').write_text(text)
names=['recon_short_diagnostic.py','activation_budget_diagnostic.py','metric_source_binding.json','auxiliary_losses.py','test_auxiliary_losses.py','photometry_detail_diagnostic.py','run_photometry_eventgenesis.py','run_photometry_v2e.py','PROTOCOL.md']
for name in names:
    if name.endswith('.py'):ast.parse((p/name).read_text())
manifest={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}
(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']}
payload=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()
(p/'upload.b64').write_text(payload)
print(json.dumps(dict(bundle_chars=len(payload),files=len(files))))
