import datetime,hashlib,json,pathlib,shutil
p=pathlib.Path(__file__).parent
repo=pathlib.Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105')
out=repo/'benchmark/sim2real/reviews/recon_activation_budget_20261108'
d=json.loads((p/'result_summary.json').read_text())
stamp=datetime.datetime.fromisoformat(d['captured_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')
lines=['',f'完成核验：{stamp}。','',
    '**结论：不能把HQF退步简单归咎于sigmoid；同预算下它降低感知误差，但牺牲部分光度/细节指标。训练进度与输出激活存在交互，不是已确定的二选一原因。**','',
    '| 合成600步、seed0 | 线性 LPIPS-Alex ↓ | sigmoid LPIPS-Alex ↓ | 感知误差下降 | 线性→sigmoid PSNR/dB ↑ | MSE增加 |','|---|---:|---:|---:|---|---:|']
for m,w in d['workers'].items():
    a={r['arm']:r for r in w['completion']['arms']};l=a['original']['validation']['600']['all200'];s=a['sigmoid']['validation']['600']['all200']
    lines.append(f"| {m} | {l['lpips_alex']:.5f} | {s['lpips_alex']:.5f} | {100*(1-s['lpips_alex']/l['lpips_alex']):.2f}% | {l['psnr_db']:.3f} → {s['psnr_db']:.3f} | {100*(s['mse']/l['mse']-1):.1f}% |")
lines+=['',
    '两组sigmoid都没有大面积饱和：600步平均导数约0.247/0.246，输出落在0.02以下或0.98以上的比例均为0%。因此没有“饱和压死梯度”的证据；导数上限0.25本身也不是饱和证据。','',
    '训练到600步比300步降低了四臂的感知误差，但线性臂追赶更快，sigmoid的感知优势缩小，不能据600步外推最终胜负。原连续sigmoid日志3900→5100仍有下降和波动；4,800步是已选VGG最优点，其后单点回升不能证明训练已到平台。','',
    '细节影响并非所有方法一致：EventGenesis的SSIM与边缘幅值降低，V2E的边缘幅值却略升。更明确且一致的是两组sigmoid的PSNR降低约1dB、MSE增加约23%/25%。这反映感知指标与逐像素光度误差的取舍，不足以证明永久容量上限，也不能解释全部真实域差距。','',
    '本轮已结束，不启动长训、不重测HQF，也不按真实分数选择新的检查点。','']
f=out/'README.md';s=f.read_text();pos=s.index('\n\n');f.write_text(s[:pos]+'\n'+'\n'.join(lines)+s[pos:])
for base in [p/'analysis',out/'analysis']:
    f=base/'README.md';s=f.read_text().replace('sigmoid在[0,1]内原则上能表示目标亮度','sigmoid原则上可以逼近[0,1]内的目标亮度')
    f.write_text(s.rstrip()+'\n')
    index_file=base/'artifact_index.json';idx=json.loads(index_file.read_text());idx['files']={str(f.relative_to(base)):dict(size_bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(base.rglob('*')) if f.is_file() and f.name!='artifact_index.json'}
    index_file.write_text(json.dumps(idx,indent=2,ensure_ascii=False)+'\n')
shutil.copyfile(p/'finalize_report.py',out/'sources/finalize_report.py')
for f in out.rglob('*.md'):f.write_text(f.read_text().rstrip()+'\n')
index={str(f.relative_to(out)):dict(size_bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact_index.json'}
(out/'artifact_index.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=index),indent=2)+'\n')
for base in [out,out/'analysis']:
    idx=json.loads((base/'artifact_index.json').read_text())
    assert all((base/n).stat().st_size==r['size_bytes'] and hashlib.sha256((base/n).read_bytes()).hexdigest()==r['sha256'] for n,r in idx['files'].items())
print('Verified both artifact indexes;',len(index),'files')
