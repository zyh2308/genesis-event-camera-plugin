import csv,datetime,hashlib,json,sys,time
from pathlib import Path

root=Path(sys.argv[1]).resolve()
methods=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
budgets={'flow':8910,'reconstruction':12000,'pose':30000}
paths=[root/'training'/task/'seed0'/method/'completion.json'
       for task in budgets for method in methods]
destination=root/'FIRST_SEED0_ROUND_COMPLETE.json'
while not all(path.is_file() for path in paths):
    time.sleep(120)
runs=[]
table=[]
for path in paths:
    row=json.loads(path.read_text())
    assert row['status']=='complete'
    assert row['completed_updates']==budgets[row['task']]
    selected=Path(row['selected_checkpoint'])
    assert selected.is_relative_to(path.parent)
    with selected.open('rb') as file:
        digest=hashlib.file_digest(file,'sha256').hexdigest()
    assert digest==row['selected_checkpoint_sha256']
    assert row['real_data_used_in_training'] is False
    status=json.loads((path.parent/'live_status.json').read_text())
    metric_key={'flow':'best_synthetic_validation_epe_px_per_30ms',
                'reconstruction':'best_synthetic_validation_lpips_vgg',
                'pose':'best_metric'}[row['task']]
    metric=status[metric_key]
    table.append([row['task'],row['method'],0,row['completed_updates'],metric_key,metric,digest])
    runs.append(row)
assert len(runs)==15
for task in budgets:
    assert len({row['initial_state_sha256'] for row in runs if row['task']==task})==1
temporary=destination.with_suffix('.tmp')
temporary.write_text(json.dumps({
    'status':'all_five_methods_three_tasks_seed0_training_complete',
    'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'runs':runs,'real_evaluation_completed':False
},indent=2)+'\n')
temporary.replace(destination)
target=root/'FIRST_SEED0_SYNTHETIC_VALIDATION.csv'
with target.open('x',newline='') as file:
    writer=csv.writer(file)
    writer.writerow(['task','method','seed','updates','synthetic_metric','value','selected_checkpoint_sha256'])
    writer.writerows(table)
print('FIRST_SEED0_ROUND_COLLECTED',flush=True)
