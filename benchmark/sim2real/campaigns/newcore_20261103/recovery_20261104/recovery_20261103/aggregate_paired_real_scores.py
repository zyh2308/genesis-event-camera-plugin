import argparse,csv,math,random,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('scores',type=Path,help='CSV: task,metric,method,seed,sequence,value')
p.add_argument('output',type=Path)
a=p.parse_args()
methods=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
scores=defaultdict(dict)
with a.scores.open(newline='') as f:
    for row in csv.DictReader(f):
        group=row['task'],row['metric']
        key=row['method'],int(row['seed']),row['sequence']
        value=float(row['value'])
        assert key[0] in methods and key[1] in (0,1,2) and key[2] and math.isfinite(value)
        assert key not in scores[group],key
        scores[group][key]=value
assert scores
rows=[]
for (task,metric),values in sorted(scores.items()):
    sequences=sorted({key[2] for key in values})
    assert len(sequences)>1
    assert set(values)=={(m,s,q) for m in methods for s in range(3) for q in sequences}
    for method in methods:
        seed_means=[statistics.mean(values[method,s,q] for q in sequences) for s in range(3)]
        deltas=[statistics.mean(values[method,s,q]-values['eventgenesis',s,q] for s in range(3)) for q in sequences]
        rng=random.Random(20261103)
        samples=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(10000))
        rows.append([task,metric,method,len(sequences),*seed_means,statistics.mean(seed_means),statistics.stdev(seed_means),statistics.mean(deltas),samples[249],samples[9749]])
with a.output.open('x',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['task','metric','method','sequence_count','seed0','seed1','seed2','mean','sample_sd','method_minus_eventgenesis','sequence_bootstrap_ci95_low','sequence_bootstrap_ci95_high'])
    writer.writerows(rows)
