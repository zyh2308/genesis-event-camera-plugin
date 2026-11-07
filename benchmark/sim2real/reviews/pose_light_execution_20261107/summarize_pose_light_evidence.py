"""Verify an exported evidence archive and write compact research tables."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.archive) as z:
        index=json.loads(z.read('INDEX.json'))
        for name,record in index.items():
            payload=z.read(name)
            assert len(payload)==record['bytes']
            assert hashlib.sha256(payload).hexdigest()==record['sha256']
        summary=json.loads(z.read('SUMMARY.json'))
        table=[]
        for study,record in summary['studies'].items():
            for key,value in record.get('aggregate',{}).items():
                method,arm,condition=key.split('/')
                table.append(dict(study=study,method=method,arm=arm,condition=condition,
                                  seeds=len(value['success']['values']),depth_mae_cm=100*value['depth_mae_m']['mean'],
                                  depth_bias_cm=100*value['depth_bias_m']['mean'],
                                  center_px=value['center_px']['mean'],success_pct=100*value['success']['mean'],
                                  auc_pct=100*value['auc']['mean']))
        with (args.out/'synthetic_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]))
            writer.writeheader();writer.writerows(table)
        verification=dict(archive=args.archive.name,archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                          checked_files=len(index),captured_utc=summary['captured_utc'],status='passed')
        (args.out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(verification))


if __name__=='__main__':main()
