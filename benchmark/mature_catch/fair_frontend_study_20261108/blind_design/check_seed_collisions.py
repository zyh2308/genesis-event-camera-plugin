"""Read-only seed-token audit of prior metadata; exclude this new task's own files.

This checks existing JSON/CSV/JSONL metadata, not outcomes or physical feasibility.
Run on the shared server run roots before launch to complete the global audit.
"""
from pathlib import Path
import argparse,datetime,json,os,re

HERE=Path(__file__).resolve().parent


def inside(path,parent):
    try:path.relative_to(parent);return True
    except ValueError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,default=HERE/'BLIND32_BANK.json');p.add_argument('--root',type=Path,action='append',required=True);p.add_argument('--exclude',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    bank=json.loads(a.bank.read_text(encoding='utf-8-sig'));seeds={str(c['seed']) for c in bank['cases']}
    pattern=re.compile(r'(?<![A-Za-z0-9_])(?:'+'|'.join(sorted(seeds))+r')(?![A-Za-z0-9_])')
    exclude=[x.resolve() for x in a.exclude];matches=[];errors=[];files=0
    for supplied in a.root:
        root=supplied.resolve();assert root.is_dir(),root
        for directory,dirs,names in os.walk(root,followlinks=False):
            directory=Path(directory);dirs[:]=[d for d in dirs if not any(inside((directory/d).resolve(),e) for e in exclude)]
            if any(inside(directory.resolve(),e) for e in exclude):continue
            for name in names:
                f=directory/name
                if f.suffix.lower() not in ['.json','.csv','.jsonl']:continue
                files+=1;found=set()
                try:
                    with f.open('r',encoding='utf-8-sig',errors='replace') as stream:
                        tail=''
                        for chunk in iter(lambda:stream.read(65536),''):
                            data=tail+chunk;found.update(pattern.findall(data));tail=data[-64:]
                except OSError as error:errors.append(dict(path=str(f),error=str(error)))
                if found:matches.append(dict(path=str(f),seed_tokens=sorted(found)))
    result=dict(status='PASS' if not matches and not errors else 'FAIL',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        roots=[str(x.resolve()) for x in a.root],excludes=[str(x) for x in exclude],metadata_files_checked=files,
        seed_count=len(seeds),seed_range=[min(map(int,seeds)),max(map(int,seeds))],matches=matches,read_errors=errors,
        scope='Exact seed tokens in JSON/CSV/JSONL metadata only. This is not a claim about unregistered runs or inaccessible roots. No simulation or outcomes inspected.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False));raise SystemExit(0 if result['status']=='PASS' else 2)


if __name__=='__main__':main()
