"""Export light receipts; large arrays/traces remain in the isolated g1 run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    out = root/'delivery_evidence'
    out.mkdir(exist_ok=False)
    copied, external = [], []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or out in path.parents:
            continue
        relative = path.relative_to(root)
        if 'candidate_release' in relative.parts or path.suffix in {'.bundle','.o','.so'} or any(part.startswith(('build_', 'private_build')) for part in relative.parts[:-1]):
            continue
        if path.suffix in {'.npy','.npz','.h5'} or (path.name=='trace.json' and 'profiles' in relative.parts) or (path.name=='trace.json' and relative.parts[:2]==('zbuf_reuse','profile')):
            external.append({'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path)})
        elif path.suffix in {'.json','.md'} or path.name.endswith('.attempt1'):
            target=out/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
            copied.append({'path':str(relative),'sha256':sha(target),'size_bytes':target.stat().st_size})
        elif path.suffix=='.log':
            target=out/(str(relative)+'.gz')
            target.parent.mkdir(parents=True,exist_ok=True)
            with path.open('rb') as src,gzip.open(target,'wb') as dst:shutil.copyfileobj(src,dst)
            copied.append({'path':str(relative)+'.gz','sha256':sha(target),'size_bytes':target.stat().st_size,'uncompressed_source':str(path),'uncompressed_sha256':sha(path)})
    builds=[]
    for path in sorted(root.glob('*/build.ninja')):
        source=path.read_text()
        builds.append({'path':str(path),'sha256':sha(path),'fmad_disabled':'--fmad=false' in source,'fast_math':'--use_fast_math' in source,'sm70':'code=sm_70' in source})
        if '--fmad=false' not in source or '--use_fast_math' in source or 'code=sm_70' not in source:
            raise RuntimeError('Compiler math contract mismatch')
        target=out/'build_receipts'/(path.parent.name+'.ninja.txt')
        target.parent.mkdir(exist_ok=True)
        target.write_text(source)
        copied.append({'path':str(target.relative_to(out)),'sha256':sha(target),'size_bytes':target.stat().st_size})
    manifest={'schema':'plugin_efficiency_continuation_delivery_v1','run_root':str(root),'implementation_head_at_export':subprocess.check_output(['git','-C',str(args.repo),'rev-parse','HEAD'],text=True).strip(),'copied_evidence':copied,'external_artifacts':external,'builds':builds,'formal_training_modified':False,'production_data_replaced':False,'default_cuda_strict_qualification':'inherited_failed; no tolerance or timestamp quantization promotion'}
    for label,core in [('projection_storage',root/'candidate_release'),('zbuf_reuse',root/'zbuf_reuse/candidate_release')]:
        if (core/'SOURCE_MANIFEST.json').exists():
            target=out/(label+'_source_manifest.json')
            shutil.copyfile(core/'SOURCE_MANIFEST.json',target)
            copied.append({'path':str(target.relative_to(out)),'sha256':sha(target),'size_bytes':target.stat().st_size})
    (out/'external_artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
    checksums=[{'path':str(p.relative_to(out)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'CHECKSUMS.json').write_text(json.dumps({'files':checksums,'scope':'all other files in this evidence directory; checksum file excludes itself'},indent=2)+'\n')
    with tarfile.open(root/'delivery_evidence.tar.gz','w:gz') as archive:
        archive.add(out,arcname='evidence')
    print('EVIDENCE_ARCHIVE',str(root/'delivery_evidence.tar.gz'),(root/'delivery_evidence.tar.gz').stat().st_size,flush=True)


if __name__=='__main__':
    main()
