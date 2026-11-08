"""Serialization-only wrapper around immutable v2 CPU data audit."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

V2_SHA='ee07d66e459cad74b73520002e95f2ad7b4c353710426ffc3fe5efde46a3c06c'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def numpy_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def json_dump(value,stream,**kwargs):
    # Preserve all original options, notably allow_nan=False and integer precision.
    assert 'default' not in kwargs
    return json.dump(value,stream,default=numpy_default,**kwargs)


def json_proxy(provenance=None):
    def dump(value,stream,**kwargs):
        if provenance is not None:
            assert isinstance(value,dict) and 'serialization_provenance' not in value
            value=dict(value,serialization_provenance=provenance)
        return json_dump(value,stream,**kwargs)
    return SimpleNamespace(load=json.load,loads=json.loads,dump=dump,dumps=json.dumps)


def load_v2():
    path=Path(__file__).resolve().with_name('audit_pose_data_contract_v2.py')
    assert sha(path)==V2_SHA
    spec=importlib.util.spec_from_file_location('preserved_pose_audit_v2',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--preserved-v2-partial',type=Path,required=True)
    parser.add_argument('--preserved-v2-partial-sha256',required=True)
    args,rest=parser.parse_known_args()
    assert args.preserved_v2_partial.is_file() and sha(args.preserved_v2_partial)==args.preserved_v2_partial_sha256
    module,v2=load_v2();own=Path(__file__).resolve().parent
    manifest=own/'source_manifest.json'
    for name,digest in json.loads(manifest.read_text()).items():
        assert Path(name).name==name and sha(own/name)==digest
    provenance=dict(wrapper_sha256=sha(Path(__file__)),source_manifest_sha256=sha(manifest),
        original_v2_script_path=str(v2),original_v2_script_sha256=V2_SHA,
        preserved_failure_partial=dict(path=str(args.preserved_v2_partial),sha256=args.preserved_v2_partial_sha256,size_bytes=args.preserved_v2_partial.stat().st_size),
        change='numpy.generic.item / ndarray.tolist at JSON boundary only; all original statistics unchanged',
        allow_nan=False,preserved_original_failure=True)
    module.json=json_proxy(provenance)
    previous=sys.argv
    try:
        sys.argv=[str(v2),*rest];module.main()
        assert sha(args.preserved_v2_partial)==args.preserved_v2_partial_sha256
    finally:sys.argv=previous


if __name__=='__main__':main()
