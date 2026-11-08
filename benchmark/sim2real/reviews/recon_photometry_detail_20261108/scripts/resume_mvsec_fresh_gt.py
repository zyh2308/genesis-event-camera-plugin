"""Use the pinned transfer after the mismatched incomplete GT was preserved.

Only the receipt directory changes. The source registration, fixed destination,
full SHA verification, prefix protection and existing SSH identity remain intact.
No download, reprojection, scoring, GPU usage or automatic retries.
"""
import hashlib,sys,types
from pathlib import Path

source=Path(__file__).resolve().with_name('resume_mvsec_outdoor.py')
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='4b42262410087e73acdd6569dfdbf6bd09b76971a00850e9920a2cfe1b2ca05b'
module=types.ModuleType('pinned_mvsec_outdoor_resume')
module.__file__=str(source)
exec(compile(raw,str(source),'exec'),module.__dict__)
module.RECEIPT_ROOT=module.BASE/'MVSEC-EINexus-left-resume-20261108-v2'
if __name__=='__main__':module.main()
