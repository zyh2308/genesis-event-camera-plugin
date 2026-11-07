"""Fetch only named native MVSEC members from an author-hosted split ZIP.

No archive paths are extracted. HTTPS ranges are pinned to immutable HF objects;
entry CRC, native byte size, full SHA and HDF5 semantics are checked before use.
"""
import argparse,datetime,fcntl,hashlib,json,os,shutil,struct,time,urllib.request,zlib,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MEMBERS={'outdoor_day1_data.hdf5':('MVSEC/outdoor_day/outdoor_day1_data.hdf5',12136423279,'data'),
         'outdoor_day1_gt.hdf5':('MVSEC/outdoor_day/outdoor_day1_gt.hdf5',6267453264,'gt')}
ANCHOR='MVSEC/indoor_flying/indoor_flying1_data.hdf5'
PARTS=['MVSEC_RPE.z01','MVSEC_RPE.zip']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):
    t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(path)

def central_entries(b):
    result=[];cursor=0
    while True:
        i=b.find(b'PK\x01\x02',cursor)
        if i<0:break
        assert i+46<=len(b),'Truncated central header'
        h=struct.unpack_from('<4s6H3I5H2I',b,i);nl,xl,cl=h[10:13];end=i+46+nl+xl+cl
        assert end<=len(b),'Truncated central entry'
        name=b[i+46:i+46+nl].decode('utf-8');extra=b[i+46+nl:i+46+nl+xl]
        values=dict(uncompressed=h[9],compressed=h[8],local_offset=h[16],disk=h[13]);k=0
        while k+4<=len(extra):
            tag,length=struct.unpack_from('<HH',extra,k);payload=extra[k+4:k+4+length];k+=4+length
            if tag==1:
                offset=0
                for key,limit,fmt in [('uncompressed',0xffffffff,'Q'),('compressed',0xffffffff,'Q'),('local_offset',0xffffffff,'Q'),('disk',0xffff,'I')]:
                    if values[key]==limit:
                        values[key]=struct.unpack_from('<'+fmt,payload,offset)[0];offset+=struct.calcsize(fmt)
        assert all(r['name']!=name for r in result),'Duplicate archive member'
        result.append(dict(name=name,flags=h[3],method=h[4],crc32=h[7],**values));cursor=end
    return result

class Ranges:
    def __init__(self,manifest,proxy):
        self.manifest=manifest;self.proxy=proxy
        files={x['path']:x for x in manifest['files']};self.sizes=[files[n]['size'] for n in PARTS]
        self.urls=['https://huggingface.co/datasets/'+manifest['dataset']+'/resolve/'+manifest['revision']+'/'+n for n in PARTS]
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':proxy}) if proxy else urllib.request.ProxyHandler())
    def span(self,absolute,length):
        assert absolute>=0 and length>0 and absolute+length<=sum(self.sizes)
        offset=0
        for disk,total in enumerate(self.sizes):
            if absolute<offset+total:
                begin=absolute-offset;n=min(length,total-begin)
                yield disk,begin,n
                absolute+=n;length-=n
                if not length:return
            offset+=total
        raise AssertionError('Invalid split-archive span')
    def read(self,absolute,length):
        parts=[]
        for disk,start,count in self.span(absolute,length):
            request=urllib.request.Request(self.urls[disk],headers={'Range':f'bytes={start}-{start+count-1}','User-Agent':'MVSEC-research-input-audit/1.0'})
            with self.opener.open(request,timeout=90) as response:
                assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{start+count-1}/{self.sizes[disk]}','Range endpoint mismatch'
                assert 'html' not in response.headers.get('Content-Type','').lower()
                b=response.read(count+1);assert len(b)==count,'Incomplete HTTP range'
                parts.append(b)
        return b''.join(parts)
    def absolute(self,entry):
        assert 0<=entry['disk']<2
        return sum(self.sizes[:entry['disk']])+entry['local_offset']
    def content_start(self,entry):
        start=self.absolute(entry);head=self.read(start,30)
        sig,version,flags,method,tm,dt,crc,comp,raw,nl,xl=struct.unpack('<4s5H3I2H',head)
        assert sig==b'PK\x03\x04' and not(flags&1) and flags==entry['flags'] and method==entry['method']
        assert self.read(start+30,nl).decode('utf-8')==entry['name'],'Local/central filename mismatch'
        return start+30+nl+xl

def decode_chunks(chunks,destination,entry):
    assert entry['method'] in (0,8) and not(entry['flags']&1)
    decompressor=zlib.decompressobj(-15) if entry['method']==8 else None
    crc=0;size=0;digest=hashlib.sha256()
    def write(f,b):
        nonlocal crc,size
        size+=len(b);assert size<=entry['uncompressed'],'Archive entry expands past declared native size'
        crc=zlib.crc32(b,crc);digest.update(b);f.write(b)
    with destination.open('xb') as output:
        for chunk in chunks:
            with chunk.open('rb') as stream:
                for b in iter(lambda:stream.read(1024**2),b''):
                    if decompressor:
                        pending=b
                        while pending:
                            write(output,decompressor.decompress(pending,8*1024**2))
                            pending=decompressor.unconsumed_tail
                    else:write(output,b)
        if decompressor:
            write(output,decompressor.flush())
            assert decompressor.eof and not decompressor.unused_data and not decompressor.unconsumed_tail
    assert size==entry['uncompressed'] and crc==entry['crc32'],'Native member size/CRC mismatch'
    return dict(size_bytes=size,crc32=crc,sha256=digest.hexdigest())

def fetch_entry(reader,entry,root,basename,chunk_bytes=64*1024**2):
    assert Path(basename).name==basename and not Path(basename).is_absolute()
    final=root/basename;receipt=root/(basename+'.json')
    if final.exists():
        old=json.loads(receipt.read_text());assert old['status']=='native_member_verified' and sha(final)==old['sha256']
        return old
    cache=root/'chunks'/basename;cache.mkdir(parents=True,exist_ok=True)
    start=reader.content_start(entry);chunks=[]
    for n,offset in enumerate(range(0,entry['compressed'],chunk_bytes)):
        length=min(chunk_bytes,entry['compressed']-offset);part=cache/f'{n:05d}.deflate';rec=cache/f'{n:05d}.json'
        if part.exists():
            recorded=json.loads(rec.read_text());assert part.stat().st_size==length and sha(part)==recorded['sha256']
            assert recorded['absolute_offset']==start+offset
        else:
            payload=reader.read(start+offset,length);temp=cache/f'{n:05d}.{os.getpid()}.download'
            with temp.open('xb') as f:f.write(payload)
            info=dict(absolute_offset=start+offset,size_bytes=length,sha256=hashlib.sha256(payload).hexdigest())
            os.link(temp,part);temp.unlink();save(rec,info)
        chunks.append(part)
        save(root/(basename+'.progress.json'),dict(utc=now(),compressed_bytes_available=offset+length,
            total_compressed_bytes=entry['compressed'],status='fetching_pinned_ranges',real_scores_computed=False))
    partial=root/(basename+'.decode_'+str(time.time_ns())+'.part')
    info=decode_chunks(chunks,partial,entry)
    assert not final.exists();os.link(partial,final);partial.unlink()
    info.update(status='native_member_verified',path=str(final),archive_member=entry,
        dataset=reader.manifest['dataset'],revision=reader.manifest['revision'],real_scores_computed=False,
        note='Member CRC and local SHA verified. Whole multipart archive LFS SHA was not verified because unrelated bytes were not downloaded. No independent official checksum exists for outdoor inputs.')
    save(receipt,info);return info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','out','anchor','calibration-zip','validator-source']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--proxy');args=p.parse_args()
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True);lock=(root/'.download.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert shutil.disk_usage(root).free>32*1024**3
    manifest=json.loads(args.manifest.read_text());reader=Ranges(manifest,args.proxy)
    marker=dict(manifest_sha256=sha(args.manifest),revision=manifest['revision'])
    if (root/'owner.json').exists():assert json.loads((root/'owner.json').read_text())==marker
    else:save(root/'owner.json',marker)
    footer=reader.read(sum(reader.sizes)-65536,65536)
    entries=central_entries(footer);assert entries==manifest['entries'],'Pinned archive central directory differs'
    by={r['name']:r for r in entries}
    assert len(entries)==33
    # Establish byte identity to a registered original indoor file before fetching outdoor members.
    anchor=fetch_entry(reader,by[ANCHOR],root,'anchor_indoor_flying1_data.hdf5')
    assert args.anchor.stat().st_size==anchor['size_bytes'] and sha(args.anchor)==anchor['sha256'],'Mirror differs from registered original anchor'
    print('ORIGINAL_INDOOR_ANCHOR_FULL_SHA_MATCH',anchor['sha256'],flush=True)
    with zipfile.ZipFile(args.calibration_zip) as reference:
        assert reference.testzip() is None
        for name in reference.namelist():
            if name.endswith('/'):continue
            entry=by['MVSEC/outdoor_day_calib/'+name]
            assert entry['compressed']<1024**2 and entry['uncompressed']<2*1024**2
            b=reader.read(reader.content_start(entry),entry['compressed'])
            value=zlib.decompress(b,-15) if entry['method']==8 else b
            assert value==reference.read(name) and zlib.crc32(value)==entry['crc32']
    save(root/'mirror_reference_checks.json',dict(status='passed',anchor_sha256=anchor['sha256'],
        anchor_original_path=str(args.anchor),calibration_zip_sha256=sha(args.calibration_zip),all_five_calibration_members_equal=True))
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_downloader',args.validator_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    validators=module.load_validators(args.validator_source.with_name('pinned_validator.py'))
    def work(name):
        member,total,kind=MEMBERS[name];entry=by[member]
        assert entry['uncompressed']==total
        info=fetch_entry(reader,entry,root,name)
        with Path(info['path']).open('rb') as f:assert f.read(8)==bytes.fromhex('894844460d0a1a0a')
        info['native_validation']=validators['validate_hdf5'](Path(info['path']),kind)
        save(root/(name+'.json'),info);print('NATIVE_MVSEC_INPUT_READY',name,info['size_bytes'],info['sha256'],flush=True)
        return info
    with ThreadPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(work,MEMBERS))
    save(root/'completion.json',dict(status='complete',utc=now(),outputs=outputs,real_scores_computed=False,
        sources='EI-Nexus author-linked HF archive with original indoor full-SHA anchor and byte-identical official calibration'))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import sys,traceback
        if '--out' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--out')+1])
            if out.exists():save(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),utc=now(),trace=traceback.format_exc()))
        raise
