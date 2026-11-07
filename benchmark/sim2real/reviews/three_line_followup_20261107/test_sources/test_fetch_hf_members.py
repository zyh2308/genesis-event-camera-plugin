import hashlib,json,struct,tempfile,zlib
from pathlib import Path
from fetch_mvsec_hf_members import Ranges,central_entries,decode_chunks,fetch_entry

checks=[]
reader=Ranges.__new__(Ranges);reader.sizes=[31,49]
assert list(reader.span(29,6))==[(0,29,2),(1,0,4)];checks.append('split disk boundary ranges')
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);raw=(b'\x89HDF\r\n\x1a\n'+bytes(range(256)))*10000
    compressed=zlib.compress(raw)[2:-4];files=[]
    for n,i in enumerate(range(0,len(compressed),271)):
        p=root/f'chunk{n}';p.write_bytes(compressed[i:i+271]);files.append(p)
    entry=dict(method=8,flags=0,uncompressed=len(raw),crc32=zlib.crc32(raw))
    result=decode_chunks(files,root/'decoded',entry)
    assert (root/'decoded').read_bytes()==raw and result['sha256']==hashlib.sha256(raw).hexdigest()
    checks.append('streamed deflate across chunks, full SHA, CRC and native length')
    stored=root/'stored';stored.write_bytes(raw)
    assert decode_chunks([stored],root/'decoded_stored',{**entry,'method':0})['sha256']==result['sha256']
    checks.append('stored ZIP entry')
    for name,params,inputs in [('CRC',{**entry,'crc32':0},files),('truncated',entry,files[:-1]),('encrypted',{**entry,'flags':1},files)]:
        try:decode_chunks(inputs,root/name,params)
        except (AssertionError,zlib.error):checks.append(name+' rejected')
        else:raise AssertionError(name+' accepted')
    try:fetch_entry(None,{},root,'../escape')
    except AssertionError:checks.append('archive traversal rejected')
    else:raise AssertionError('Traversal accepted')
name=b'MVSEC/outdoor_day/outdoor_day1_data.hdf5'
extra=struct.pack('<HHQQQ',1,24,2**32+7,2**32+9,2**32+20)
header=struct.pack('<4s6H3I5H2I',b'PK\x01\x02',45,45,0,8,0,0,678,0xffffffff,0xffffffff,len(name),len(extra),0,1,0,0,0xffffffff)
entry=central_entries(header+name+extra)[0]
assert (entry['uncompressed'],entry['compressed'],entry['local_offset'],entry['disk'])==(2**32+7,2**32+9,2**32+20,1)
checks.append('ZIP64 sizes, offsets and disk number')
try:central_entries((header+name+extra)*2)
except AssertionError:checks.append('duplicate central filename rejected')
else:raise AssertionError('Duplicate filename accepted')
result=dict(status='passed',procedural_only=True,SSH_used=False,real_data_used=False,checks=checks)
Path(__file__).with_name('hf_extractor_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
