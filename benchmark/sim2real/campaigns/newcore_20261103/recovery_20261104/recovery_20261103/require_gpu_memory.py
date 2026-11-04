import os,subprocess,sys,time
uuid=os.environ['CUDA_VISIBLE_DEVICES']
while True:
    free=int(subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free >= int(sys.argv[1]):
        print('GPU free MiB:',free,flush=True)
        os.execv(sys.argv[2],sys.argv[2:])
    print('Waiting for GPU free MiB:',free,flush=True)
    time.sleep(60)
