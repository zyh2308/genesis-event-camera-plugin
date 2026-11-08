"""Read-only, byte-preserving receipts of the actual task-policy data path."""
from __future__ import annotations
import base64
import hashlib
import json


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def encode_value(value,np,torch):
    if isinstance(value,torch.Tensor):
        array=value.detach().cpu().contiguous().numpy()
        return dict(kind='torch.Tensor',dtype=str(value.dtype),numpy_dtype=array.dtype.str,
            shape=list(value.shape),device=str(value.device),requires_grad=bool(value.requires_grad),
            bytes_base64=base64.b64encode(array.tobytes(order='C')).decode('ascii'))
    if isinstance(value,(np.ndarray,np.generic)):
        array=np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError('Object arrays cannot serve as numeric policy evidence')
        return dict(kind='numpy.ndarray' if isinstance(value,np.ndarray) else 'numpy.scalar',
            dtype=array.dtype.str,shape=list(array.shape),
            bytes_base64=base64.b64encode(array.tobytes(order='C')).decode('ascii'))
    if isinstance(value,dict):
        if not all(isinstance(key,str) for key in value):
            raise TypeError('Policy receipt expects string dictionary keys')
        return dict(kind='mapping',items=[[key,encode_value(item,np,torch)] for key,item in value.items()])
    if isinstance(value,(tuple,list)):
        return dict(kind='tuple' if isinstance(value,tuple) else 'list',items=[encode_value(x,np,torch) for x in value])
    if value is None or isinstance(value,(str,bool,int)):
        return dict(kind=type(value).__name__,value=value)
    if isinstance(value,float):
        return dict(kind='float',hex=value.hex())
    raise TypeError('Unsupported actual policy value: '+str(type(value)))


class QueryWitness:
    def __init__(self,np,torch):
        self.np,self.torch=np,torch
        self.current=None
        self.queries=[]

    def encode(self,value):
        return encode_value(value,self.np,self.torch)

    def begin_query(self,time_s):
        if self.current is not None:
            raise RuntimeError('Previous actual query was not completed by an environment step')
        self.current=dict(query_index=len(self.queries),time_s=float(time_s),model_calls=[])

    def frontend_result(self,result):
        if self.current is None:
            raise RuntimeError('No active actual frontend query')
        self.current['frontend_result']=self.encode(result)

    def model_input(self,args,kwargs,time_s):
        if self.current is None or abs(float(time_s)-self.current['time_s'])>1e-8:
            raise RuntimeError('Actor called without its matching actual query clock')
        return dict(call_index=len(self.current['model_calls']),time_s=float(time_s),
            inputs=self.encode(dict(args=args,kwargs=kwargs)))

    def model_output(self,call,output):
        call['output']=self.encode(output)
        self.current['model_calls'].append(call)

    def complete_query(self,time_s,held,event_state_none,actual_action,normalized_action):
        if self.current is None or abs(float(time_s)-self.current['time_s'])>1e-8:
            raise RuntimeError('Environment step did not follow its actual frontend query')
        if 'frontend_result' not in self.current:
            raise RuntimeError('Missing actual frontend return value')
        if bool(event_state_none)!=(self.current['frontend_result']['kind']=='NoneType'):
            raise RuntimeError('Caller None flag differs from the actual frontend return')
        if len(self.current['model_calls'])!=(0 if event_state_none else 1):
            raise RuntimeError('Model-call count differs from the sealed runner actual state branch')
        if event_state_none and not held:
            raise RuntimeError('Sealed runner must hold its action when event state is None')
        self.current.update(held=bool(held),event_state_none=bool(event_state_none),
            model_called=bool(self.current['model_calls']),model_call_count=len(self.current['model_calls']),
            actual_ignored_actor_action=self.encode(actual_action),
            actual_normalized_runner_action=self.encode(normalized_action))
        self.queries.append(self.current)
        self.current=None

    def checked_payload(self,expected_times):
        if self.current is not None or len(self.queries)!=len(expected_times):
            raise ValueError('Incomplete actual policy-query witness')
        if any(abs(query['time_s']-float(t))>1e-8 for query,t in zip(self.queries,expected_times)):
            raise ValueError('Actual policy query timestamps differ from the fixed control clock')
        return dict(schema='actual_task_policy_query_bytes_v1',
            information='Actual full act_inference inputs/outputs and actual pre-shadow action dict; all control queries, including warmup',
            query_count=len(self.queries),queries=self.queries,
            query_payload_sha256=canonical_sha256(self.queries))


def raw_event_summary(events,np):
    content=dict(dtype=events.dtype.str,shape=list(events.shape))
    digest=hashlib.sha256(json.dumps(content,sort_keys=True).encode()+events.tobytes(order='C')).hexdigest()
    return dict(**content,content_sha256=digest,count=len(events))


def compare_raw_events(left,right,np):
    a,b=raw_event_summary(left,np),raw_event_summary(right,np)
    exact=a['content_sha256']==b['content_sha256']
    result=dict(raw_exact=exact,left=a,right=b,per_pixel_polarity_multiplicity_exact=False,
        matched_event_time_different_n=None,matched_event_time_diff_max_s=None,matched_event_time_diff_p99_s=None)
    if left.shape==right.shape and left.ndim==2 and left.shape[1]==4:
        # Sort within each (x,y,polarity) group before pairing time ranks. Never
        # mislabel a global row shift as a changed spatial event population.
        ai=np.lexsort((left[:,0],left[:,3],left[:,2],left[:,1]))
        bi=np.lexsort((right[:,0],right[:,3],right[:,2],right[:,1]))
        x,y=left[ai],right[bi]
        same=np.array_equal(x[:,1:],y[:,1:])
        result['per_pixel_polarity_multiplicity_exact']=bool(same)
        if same:
            delta=np.abs(x[:,0]-y[:,0])
            result.update(matched_event_time_different_n=int(np.count_nonzero(delta)),
                matched_event_time_diff_max_s=float(delta.max()) if len(delta) else 0.,
                matched_event_time_diff_p99_s=float(np.percentile(delta,99)) if len(delta) else 0.)
    return result
