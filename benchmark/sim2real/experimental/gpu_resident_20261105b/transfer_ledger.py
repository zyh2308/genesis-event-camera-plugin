"""Python-visible device downloads/synchronizations, not a PCIe bus counter."""
import torch
class TransferLedger:
    def __init__(self):self.active=False;self.values={}
    def install(self):
        originals={name:getattr(torch.Tensor,name) for name in ['cpu','item','__bool__']}
        ledger=self
        for name,method in originals.items():
            def wrapped(tensor,*args,_name=name,_method=method,**kwargs):
                if ledger.active and tensor.is_cuda:
                    key={'cpu':'tensor_cpu_calls','item':'scalar_item_calls','__bool__':'tensor_bool_calls'}[_name]
                    ledger.values[key]=ledger.values.get(key,0)+1
                    if _name=='cpu':ledger.values['tensor_cpu_logical_bytes']=ledger.values.get('tensor_cpu_logical_bytes',0)+tensor.numel()*tensor.element_size()
                return _method(tensor,*args,**kwargs)
            setattr(torch.Tensor,name,wrapped)
        self.originals=originals
    def start(self):self.values={};self.active=True
    def stop(self):self.active=False;return dict(self.values)
    def uninstall(self):
        for name,method in self.originals.items():setattr(torch.Tensor,name,method)
