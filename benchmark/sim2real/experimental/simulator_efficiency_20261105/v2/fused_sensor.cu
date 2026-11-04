#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/generic_math.h>
using torch::Tensor;
__device__ void emit(double* events,int64_t* keys,int& count,int cap,int pixel,int N,int W,int t,int i,int phase,int polarity,double stamp,int* flags){
    const int k=count++;
    if(k>=cap){atomicOr(flags,2);return;}
    const int64_t pos=int64_t(pixel)*cap+k;
    events[pos*4]=stamp;events[pos*4+1]=pixel%W;events[pos*4+2]=pixel/W;events[pos*4+3]=polarity;
    // Original append order: timestep, signal/noise, crossing layer, ON/OFF, y/x.
    keys[pos]=(int64_t(t)<<48)+(int64_t(phase)<<47)+(int64_t(i)*2+(polarity<0))*(1<<20)+pixel;
}
__global__ void sensor_kernel(const double* logs,const double* eps,const double* leak,const bool* on,const bool* off,const double* noise_ts,const double* times,const double* deltas,const double* lp0,const double* base0,const double* mem0,const double* pos,const double* neg,const double* photo,double* lp_out,double* base_out,double* mem_out,double* diff_out,double* events,int64_t* keys,int* counts,int* flags,int T,int N,int W,double refractory,int cap){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=N)return;
    double lp=lp0[p],base=base0[p],last=mem0[p],diff=0;int count=0;
    for(int t=0;t<T;++t){
        const int64_t j=int64_t(t)*N+p;
        const double signal_start=lp+photo[p];const double base_start=base;
        const double e=eps[j];lp=(1.0-e)*lp+e*logs[j];base=base-leak[j];
        const double signal_end=lp+photo[p];diff=signal_end-base;
        const double rel_start=signal_start-base_start,delta=diff-rel_start;
        const int np=int(c10::div_floor_floating(fmax(diff,0.0),pos[p]));
        const int nn=int(c10::div_floor_floating(fmax(-diff,0.0),neg[p]));
        if((np>0 || nn>0) && fabs(delta)<=1e-12){atomicOr(flags,1);counts[p]=0;return;}
        if(np>262000 || nn>262000){atomicOr(flags,4);counts[p]=0;return;}
        const double stable=fabs(delta)>1e-12?delta:1.0;
        int accepted_pos=0,accepted_neg=0;
        for(int i=0;i<max(np,nn);++i){
            double alpha=fmin(1.0,fmax(0.0,((double(i)+1.0)*pos[p]-rel_start)/stable));
            double stamp=(times[t]-deltas[t])+deltas[t]*alpha;
            if(i<np && stamp-last>=refractory){last=stamp;++accepted_pos;emit(events,keys,count,cap,p,N,W,t,i,0,1,stamp,flags);}
            alpha=fmin(1.0,fmax(0.0,(-(double(i)+1.0)*neg[p]-rel_start)/stable));
            stamp=(times[t]-deltas[t])+deltas[t]*alpha;
            if(i<nn && stamp-last>=refractory){last=stamp;++accepted_neg;emit(events,keys,count,cap,p,N,W,t,i,0,-1,stamp,flags);}
        }
        bool sn_on=on[j],sn_off=off[j]&&!sn_on;
        const double nt=noise_ts[j];
        if(nt-last<refractory){sn_on=false;sn_off=false;}
        if(sn_on||sn_off){last=nt;emit(events,keys,count,cap,p,N,W,t,0,1,sn_on?1:-1,nt,flags);}
        // Preserve separate multiply/add operations from the frozen implementation.
        base=base+double(float(accepted_pos))*pos[p];base=base-double(float(accepted_neg))*neg[p];
        if(sn_on||sn_off)base=signal_end;
    }
    lp_out[p]=lp;base_out[p]=base;mem_out[p]=last;diff_out[p]=diff;counts[p]=count;
}
std::vector<Tensor> sensor_cuda(Tensor logs,Tensor eps,Tensor leak,Tensor on,Tensor off,Tensor noise_ts,Tensor times,Tensor deltas,Tensor lp,Tensor base,Tensor mem,Tensor pos,Tensor neg,Tensor photo,double refractory,int cap){
    for(const auto& x:{logs,eps,leak,on,off,noise_ts,times,deltas,lp,base,mem,pos,neg,photo}){
        TORCH_CHECK(x.is_cuda() && x.is_contiguous(),"CUDA contiguous tensors required");
        TORCH_CHECK(x.device()==logs.device(),"device mismatch");
    }
    TORCH_CHECK(logs.dim()==3 && logs.scalar_type()==torch::kFloat64,"expected double T,H,W log signal");
    TORCH_CHECK(eps.sizes()==logs.sizes() && leak.sizes()==logs.sizes() && on.sizes()==logs.sizes() && off.sizes()==logs.sizes() && noise_ts.sizes()==logs.sizes(),"time stack shape mismatch");
    TORCH_CHECK(on.scalar_type()==torch::kBool && off.scalar_type()==torch::kBool,"noise mask dtype mismatch");
    TORCH_CHECK(times.numel()==logs.size(0) && deltas.numel()==logs.size(0),"timestamp shape mismatch");
    for(const auto& x:{eps,leak,noise_ts,times,deltas,lp,base,mem,pos,neg,photo})TORCH_CHECK(x.scalar_type()==torch::kFloat64,"double precision required");
    const int T=logs.size(0),H=logs.size(1),W=logs.size(2),N=H*W;
    TORCH_CHECK(T>0 && T<32768 && N<(1<<20) && cap>0,"unvalidated dimensions/capacity");
    for(const auto& x:{lp,base,mem,pos,neg,photo})TORCH_CHECK(x.numel()==N,"state shape mismatch");
    TORCH_CHECK(refractory>0,"positive refractory period required");
    const c10::cuda::CUDAGuard guard(logs.device());
    auto lo=torch::empty_like(lp),bo=torch::empty_like(base),mo=torch::empty_like(mem),df=torch::empty_like(lp);
    auto events=torch::empty({N,cap,4},logs.options());auto keys=torch::empty({N,cap},logs.options().dtype(torch::kInt64));
    auto counts=torch::zeros({N},logs.options().dtype(torch::kInt32));auto flags=torch::zeros({1},logs.options().dtype(torch::kInt32));
    sensor_kernel<<<(N+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(logs.data_ptr<double>(),eps.data_ptr<double>(),leak.data_ptr<double>(),on.data_ptr<bool>(),off.data_ptr<bool>(),noise_ts.data_ptr<double>(),times.data_ptr<double>(),deltas.data_ptr<double>(),lp.data_ptr<double>(),base.data_ptr<double>(),mem.data_ptr<double>(),pos.data_ptr<double>(),neg.data_ptr<double>(),photo.data_ptr<double>(),lo.data_ptr<double>(),bo.data_ptr<double>(),mo.data_ptr<double>(),df.data_ptr<double>(),events.data_ptr<double>(),keys.data_ptr<int64_t>(),counts.data_ptr<int>(),flags.data_ptr<int>(),T,N,W,refractory,cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {lo,bo,mo,df,events,keys,counts,flags};
}
