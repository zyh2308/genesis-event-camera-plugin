#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
template<typename T> __global__ void backward_kernel(const T* a,const T* b,const T* g,T* da,T* db,int B,int C,int H,int W,int64_t count){
  const int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(k>=count)return;
  const int x=k%W,y=(k/W)%H,n=k/(int64_t(C)*H*W);
  T va=T(0),vb=T(0);
  // Match the reference's dy-major / dx-minor additions exactly. No FMA.
  for(int dy=-4;dy<=4;++dy)for(int dx=-4;dx<=4;++dx){
    const int shift=(dy+4)*9+dx+4;
    const int yt=y+dy,xt=x+dx;
    if(yt>=0 && yt<H && xt>=0 && xt<W){
      const int64_t q=k+int64_t(dy)*W+dx;
      const int64_t gi=((int64_t(n)*81+shift)*H+y)*W+x;
      const T product=g[gi]*b[q];va=va+product;
    }
    const int ys=y-dy,xs=x-dx;
    if(ys>=0 && ys<H && xs>=0 && xs<W){
      const int64_t q=k-int64_t(dy)*W-dx;
      const int64_t gi=((int64_t(n)*81+shift)*H+ys)*W+xs;
      const T product=g[gi]*a[q];vb=vb+product;
    }
  }
  da[k]=va;db[k]=vb;
}
std::vector<torch::Tensor> ordered_backward_cuda(torch::Tensor a,torch::Tensor b,torch::Tensor g){
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && g.is_cuda(),"CUDA required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && g.is_contiguous(),"contiguous required");
  TORCH_CHECK(a.sizes()==b.sizes() && a.dim()==4 && g.dim()==5,"shape mismatch");
  TORCH_CHECK(a.device()==b.device() && a.device()==g.device(),"device mismatch");
  TORCH_CHECK(a.scalar_type()==b.scalar_type() && a.scalar_type()==g.scalar_type(),"dtype mismatch");
  TORCH_CHECK(g.size(0)==a.size(0) && g.size(1)==9 && g.size(2)==9 && g.size(3)==a.size(2) && g.size(4)==a.size(3),"grad shape mismatch");
  const c10::cuda::CUDAGuard guard(a.device());
  auto da=torch::empty_like(a),db=torch::empty_like(b);
  const int64_t count=a.numel();
  if(count){
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(),"ordered_correlation_backward",[&]{
      backward_kernel<scalar_t><<<(count+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),g.data_ptr<scalar_t>(),da.data_ptr<scalar_t>(),db.data_ptr<scalar_t>(),a.size(0),a.size(1),a.size(2),a.size(3),count);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {da,db};
}
