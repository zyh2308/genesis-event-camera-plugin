#include <torch/extension.h>
std::vector<torch::Tensor> sensor_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,int);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&sensor_cuda);}
