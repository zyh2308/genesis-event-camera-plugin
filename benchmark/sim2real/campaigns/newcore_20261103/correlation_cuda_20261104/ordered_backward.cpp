#include <torch/extension.h>
std::vector<torch::Tensor> ordered_backward_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor g);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&ordered_backward_cuda);}
