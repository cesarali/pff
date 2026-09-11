import torch
import gpytorch

def create_kernel(config):
    kernel_params = config.kernel_params
    if 'type' not in kernel_params:
        raise ValueError("Kernel type must be specified in kernel_params")
    if kernel_params['type'] == 'RBF':
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=config.input_dim, requires_grad=False)
        kernel_params_ = kernel_params.get('params', {})
        kernel_length_scale = kernel_params_["raw_lengthscale"]
        kernel_length_scale = torch.tensor([kernel_length_scale] * config.input_dim)
        kernel.initialize(raw_lengthscale=kernel_length_scale)
        return kernel
    raise ValueError(f"Unsupported kernel type: {kernel_params['type']}")


def create_kernel_mix(kernel_params,input_dim=1):
    if 'type' not in kernel_params:
        raise ValueError("Kernel type must be specified in kernel_params")
    if kernel_params['type'] == 'RBF':
        kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, requires_grad=False)
        kernel_params_ = kernel_params.get('params', {})
        kernel_length_scale = kernel_params_["raw_lengthscale"]
        kernel_length_scale = torch.tensor([kernel_length_scale] * input_dim)
        kernel.initialize(raw_lengthscale=kernel_length_scale)
        return kernel
    raise ValueError(f"Unsupported kernel type: {kernel_params['type']}")