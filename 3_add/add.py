
import torch
from add_bwd import mask_func
from add_fwd import add_fwd_func
from add_relu_fwd import add_relu_fwd_func


class AddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs, rhs):
        # return lhs + rhs
        return add_fwd_func(lhs, rhs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class AddReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lhs, rhs):
        # NOTE `mask` is boolean, packed into `uint8` tensor
        result, mask = add_relu_fwd_func(lhs, rhs)
        ctx.save_for_backward(mask)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]

        # - compute `grad_output * mask` elem-wise
        grad_input = mask_func(grad_output, mask)
        # grad_input = grad_output * mask * 0.5

        return grad_input, grad_input

class Add(torch.nn.Module):
    def __init__(self, init_param, apply_relu=False):
        super().__init__()
        self.bias = torch.nn.Parameter(init_param, requires_grad=True)
        self.apply_relu = apply_relu

    def forward(self, x):
        if self.apply_relu:
            return AddReluFunction.apply(self.bias, x)
        else:
            return AddFunction.apply(self.bias, x)
