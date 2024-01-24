import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import gradcheck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

def linear(input, weight, bias=None):
    return LinearFunction.apply(input, weight, bias)


class BatchNorm1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, moving_mean, moving_var, grad_enable, eps=1e-5, momentum=0.1):  # moving_mean/var 用list的可变对象来处理
        if not grad_enable:
            # print(len(moving_mean[0]), moving_var[0])
            x_normalized = (x - moving_mean[0]) / torch.sqrt(moving_var[0] + eps)
        else:
            ctx.eps = eps
            ctx.momentum = momentum
            ctx.save_for_backward(x, gamma, beta)

            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(var + eps)
            moving_mean[0] = (1 - momentum) * moving_mean[0] + momentum * mean
            moving_var[0] = (1 - momentum) * moving_var[0] + momentum * var
            ctx.save_for_backward(x, gamma, beta, mean, var, x_normalized)

        # print(x_normalized.shape, gamma.shape)
        y = gamma * x_normalized + beta
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, gamma, beta, mean, var, x_normalized = ctx.saved_tensors

        N = x.size(0)
        inv_var = 1.0 / torch.sqrt(var + ctx.eps)

        grad_gamma = (grad_output * x_normalized).sum(dim=0, keepdim=True)
        grad_beta = grad_output.sum(dim=0, keepdim=True)
        grad_x_normalized = grad_output * gamma

        grad_var = (grad_x_normalized * (x - mean) * -0.5 * inv_var**3).sum(dim=0, keepdim=True)  # inv_var**3 = (var + ctx.eps)**(-1.5)
        grad_mean = (grad_x_normalized * -inv_var).sum(dim=0, keepdim=True) + grad_var * (x - mean).mean(dim=0, keepdim=True) * -2.0 / N
        grad_x = grad_x_normalized * inv_var + grad_var * 2.0 * (x - mean) / N + grad_mean / N

        return grad_x, grad_gamma, grad_beta, None, None, None


def test():
    input = (torch.randn(20,20,dtype=torch.double,requires_grad=True), torch.randn(30,20,dtype=torch.double,requires_grad=True))
    test = gradcheck(linear, input, eps=1e-6, atol=1e-4)
    print('Linear:', test)
    input = (
        torch.randn(20, 20, dtype=torch.double, requires_grad=True),  # x
        torch.randn(20, dtype=torch.double, requires_grad=True),      # gamma
        torch.randn(20, dtype=torch.double, requires_grad=True),      # beta
    )
    test = gradcheck(BatchNorm1dFunction.apply, input, eps=1e-6, atol=1e-4)
    print('BN:', test)

class BatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm1d, self).__init__()
        self.moving_mean = [0.0]
        self.moving_var = [0.0]
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))

    def forward(self, x):
        y = BatchNorm1dFunction.apply(x, self.gamma, self.beta, self.moving_mean, self.moving_var, torch.is_grad_enabled())
        return y
    

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # He Initialization for weights
        init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        init.zeros_(self.bias)

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, BN = False):
        super(MLP, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            layers.append(LinearLayer(sizes[i], sizes[i+1]))
            # layers.append(nn.Linear(sizes[i], sizes[i+1]))   # 官方
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if BN:
                    layers.append(BatchNorm1d(sizes[i+1]))
                    # layers.append(nn.BatchNorm1d(sizes[i+1]))   # 官方

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class Residual_MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, BN = False):
        super(Residual_MLP, self).__init__()
        self.begin = MLP(input_size, [], hidden_sizes[0], BN)

        MLP_layer = []
        redusial_layer = []
        for i in range(8):
            head = i * 5
            last = min(39, (i + 1) * 5)
            MLP_layer.append(MLP(hidden_sizes[head], hidden_sizes[head + 1 : last], hidden_sizes[last], BN))
            redusial_layer.append(MLP(hidden_sizes[head], [], hidden_sizes[last], BN))

        self.MLP = nn.Sequential(*MLP_layer)
        self.redusial = nn.Sequential(*redusial_layer)
        self.end = MLP(hidden_sizes[-1], [], output_size, BN)

    def forward(self, x):
        x = self.begin(x)
        for i in range(8):
            residual = self.redusial[i](x)
            x = self.MLP[i](x) + residual
        x = self.end(x)
        return x
    