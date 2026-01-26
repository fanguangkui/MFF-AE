import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd



class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.save_for_backward(inputs, targets, momentum)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, momentum = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y, weight in zip(inputs, targets, momentum):
            ctx.features[y] = weight * ctx.features[y] + (1. - weight) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, momentum)

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_cluster, temp=0.5, momentum=0.1, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_cluster = num_cluster

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_cluster, num_features))

    def forward(self, inputs, targets=None, weights=None):
        if targets is not None:
            inputs = F.normalize(inputs, dim=1).cuda()
            if self.use_hard:
                # outputs = cm_hard(inputs, targets, self.features, weights)
                outputs = cm(inputs, targets, self.features, self.momentum)
                feature_norm = torch.norm(self.features, dim=1)
                cos_simi = outputs / feature_norm
                # outputs /= self.temp
                # loss = F.cross_entropy(outputs, targets, label_smoothing=0.005)
                result = torch.gather(cos_simi, 1, targets.unsqueeze(1)).squeeze()

                return {
                    # "cos_simi": (1-result).mean(),
                    "cos_simi": ((1-result)*(torch.exp(weights)/self.temp)).mean(),
                    "logits": outputs,
                }
            else:
                outputs = cm(inputs, targets, self.features, self.momentum)

                feature_norm = torch.norm(self.features, dim=1)
                cos_simi = outputs / feature_norm
                # outputs /= self.temp
                # loss = F.cross_entropy(outputs, targets, label_smoothing=0.005)
                result = torch.gather(cos_simi, 1, targets.unsqueeze(1)).squeeze()
                return {
                    # "cos_simi": cos_simi[:, targets].mean(),
                    "cos_simi": (1 - result).mean(),
                    "logits": outputs,
                }
        else:
            inputs = F.normalize(inputs, dim=1).cuda()
            outputs = inputs.mm(self.features.t())
            feature_norm = torch.norm(self.features, dim=1)
            outputs = outputs / feature_norm

            # outputs /= self.temp
            return {
                # "cos_simi": cos_simi,
                "logits": outputs
            }