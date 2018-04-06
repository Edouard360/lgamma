from torch import autograd

from functions.internals import lgamma, polygamma


class Lgamma(autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return lgamma(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return grad_output * polygamma(0, input)
