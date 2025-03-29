import torch
from torch.optim.optimizer import Optimizer
from typing import List, Optional
from torch import Tensor
import math


class AdaAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach,
                        capturable=capturable)
        super().__init__(params, defaults)
        self._confidence = 1
        self._acc_beta1 = 1
        self._acc_beta2 = 1

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = ((len(state_values) != 0) and
                          torch.is_tensor(state_values[0]['step']))
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float,
                                                    device=p.device) \
                            if self.defaults['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            # state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # if group['amsgrad']:
                    #     max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            beta1 = (beta1 ** self._confidence)
            beta2 = (beta2 ** self._confidence)
            self._acc_beta1 *= beta1
            self._acc_beta2 *= beta2
            lr = group['lr'] * self._confidence if state_steps[0] > 5 else group['lr']
            # print(f"beta1: {beta1}, beta2: {beta2}, lr: {lr}, "
            #       f"confidence: {round(self.get_confidence(), 4)}")
            # lr = group['lr'] * (self._confidence ** (1-self._acc_beta1))
            adaadam(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    self._acc_beta1,
                    self._acc_beta2,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize'],
                    capturable=group['capturable']
                    )

        return loss

    def set_confidence(self, confidence):
        # self._confidence = confidence ** 0.5
        self._confidence = confidence

    def get_confidence(self):
        return self._confidence


def adaadam(params: List[Tensor],
            grads: List[Tensor],
            exp_avgs: List[Tensor],
            exp_avg_sqs: List[Tensor],
            max_exp_avg_sqs: List[Tensor],
            state_steps: List[Tensor],
            acc_beta1,
            acc_beta2,
            # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
            # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
            capturable: bool = False,
            *,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
            maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - acc_beta1  # torch.pow(beta1, step)
            bias_correction2 = 1 - acc_beta2  # torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()

            bias_correction1 = 1 - acc_beta1
            bias_correction2 = 1 - acc_beta2

            step_size = lr / bias_correction1

            bias_correction2_sqrt = math.sqrt(bias_correction2)

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size)
