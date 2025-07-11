import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import copy


def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret

    return _use_grad


class IESGD_6(Optimizer):
    def __init__(self, params, lr=0.1, lambda_1=1.0, lambda_2=1.0, gamma=1.0, omega=0., momentum=0.,
                 weight_decay=0., *, maximize=False, foreach: Optional[bool] = None,
                 differentiable=False):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lambda_1=lambda_1, lambda_2=lambda_2, gamma=gamma, omega=omega, momentum=momentum,
                        weight_decay=weight_decay,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable)
        super(IESGD_6, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)
        # state_values = list(self.state.values())
        # step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        # if not step_is_tensor:
        #     for s in state_values:
        #         s['step'] = torch.tensor(float(s['step']))

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            param_list = []
            param_grad_list = []
            m_buffer_list = []
            v_buffer_list = []
            param_grad_pre_list = []

            for p in group['params']:
                if p.grad is not None:
                    param_list.append(p)
                    param_grad_list.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        # Single integral
                        # m_buffer_list.append(None)
                        state['m_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Double integral
                        # v_buffer_list.append(None)
                        state['v_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # 前一个梯度
                        state['param_grad_pre'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                        # param_grad_pre_list.append(None)

                    m_buffer_list.append(state['m_buffer'])
                    v_buffer_list.append(state['v_buffer'])
                    param_grad_pre_list.append(state['param_grad_pre'])
            gradFlowAsError(param_list,
                            param_grad_list,  # 当前
                            m_buffer_list,
                            v_buffer_list,
                            param_grad_pre_list,  # 从state获取
                            weight_decay=group['weight_decay'],
                            momentum=group['momentum'],
                            lr=group['lr'],
                            lambda_1=group['lambda_1'],
                            lambda_2=group['lambda_2'],
                            gamma=group['gamma'],
                            omega=group['omega'],
                            maximize=group['maximize'],
                            foreach=group['foreach'])

            # update momentum_buffers in state
            for p, m, v, pd_pre in zip(param_list, m_buffer_list, v_buffer_list, param_grad_pre_list):
                state = self.state[p]  # 对62个参数的每个状态都更新相应的buf
                state['m_buffer'] = m
                state['v_buffer'] = v
                state['param_grad_pre'] = pd_pre
        return loss


def gradFlowAsError(params: List[Tensor], param_grad_list: List[Tensor],
                    m_buffer_list: List[Optional[Tensor]], v_buffer_list: List[Optional[Tensor]],
                    param_grad_pre_list: List[Optional[Tensor]],
                    foreach: bool = None,
                    *,
                    weight_decay: float, momentum: float, lr: float,
                    lambda_1: float, lambda_2: float, gamma: float, omega: float,
                    maximize: bool):
    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = None  # _multi_tensor_gradFlowAsError
    else:
        func = _single_tensor_gradFlowAsError

    func(params,
         param_grad_list,
         m_buffer_list,
         v_buffer_list,
         param_grad_pre_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         lambda_1=lambda_1,
         lambda_2=lambda_2,
         gamma=gamma,
         omega=omega,
         maximize=maximize)


def _single_tensor_gradFlowAsError(params: List[Tensor],
                                   param_grad_list: List[Tensor],
                                   m_buffer_list: List[Optional[Tensor]],
                                   v_buffer_list: List[Optional[Tensor]],
                                   param_grad_pre_list: List[Optional[Tensor]],
                                   *,
                                   weight_decay: float,
                                   momentum: float,
                                   lr: float,
                                   lambda_1: float,
                                   lambda_2: float,
                                   gamma: float,
                                   omega: float,
                                   maximize: bool):
    for i, param in enumerate(params):  # param是网络的第i个的参数
        # param_grad, m_buf, v_buf 更新时对应的列表元素也更新了
        param_grad = param_grad_list[i]
        m_buf = m_buffer_list[i]
        v_buf = v_buffer_list[i]
        param_grad_pre = param_grad_pre_list[i]

        if weight_decay != 0:
            param_grad = param_grad.add(param, alpha=weight_decay)

        # sum The single integral and the double integral
        m_buf.add_(param_grad)
        v_buf.add_(m_buf)

        tmp_param_grad = (1+lr*gamma)*param_grad - param_grad_pre + lambda_1*lr*lr*m_buf + lambda_2*lr*lr*lr*v_buf
        param.add_(tmp_param_grad, alpha=-omega)
        
        # param_grad_list[i] = param_grad.detach().clone()
        param_grad_pre = param_grad.detach().clone()
        param_grad_pre_list[i] = param_grad_pre
        


