import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional
import copy

class GradFlowAsError(Optimizer):

    def __init__(self, params, lr=0.1, lam=1.0, gam=1.0, w = 0., momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False, *, maximize=False, foreach: Optional[bool] = None):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, lam=lam, gam=gam, w=w, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach)
        # if nesterov and (momentum <= 0 or dampening != 0):
        #     raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GradFlowAsError, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)

    # @torch.no_grad()
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
            d_p_list = []
            momentum_buffer_list = []
            para_buffer_list = []
            p_km1_list = []
            dp_km1_list = []
            
            has_sparse_grad = False

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        has_sparse_grad = True

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

                    if 'para_buffer' not in state:
                        para_buffer_list.append(None)
                    else:
                        para_buffer_list.append(state['para_buffer'])
                        
                    if 'p_km1' not in state:
                        p_km1_list.append(None)
                    else:
                        p_km1_list.append(state['p_km1'])   
                        
                    if 'dp_km1' not in state:
                        dp_km1_list.append(None)
                    else:
                        dp_km1_list.append(state['dp_km1'])                                               
            gradFlowAsError(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                para_buffer_list,
                p_km1_list,
                dp_km1_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                # momentum2=group['momentum2'],

                lr=group['lr'],
                lam=group['lam'],
                gam=group['gam'],
                w=group['w'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'])

            # update momentum_buffers in state
            for p, momentum_buffer,para_buffer,p_km1,dp_km1 in zip(params_with_grad, momentum_buffer_list,para_buffer_list,p_km1_list,dp_km1_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                state['para_buffer'] = para_buffer
                state['p_km1'] = p_km1
                state['dp_km1'] = dp_km1                
        return loss


def gradFlowAsError(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        para_buffer_list: List[Optional[Tensor]],
        p_km1_list: List[Optional[Tensor]],
        dp_km1_list: List[Optional[Tensor]],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: bool = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        lam: float,
        gam: float,
        w: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_gradFlowAsError
    else:
        func = _single_tensor_gradFlowAsError

    func(params,
         d_p_list,
         momentum_buffer_list,
         para_buffer_list,
         p_km1_list,
         dp_km1_list,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         lam=lam,
         gam=gam,
         w=w,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_gradFlowAsError(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       para_buffer_list: List[Optional[Tensor]],
                       p_km1_list: List[Optional[Tensor]],
                       dp_km1_list: List[Optional[Tensor]],
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       lam: float,
                       gam: float,
                       w: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):
    lam_ = lam/lr
    # lam = lambda_/lr
    # gam = 10
    for i, p_k in enumerate(params):

        dp_k = d_p_list[i]
        # dp_k += torch.randn(dp_k.size()).mul(dp_k*0.5)
        if weight_decay != 0:
            dp_k = dp_k.add(p_k, alpha=weight_decay)

        if momentum != 0:
            
            sum_dp = momentum_buffer_list[i]
            dot_p_buf = para_buffer_list[i]
            p_km1 = p_km1_list[i]
            dp_km1 = dp_km1_list[i]
            ini = 0
            if sum_dp is None:
                # print('sum_dp is None')
                sum_dp = dp_k#.detach().clone()#torch.clone(dp_k).detach()copy.deepcopy(dp_k.data)
                momentum_buffer_list[i] = sum_dp
                ini = 1
            if dot_p_buf is None:
                # print('dot_p_buf is None')
                dot_p_buf =  p_k#.detach().clone() # copy.deepcopy(p_k.data)
                para_buffer_list[i] = dot_p_buf
                ini = 1
            if p_km1 is None:
                # print('p_km1 is None')
                p_km1 = p_k#.detach().clone()#copy.deepcopy(p_k.data)torch.clone(p_k).detach()
                p_km1_list[i] = p_km1   
                ini = 1 
            if dp_km1 is None:
                # print('dp_km1 is None')
                dp_km1 = dp_k#.detach().clone()# copy.deepcopy(dp_k.data)
                dp_km1_list[i] = dp_km1     
                ini = 1     
            if ini == 0:
                # print("GradFlowAsError")
                # sum_dp.mul_(momentum).add_(dp_k, alpha=1 - dampening)
                # 
                # sum_dp.mul_(momentum).add_(dp_k, alpha=1)
                sum_dp.mul_(momentum)
                sum_dp = sum_dp.add(dp_k, alpha=1)
                
                # dot_p_buf.mul_(momentum).add_((p_k-p_km1)/lr, alpha=1)
          
                # p_k = (-lr* (1+lam *lr)* dp_k-2* p_k+p_km1+lr* (dp_km1+lam* p_k-gam* lr* (sum_dp+dot_p_buf)))/(1+lam* lr)
                # p_k = copy.deepcopy(p_k)
                # print('p_k_bef',p_k)
                
                # p_k.mul_((-2+lam *lr)/(1+lam *lr)).add_((-lr* (1+lam *lr)* dp_k+p_km1+lr* (dp_km1-gam* lr* (sum_dp+dot_p_buf))), alpha=1.0/(1+lam* lr))
                # p_k.mul_((2+lam_ *lr-gam*lr)/(1+lam_ *lr)).add_(((dp_km1 *lr - dp_k *lr* (1 + lam_* lr) + gam* lr* dot_p_buf - p_km1 -  gam* lr**2* sum_dp)), alpha=1.0/(1+lam_* lr))
                # p_k.add_(-1./lam*(dp_k-dp_km1) - gam/lam*lr*sum_dp)
                # p_k.add_(-1./lam*(dp_k-dp_km1) - gam/lam*lr*sum_dp-lr/lam*dp_k)
                # p_k.add_(-(dp_k-dp_km1) -lr*lam*dp_k - gam*lr*sum_dp)
                
                # p_k.add_( -w*(dp_k-dp_km1)-lr*lam*w*dp_k - gam*lr*w*sum_dp)
                param_1 = dp_km1-dp_k   #-1*(dp_k-dp_km1)
                param_2 = dp_k.mul_(lr*lam)
                param_3 = sum_dp.mul_(gam*lr)
                param_4 = param_1.sub_(param_2)
                param_5 = param_4.sub_(param_3)
                p_k = p_k.add(param_5)#, alpha=w
                p_k.mul_(w)

                # p_k.add_(-lr/1*dp_k)
                # print('p_k_aft',p_k)
                params[i] = p_k
                p_km1 = p_k#.detach().clone()#.detach()copy.deepcopy(p_k.data)
                p_km1_list[i] = p_km1
                dp_km1 = dp_k#.detach().clone()#.detach()copy.deepcopy(dp_k.data)
                dp_km1_list[i] = dp_km1      
                # dot_p_buf.mul_(momentum).add_(sum_dp, alpha=1 - momentum)
                # buf_tmp = torch.clone(sum_dp).detach()
                # nested = 10
                # for i in range(0, nested):
                #     dot_p_buf.mul_(momentum/3).add_(buf_tmp, alpha=1 - momentum/3)
                #     buf_tmp = torch.clone(dot_p_buf).detach()
                # # dot_p_buf.mul_(momentum*0).add_(sum_dp, alpha=1 - momentum*0)
            
            # if nesterov:
            #     dp_k = dp_k.add(sum_dp, alpha=momentum)
            # else:
            #     dp_k = sum_dp
            else:
                # print("SGD")
                alpha = lr if maximize else -lr
                # print('alpha',alpha)        
                # p_k.add_(dp_k, alpha=alpha)
                p_k = p_k.add(dp_k, alpha=alpha)


def _multi_tensor_gradFlowAsError(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       para_buffer_list: List[Optional[Tensor]],
                       p_km1_list: List[Optional[Tensor]],
                       dp_km1_list: List[Optional[Tensor]],

                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    if len(params) == 0:
        return

    if has_sparse_grad is None:
        has_sparse_grad = any([grad.is_sparse for grad in d_p_list])

    if weight_decay != 0:
        d_p_list = torch._foreach_add(d_p_list, params, alpha=weight_decay)

    if momentum != 0:
        bufs = []
        bufs2 = []

        all_states_with_momentum_buffer = True
        all_states_with_para_buffer = True

        for i in range(len(momentum_buffer_list)):
            if momentum_buffer_list[i] is None:
                all_states_with_momentum_buffer = False
                break
            else:
                bufs.append(momentum_buffer_list[i])

        if all_states_with_momentum_buffer:
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, d_p_list, alpha=1 - dampening)
        else:
            bufs = []
            for i in range(len(momentum_buffer_list)):
                if momentum_buffer_list[i] is None:
                    sum_dp = momentum_buffer_list[i] = torch.clone(d_p_list[i]).detach()
                else:
                    sum_dp = momentum_buffer_list[i]
                    sum_dp.mul_(momentum).add_(d_p_list[i], alpha=1 - dampening)

                bufs.append(sum_dp)

        for i in range(len(para_buffer_list)):
            if para_buffer_list[i] is None:
                all_states_with_para_buffer = False
                break
            else:
                bufs2.append(para_buffer_list[i])

        if all_states_with_para_buffer:
            torch._foreach_mul_(bufs2, momentum)
            torch._foreach_add_(bufs2, bufs, alpha=1 - dampening)
        else:
            bufs2 = []
            for i in range(len(para_buffer_list)):
                if para_buffer_list[i] is None:
                    dot_p_buf = para_buffer_list[i] = torch.clone(bufs[i]).detach()
                else:
                    dot_p_buf = para_buffer_list[i]
                    nested = 10
                    for i in range(0, nested):
                        dot_p_buf.mul_(momentum/3).add_(bufs[i], alpha=1 - momentum/3)

                bufs2.append(dot_p_buf)

        if nesterov:
            torch._foreach_add_(d_p_list, bufs2, alpha=momentum)
        else:
            d_p_list = bufs2

    alpha = lr if maximize else -lr
    if not has_sparse_grad:
        torch._foreach_add_(params, d_p_list, alpha=alpha)
    else:
        # foreach APIs dont support sparse
        for i in range(len(params)):
            params[i].add_(d_p_list[i], alpha=alpha)  