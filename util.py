import torch


def report_net_param(model):
    param_avg, grad_avg = .0, .0
    param_max, grad_max = None, None
    param_groups_count = .0
    grad_groups_count = .0
    for p in model.parameters():
        if p.grad is not None:
            grad_avg += p.grad.abs().mean()
            g_max = p.grad.abs().max()
            grad_max = max(g_max, grad_max) if grad_max else g_max
            grad_groups_count += 1

        param_avg += p.abs().mean()
        p_max = p.abs().max()
        param_max = max(p_max, param_max) if param_max else p_max
        param_groups_count += 1

    param_avg = param_avg / param_groups_count
    grad_avg = grad_avg / grad_groups_count
    return param_max.item(), param_avg.item(), grad_max.item(), grad_avg.item()
