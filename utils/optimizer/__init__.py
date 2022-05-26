from .ranger import Ranger
import torch.optim as opt
import sys


def get_optimizer(model_named_params, opt_params):
    # create wd-based groups of learnable parameters
    decay, no_decay = [], []
    for name, param in model_named_params:
        if not param.requires_grad:
            continue
        elif opt_params['weight decay weights only'] and \
                (len(param.shape) == 1 or name.endswith(".bias")):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay, 'weight_decay': opt_params['weight decay']}]
    if no_decay:
        params.append({'params': no_decay, 'weight_decay': 0.0})
    # set optimizer
    if opt_params['optimizer']['name'] == 'Sgd':
        optimizer = opt.SGD(params,
                            opt_params['initial learning rate'],
                            momentum=opt_params['momentum'])
    elif opt_params['optimizer']['name'] == 'Ranger':
        optimizer = Ranger(params,
                           opt_params['initial learning rate'],
                           opt_params['optimizer']['params']['alpha'],
                           opt_params['optimizer']['params']['inter iters'],
                           opt_params['optimizer']['params']['sma threshold'])
    elif opt_params['optimizer']['name'] == 'AdamW':
        optimizer = opt.AdamW(params, lr=opt_params['initial learning rate'],
                              amsgrad=False)
    else:
        sys.exit('Unsupported optimizer type: %s' % opt_params['type'])
    return optimizer


def get_scheduler(optimizer, params):
    if params['mode'] == 'step':
        scheduler = opt.lr_scheduler.StepLR(optimizer,
                                            step_size=params['epoch step'],
                                            gamma=params['power'])
    else:
        sys.exit('Unsupported lr scheduler type: %s' % params['type'])
    return scheduler
