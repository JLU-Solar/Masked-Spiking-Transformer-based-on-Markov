# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from typing import Union, Tuple, Type, Dict

from torch import optim as optim, nn

from models.mst import MaskedSpikingTransformer
from modules_QCFS import MyDarts, MyMarkov

try:
    from apex.optimizers import FusedAdam, FusedLAMB
except:
    FusedAdam = None
    FusedLAMB = None
    print("To use FusedLAMB or FusedAdam, please install apex.")


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False


def regular_set(model, paras=([], [], [], [])):
    for n, module in model._modules.items():
        if isActivation(module.__class__.__name__.lower()) and hasattr(module, "up"):
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[0].append(para)
                    print(f"{name} 0")
        elif 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[2].append(para)
                    print(f"{name} 2")
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                if not para.requires_grad:
                    continue  # frozen weights
                if name.endswith(".bias"):
                    paras[3].append(para)
                    print(f"{name} 3")
                else:
                    paras[1].append(para)
                    print(f"{name} 1")
    return paras


def build_optimizer(config,
                    model: MaskedSpikingTransformer) -> None:
    r"""
    Build optimizer, set weight decay of normalization to 0 by default.
    :param config:
    :type config:
    :param model:
    :type model:
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    # para1, para2, para3, para4 = regular_set(model)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()

    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters,
                                eps=config.TRAIN.OPTIMIZER.EPS,
                                betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR,
                                weight_decay=config.TRAIN.WEIGHT_DECAY)
        # optimizer = optim.AdamW([
        #                         {'params': para1, 'weight_decay': 5e-4, "lr" : 1e-5 }, 
        #                         {'params': para2, 'weight_decay': config.TRAIN.WEIGHT_DECAY}, 
        #                         {'params': para3, 'weight_decay': config.TRAIN.WEIGHT_DECAY},
        #                         {'params': para4, 'weight_decay': 0.},
        #                         ], eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
        #                         lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_adam':
        optimizer = FusedAdam(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    elif opt_lower == 'fused_lamb':
        optimizer = FusedLAMB(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError(f"Unknown optimizer identifyer {opt_lower}")

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    up = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        # if (len(param.shape) == 1 and 'up' not in name) or name.endswith(".bias") or (name in skip_list) or \
        #         check_keywords_in_name(name, skip_keywords):
        if (len(param.shape) == 1) or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        elif 'up' in name:
            up.append(param)
            # print(f"{name} threshold")               
        else:
            has_decay.append(param)
    # return [{'params': has_decay},
    #         {'params': up, 'weight_decay': 5e-4},
    #         {'params': no_decay, 'weight_decay': 0.}]
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


Mode = Union[str]  # "all" | "none" | "only_types" | "except_types"


def set_requires_grad_switch(
        model: nn.Module,
        mode: Mode,
        target_types: Tuple[Type[nn.Module], ...] = (MyDarts, MyMarkov),
) -> Dict[str, int]:
    r"""
    统一开关：设置 model 各参数的 requires_grad。

    参数
    :param model:  MaskedSpikingTransformer 或任意 nn.Module。
    :type model: nn.Module

    :param mode:- "all"        -> 1) 全部参数可学习 (requires_grad=True)
                - "only_ZJ"    -> 2) 仅 target_types 模块的参数可学习，其他冻结
                - "except_ZJ"  -> 3) 冻结 target_types 模块的参数，其他可学习
                - "none"       -> 4) 全部参数冻结 (requires_grad=False)
    :type mode : str
    :param target_types:目标模块类型集合，默认 (MyDarts, MyMarkov)。
    :type target_types : Tuple[type]
    :return: stats:{'set_true': 数量, 'set_false': 数量, 'total': 总参数量}
    :rtype: Dict[str, int]
    """

    assert mode in {"all", "none", "only_ZJ", "except_ZJ"}, f"Unsupported mode: {mode}"

    # 去重集合，防止共享参数重复统计/设置
    seen = set()

    def _mark_params(mod: nn.Module, flag: bool):
        r"""
        【安全地、无重复地】 修改当前模块中所有参数的 requires_grad，
        并记录统计信息。
        :param mod:
        :param flag:
        :return:
        """
        nonlocal seen_true, seen_false

        for p in mod.parameters(recurse=False):  # recurse=False 只处理当前模块直接持有的参数
            pid = id(p)
            if pid in seen:
                # 已处理过的共享参数，仅在需要时覆盖
                p.requires_grad = flag if mode in {"all", "none"} else p.requires_grad if pid in seen else flag
                continue
            seen.add(pid)
            p.requires_grad = flag
            if flag:
                seen_true += 1
            else:
                seen_false += 1

    total = sum(1 for _ in model.parameters())
    seen_true = 0
    seen_false = 0

    if mode == "all":
        # 1) 全训练
        for m in model.modules():
            _mark_params(m, True)

    elif mode == "none":
        # 4) 全冻结
        for m in model.modules():
            _mark_params(m, False)

    elif mode == "only_types":
        # 2) 仅训练张嘉
        # 先全 False
        for m in model.modules():
            _mark_params(m, False)

        # 再将张嘉类型（及其子模块的直接) 参数设置为 True
        for m in model.modules():
            if isinstance(m, target_types):
                # 目标块及其子树里的“直接参数”都设置 True：
                # 遍历子树时使用 modules()，每个节点调用 _mark_params(recurse=False)
                for sub in m.modules():
                    _mark_params(sub, True)

    elif mode == "except_types":
        # 3) 只有张嘉不训练
        # 先全 True
        for m in model.modules():
            _mark_params(m, True)
        # 再把目标类型（及其子模块）置 False
        for m in model.modules():
            if isinstance(m, target_types):
                for sub in m.modules():
                    _mark_params(sub, False)

    return {"set_true": seen_true, "set_false": seen_false, "total": total}


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
