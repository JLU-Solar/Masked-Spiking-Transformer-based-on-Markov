import logging
from pprint import pformat

import torch
from spikingjelly.clock_driven import neuron
from torch import nn, Tensor
from torch.autograd import Function


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class ScaledNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)

    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x) * 0.5)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale

    def reset(self):
        self.t = 0
        self.neuron.reset()


class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply


class ShiftNeuron(nn.Module):
    def __init__(self, scale=1., alpha=1 / 50000):
        super().__init__()
        self.alpha = alpha
        self.vt = 0.
        self.scale = scale
        self.neuron = neuron.IFNode(v_reset=None)

    def forward(self, x):
        x = x / self.scale
        x = self.neuron(x)
        return x * self.scale

    def reset(self):
        if self.training:
            self.vt = self.vt + self.neuron.v.reshape(-1).mean().item() * self.alpha
        self.neuron.reset()
        if self.training == False:
            self.neuron.v = self.vt


class MyFloor(nn.Module):
    def __init__(self, up=2., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)

        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x * self.t + 0.5) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x


class MyMarkov(nn.Module):
    def __init__(self,
                 up: float = 2.,
                 T: int = 32,  # 父类量化步数
                 group: int = 8,  # G
                 tau: float = 0.5,  # Concrete 温度
                 concrete: bool = False,
                 eps: float = 1e-6,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device(f"cuda")) -> None:
        r"""

        :param up:
        :type up: float
        :param T:
        :type T: int
        :param group:
        :type group: int
        :param init_keep:
        :type init_keep: float
        :param tau:
        :type tau: float
        :param concrete:
        :type concrete: bool
        :param eps:
        :type eps: float
        """
        super().__init__()
        if up is not None and T is not None:
            self.inner = MyFloor(up=up, t=T)
        else:
            self.inner = TCL()

        self.sig = nn.Sigmoid()
        self.group = group
        self.T = T
        self.concrete = concrete  # 控制概率采样方式。
        self.eps = eps  # 防止分母为 0 的。
        self.tau = tau  # 温度系数。

        # 初始化 probs (T, G), 注意，probs 是保留的概率。
        # 返回的是 Logits 实数概率，方便优化。取值范围为实数域，用之前，先映射回 0-1 范围（用 sigmoid 函数）
        self._init_probs(device=device,
                         dtype=dtype)

    def logit(self,
              p: Tensor) -> Tensor:
        r"""
        将给定的输入张量 p（通常是概率值）进行 logit 转换。
        Logit 是概率到对数几率（log-odds）的映射，是一种常用于概率建模和分类模型的转换方式。
        概率值（介于 0 和 1 之间）被映射到一个无界的实数范围。
        :param p:
        :type p: Tensor
        :return:
        :rtype: Tensor
        """
        eps = self.eps
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _init_probs(self, device, dtype) -> None:
        r"""
        :param device:
        :type device: torch.device
        :param dtype:
        :type dtype: torch.dtype
        :return:
        :rtype: None
        """
        q0 = 1.0 / self.group * self.T
        init = self.logit(torch.tensor(q0, dtype=dtype, device=device))
        param = init.expand(self.T, self.group).clone().contiguous()
        self.probs = nn.Parameter(param, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        # 先经过已有的量化步骤。形状是（T,B,C,H,W).
        y = self.inner(x)
        T, B, C, H, W = y.shape

        assert T == self.T, f"T 不匹配: 量化输出 y 的 T = {T}, 模块定义的 T={self.T}"
        assert C % self.G == 0, f"C={C} 不能被 G={self.group} 整除"

        perG: int = C // self.group

        probs = self.probs[:, None, :, None, None, None, None]  # (T,1,G,1,1,1)

        yg = y.view(T, B, self.group, perG, H, W)

        # region === Concrete (Gumbel-Sigmoid) 采样 ===
        # 连续化近似伯努利采样，也叫 Concrete dropout 或 Gumbel-Sigmoid reparameterization trick。
        if self.concrete:
            # Concrete/Gumbel-Sigmoid 连续掩码
            u = torch.rand_like(yg)
            noiseG = torch.log(u) - torch.log1p(-u)
            mask = torch.sigmoid((probs + noiseG) / self.tau)  # (0,1)
        # endregion

        # region === 直通概率 ===
        else:
            # 现将 Logits 类型的实数概率，映射回 0-1 之间的常规概率。
            # 概率 (T,G) → 广播到 (T,1,G,1,1,1)
            probs = self.sig(self.probs)  # (T,G)
            probs = torch.clamp(probs, self.eps, 1 - self.eps)  # 防止超出 [0, 1] 的范围
            # 硬伯努利 + 直通估计
            with torch.no_grad():
                hard = torch.bernoulli(probs.expand_as(yg))
            mask = (hard - probs).detach() + probs  # STE
        # endregion

        out_g = yg * mask
        out = out_g.reshape(T, B, C, H, W)
        return out


class MyDarts(nn.Module):
    def __init__(self,
                 nameLogger: str,
                 up: float = 2.,
                 T: int = 32,  # 父类量化步数
                 group: int = 8,  # G
                 tau: float = 0.5,  # Concrete 温度
                 concrete: bool = False,
                 eps: float = 1e-6,
                 init_probs_ratio: float = 0.5,
                 if_learn_top_probs: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = torch.device(f"cuda")
                 ):
        super().__init__()

        if up is not None and T is not None:
            self.inner = MyFloor(up=up, t=T)
        else:
            self.inner = TCL()
        self.logger = logging.getLogger(nameLogger)
        self.probs: nn.Parameter = None
        self.sig = nn.Sigmoid()
        self.group = group
        self.if_learn_top_probs = if_learn_top_probs
        self.T = T
        self.concrete = concrete  # 控制概率采样方式。
        self.eps = eps  # 防止分母为 0 的。
        self.tau = tau  # 温度系数。
        self.top_probs_ratio: nn.Parameter | torch.Tensor = None
        self.init_probs_ratio = torch.tensor(init_probs_ratio,
                                             dtype=dtype,
                                             device=device)
        # 初始化 probs (T, G), 注意，probs 是保留的概率。
        # 返回的是 Logits 实数概率，方便优化。取值范围为实数域，用之前，先映射回 0-1 范围（用 sigmoid 函数）
        self._init_probs(device=device,
                         dtype=dtype)

    def logit(self,
              p: Tensor) -> Tensor:
        r"""
        将给定的输入张量 p（通常是概率值）进行 logit 转换。
        Logit 是概率到对数几率（log-odds）的映射，是一种常用于概率建模和分类模型的转换方式。
        概率值（介于 0 和 1 之间）被映射到一个无界的实数范围。
        log(p) - log(1 - p) 这个操作是单调递增的，因此会保留输入 p 中元素之间的大小关系。
        :param p:
        :type p: Tensor
        :return:
        :rtype: Tensor
        """
        eps = self.eps
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _init_top_probs_ratio(self):
        if self.if_learn_top_probs:
            self.top_probs_ratio = nn.Parameter(self.init_probs_ratio,
                                                requires_grad=True)
            self.logger.debug(f"概率的 topk 百分比可学习")
        else:
            self.top_probs_ratio = self.init_probs_ratio
            self.logger.debug(f"概率的 topk 百分比不可学习")

    def _init_probs(self, device, dtype) -> None:
        r"""
        :param device:
        :type device: torch.device
        :param dtype:
        :type dtype: torch.dtype
        :return:
        :rtype: None
        """
        q0 = 0.5
        init = self.logit(torch.tensor(q0, dtype=dtype, device=device))
        param = init.expand(self.T, self.group).clone().contiguous()
        self.probs = nn.Parameter(param, requires_grad=True)
        self.logger.debug(f"初始概率：{pformat(self.probs)}")

    def top_probs_mask(self, ):
        top_probs_ratio = torch.clamp(self.top_probs_ratio, 0, 1)

        # 展平概率张量 (T*G, )
        flat = self.probs.flatten()
        N = flat.numel()  # 总的概率个数。

        # 将百分比的 top，变为 topk（整数） 问题。
        k = max(1, int(top_probs_ratio * N))  # 至少保留 1 个概率，不然可能全都是0了。

        # 找到第 k 大的值作为阈值
        # 当所有的元素相等时，topk 会返回前 k 个最大值的位置，而这些值是相同的。
        # 所以 topk 返回的是这些相同值的任意 k 个位置。
        threshold = torch.topk(flat, k).values.min()

        # 生成 mask
        mask = (self.probs >= threshold).float()
        self.logger.debug(f"按照：{pformat(mask)} 对概率张量进行掩码。")
        # 应用 mask
        self.probs = self.probs * mask
        self.logger.debug(f"掩码更新后的概率为：{pformat(self.probs)}")

    def forward(self, x: Tensor) -> Tensor:
        # 先经过已有的量化步骤。形状是（T,B,C,H,W).
        y = self.inner(x)
        T, B, C, H, W = y.shape

        assert T == self.T, f"T 不匹配: 量化输出 y 的 T = {T}, 模块定义的 T={self.T}"
        assert C % self.G == 0, f"C={C} 不能被 G={self.group} 整除"
        perG: int = C // self.group

        self.top_probs_mask()
        probs = self.probs[:, None, :, None, None, None, None]  # (T,1,G,1,1,1)

        yg = y.view(T, B, self.group, perG, H, W)

        # region === Concrete (Gumbel-Sigmoid) 采样 ===
        # 连续化近似伯努利采样，也叫 Concrete dropout 或 Gumbel-Sigmoid reparameterization trick。
        if self.concrete:
            # Concrete/Gumbel-Sigmoid 连续掩码
            u = torch.rand_like(yg)
            noiseG = torch.log(u) - torch.log1p(-u)
            mask = torch.sigmoid((probs + noiseG) / self.tau)  # (0,1)
        # endregion

        # region === 直通概率 ===
        else:
            # 现将 Logits 类型的实数概率，映射回 0-1 之间的常规概率。
            # 概率 (T,G) → 广播到 (T,1,G,1,1,1)
            probs = self.sig(self.probs)  # (T,G)
            probs = torch.clamp(probs, self.eps, 1 - self.eps)  # 防止超出 [0, 1] 的范围
            # 硬伯努利 + 直通估计
            with torch.no_grad():
                hard = torch.bernoulli(probs.expand_as(yg))
            mask = (hard - probs).detach() + probs  # STE
        # endregion

        out_g = yg * mask
        out = out_g.reshape(T, B, C, H, W)
        return out


class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.up - x
        x = self.relu(x)
        x = self.up - x
        return x


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
