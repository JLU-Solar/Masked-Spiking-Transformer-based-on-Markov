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
                 t: int = 32,  # 父类量化步数
                 group: int = 8,  # G
                 init_keep: float = None,  # 初始保留概率(默认 1/G)
                 tau: float = 0.5,  # Concrete 温度
                 concrete: bool = False,
                 eps: float = 1e-6) -> None:
        super().__init__(up=up,
                         t=t)
        if up is not None and t is not None:
            self.inner = MyFloor(up=up, t=t)
        else:
            self.inner = TCL()
        # 注意，probs 是保留的概率。
        self.probs = nn.Parameter(1. / group * torch.ones(t, group),
                                  requires_grad=True)
        self.sig = nn.Sigmoid()
        self.group = group
        self.concrete = concrete  # 控制概率采样方式。
        self.eps = eps  # 防止分母为 0 的。
        self.tau = tau  # 温度系数。

        # “学习 logit”，在第一次 forward 根据 (T,G) 懒初始化参数
        self._logit_q = None
        self._initialized = False
        self._init_keep = init_keep  # 记住意图；真正大小在 lazy init 时决定

    def logit(self,
              p: Tensor) -> Tensor:
        eps = self.eps
        p = torch.clamp(p, eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)

    def _lazy_init(self, T: int, device, dtype):
        if self._initialized:
            # 如果已经初始化过，就不再执行初始化
            return

        # 只有第一次 forward 时初始化
        if self.logit_q is None:
            q0 = (1.0 / self.group) if self._init_keep is None else self._init_keep
            init = self.logit(torch.tensor(q0, dtype=dtype, device=device))
            param = init.expand(T, self.group).clone().contiguous()
            self.logit_q = nn.Parameter(param, requires_grad=True)
            self.register_parameter("logit_q", self.logit_q)

            # 标记为已初始化, 后续不再执行初始化。
            self._initialized = True

        elif self.logit_q.shape != (T, self.group):
            # 这里选择抛错；也可以改成重建参数或按需 pad/截断
            raise ValueError(f"logit_q shape {tuple(self.logit_q.shape)} != (T,G)=({T},{self.group})")

    def forward(self, x: Tensor) -> Tensor:
        # 先经过父类的量化步骤。形状是（T,B,C,H,W).
        y = self.inner(x)
        T, B, C, H, W = y.shape

        assert T == self.T, f"T 不匹配: 量化输出 y 的 T = {T}, 模块定义的 T={self.T}"
        assert C % self.G == 0, f"C={C} 不能被 G={self.group} 整除"

        perG: int = C // self.group

        self._lazy_init(T, x.device, x.dtype)

        # 概率 (T,G) → 广播到 (T,1,G,1,1,1)
        probs = self.sig(self.probs)  # (T,G)
        probs = probs[:, None, :, None, None, None, None]  # (T,1,G,1,1,1)

        yg = y.view(T, B, self.group, perG, H, W)

        # region === 使用 Concrete (Gumbel-Sigmoid) 采样 ===
        # 这是 连续化近似伯努利采样 的方式，也叫 Concrete dropout 或 Gumbel-Sigmoid reparameterization trick。
        if self.concrete:
            # Concrete/Gumbel-Sigmoid 连续掩码
            u = torch.rand_like(yg)
            noiseG = torch.log(u) - torch.log1p(-u)
            probs = torch.clamp(probs, self.eps, 1 - self.eps)
            probs = torch.log(probs) - torch.log1p(-probs)  # logit(probs)
            mask = torch.sigmoid((probs + noiseG) / self.tau)  # (0,1)
        # endregion

        # region === 直通概率 ===
        else:
            # 硬伯努利 + 直通估计
            with torch.no_grad():
                hard = torch.bernoulli(probs.expand_as(yg))
            mask = (hard - probs).detach() + probs  # STE
        # endregion

        out_g = yg * mask / torch.clamp(probs, min=self.eps)
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
