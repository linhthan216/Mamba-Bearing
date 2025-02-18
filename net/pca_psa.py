import torch.nn as nn
import torch
from einops import reduce, rearrange
class PCA(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dw = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding="same", groups=dim)
    self.prob = nn.Softmax(dim=1)

  def forward(self,x):
    c = reduce(x, 'b c h w -> b c', 'mean')
    x = self.dw(x)
    c_ = reduce(x, 'b c h w -> b c', 'mean')
    raise_ch = self.prob(c_ - c)
    att_score = torch.sigmoid(c_ + c_*raise_ch)
    return torch.einsum('bchw, bc -> bchw', x, att_score)

class PSA(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.pw = nn.Conv2d(dim, dim, kernel_size=1)
    self.prob = nn.Softmax2d()

  def forward(self,x):
    s = reduce(x, 'b c w h -> b w h', 'mean')
    xp = self.pw(x)
    s_ = reduce(xp, 'b c w h -> b w h', 'mean')
    raise_sp = self.prob(s_ - s)
    att_score = torch.sigmoid(s_ + s_*raise_sp)
    return torch.einsum('bchw, bwh -> bchw', x, att_score)