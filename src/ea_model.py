"""EA-LSTM equations 7--12, Kratzert et al. (2019), under our common budget.

https://doi.org/10.5194/hess-23-5089-2019
Static attributes drive only the time-invariant input gate. The f/g/o gates
receive dynamic forcing and previous hidden state. This is not a replication
of the paper's hyperparameters or a parameter-count-matched comparison.
"""
import torch
from torch import nn
from torch.nn import functional as F
from model import LandMemoryLSTM


class EALSTM(nn.Module):
    def __init__(self, dynamic_size, static_size, hidden_size=64, dropout=.2):
        super().__init__()
        if min(dynamic_size, static_size, hidden_size) < 1 or not 0 <= dropout < 1:
            raise ValueError('Invalid EA dimensions/dropout')
        self.dynamic_size, self.static_size, self.hidden_size = dynamic_size, static_size, hidden_size
        self.adapter_mode = 'ea_static_input_gate'  # Non-fused route in shared trainer.
        self.execution_backend = 'reference_manual_loop'
        self.input_gate = nn.Linear(static_size, hidden_size)
        self.dynamic_gates = nn.Linear(dynamic_size, 3*hidden_size)
        self.recurrent_gates = nn.Linear(hidden_size, 3*hidden_size)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.input_gate.bias.zero_()
            self.dynamic_gates.bias.zero_()
            self.dynamic_gates.bias[:hidden_size].fill_(1.)
            self.recurrent_gates.bias.zero_()

    def forward(self, dynamic, static, land=None, *, return_diagnostics=False):
        if dynamic.ndim != 3 or dynamic.shape[2] != self.dynamic_size or min(dynamic.shape[:2]) < 1:
            raise ValueError('Expected nonempty [batch,time,dynamic]')
        if static.shape != (dynamic.shape[0], self.static_size):
            raise ValueError('Invalid static shape')
        if not torch.is_floating_point(dynamic) or static.dtype != dynamic.dtype or static.device != dynamic.device:
            raise ValueError('Inputs must share floating dtype and device')
        # land is intentionally unused: all static information is already in static.
        i = torch.sigmoid(self.input_gate(static))
        terms = self.dynamic_gates(dynamic)
        h = dynamic.new_zeros(dynamic.shape[0], self.hidden_size)
        c = torch.zeros_like(h)
        states, forgets = [], []
        for t in range(dynamic.shape[1]):
            f, g, o = (terms[:,t]+self.recurrent_gates(h)).chunk(3, dim=-1)
            f, g, o = torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f*c+i*g
            h = o*torch.tanh(c)
            if return_diagnostics:
                states.append(h.detach())
                forgets.append(f.detach())
        pred = F.softplus(self.head(self.head_dropout(h)))
        if return_diagnostics:
            return pred, dict(input_gate=i.detach(), hidden_sequence=torch.stack(states,1),
                             forget_gate=torch.stack(forgets,1), last_cell=c.detach())
        return pred


def make_ea(cfg, meta, seed, device):
    """Copy corresponding B0 initial coefficients, not trained B0 weights.

    Retain two f/g/o biases to share B0's optimization parameterization;
    static input-gate bias is the initial sum of B0's two zero biases.
    No dynamic or recurrent input-gate coefficients, nor static f/g/o
    coefficients, exist in EA. All surviving weight slices and the head
    are identical to the same-seed B0 initialization.
    """
    torch.manual_seed(seed)
    d, s, h = len(meta['dynamic_names']), len(meta['static_names']), cfg['model']['hidden_size']
    base = LandMemoryLSTM(d,s,len(meta['land_names']),h,cfg['model']['rank'],
                          cfg['model']['output_dropout'],adapter_mode='none')
    net = EALSTM(d,s,h,cfg['model']['output_dropout'])
    with torch.no_grad():
        net.input_gate.weight.copy_(base.core.weight_ih_l0[:h,d:])
        net.input_gate.bias.copy_(base.core.bias_ih_l0[:h]+base.core.bias_hh_l0[:h])
        net.dynamic_gates.weight.copy_(base.core.weight_ih_l0[h:,:d])
        net.dynamic_gates.bias.copy_(base.core.bias_ih_l0[h:])
        net.recurrent_gates.weight.copy_(base.core.weight_hh_l0[h:])
        net.recurrent_gates.bias.copy_(base.core.bias_hh_l0[h:])
        net.head.load_state_dict(base.head.state_dict())
    net = net.to(device)
    torch.manual_seed(seed+100000)
    return net
