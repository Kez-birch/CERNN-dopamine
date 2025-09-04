import torch
from torch import nn

from src.models.leaky_rnn_base import RNNCell_base, RNNLayer


class LeakyRNNModel(nn.Module):
    def __init__(
        self,
        n_input,
        n_rnn,
        n_output,
        dt,
        tau,
        noise,
        w_rec_init,
        sigma_rec,
        activation,
    ):
        super().__init__()
        decay = dt / tau
        self.n_rnn = n_rnn
        rnncell = leaky_RNNCell(
            n_input,
            n_rnn,
            activation,
            decay,
            w_rec_init,
            True,  # bias
            noise,
            sigma_rec,
        )
        self.rnn = RNNLayer(rnncell)

        self.readout = nn.Linear(n_rnn, n_output, bias=False)

    def init_hidden_(self, x, h0):
        max_steps = 100
        stable_count = 0
        h0 = h0.to(self.rnn.rnncell.weight_hh.device)
        for _ in range(max_steps):
            _, h_next = self.rnn(x[:2, 0:1, :], h0)
            if torch.allclose(h_next, h0, atol=0.1):
                stable_count += 1
                if stable_count >= 4:
                    return h0
            else:
                stable_count = 0
            h0 = h_next
        return h0

    def forward(self, x):
        hidden0 = torch.zeros([1, x.shape[1], self.n_rnn])
        hidden0 = self.init_hidden_(x, hidden0)
        hidden, _ = self.rnn(x, hidden0)

        output = self.readout(hidden)
        return output, hidden


class leaky_RNNCell(RNNCell_base):
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity,
        decay,
        w_rec_init,
        bias=True,
        noise=True,
        sigma_rec=0.05,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity=nonlinearity,
            w_rec_init=w_rec_init,
            bias=bias,
            noise=noise,
            decay=decay,
            sigma_rec=sigma_rec,
        )

    def forward(self, input, hidden):
        input = input @ self.weight_ih.t()
        out = self.leaky_rnn_step(input, hidden)
        return out
