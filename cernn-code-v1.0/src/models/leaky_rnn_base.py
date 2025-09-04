import math
import torch
import torch.nn as nn


class RNNCell_base(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity,
        w_rec_init,
        noise,
        bias,
        decay,
        sigma_rec,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.noise = noise
        self.bias = bias  # TODO we don't have option for no bias
        self.decay = decay
        self.sigma_rec = sigma_rec

        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(w_rec_init, nonlinearity)

        if nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == "leaky_relu":
            self.nonlinearity = nn.LeakyReLU()
        elif nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "softplus":
            self.nonlinearity = nn.Softplus()
        else:
            raise NotImplementedError

    def reset_parameters(self, w_rec_init, nonlinearity):
        """Initialize the weights and biases of the RNN cell
        weights are initialised according to the method specified in w_rec_init.
        Biases are initiased to zero by default, unless activation = relu
        or init = kaiming"""

        if self.bias is not None:
            if nonlinearity == "relu":
                nn.init.normal_(self.bias, 0, 0.001)
            else:
                nn.init.zeros_(self.bias)

        # init input weights with normal distribution with mean 0 and std sqrt(1/n)
        nn.init.normal_(self.weight_ih, 0, math.sqrt(1 / self.input_size))

        if w_rec_init == "randortho":
            nn.init.orthogonal_(self.weight_hh)
        if w_rec_init == "diag":
            nn.init.eye_(self.weight_hh)
            self.weight_hh.data *= 0.5
        elif w_rec_init == "randn":
            nn.init.normal_(self.weight_hh, 0, math.sqrt(1 / self.hidden_size))
        elif w_rec_init == "randu":
            nn.init.uniform_(
                self.weight_hh,
                -math.sqrt(1 / self.hidden_size),
                math.sqrt(1 / self.hidden_size),
            )
        elif w_rec_init == "kaiming":
            """kaiming works better with relu or leaky relu"""
            nn.init.kaiming_uniform_(
                self.weight_hh,
                a=math.sqrt(5),
                nonlinearity=nonlinearity,  # TODO: doesn't work for softplus
            )
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_hh)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

        elif w_rec_init == "xavier":
            """xavier works less well with relu so relu has
            been changed to tanh earlier in the code"""
            nn.init.xavier_uniform_(
                self.weight_hh, gain=nn.init.calculate_gain(nonlinearity)
            )
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_hh)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def leaky_rnn_step(self, input, hidden, w_hh_mask=None, weight_hh=None, decay=None):

        if weight_hh is None:
            weight_hh = self.weight_hh
        if decay is None:
            decay = self.decay

        if self.noise == True:
            noise = torch.normal(
                0, self.sigma_rec, size=hidden.size(), device=hidden.device
            )
        else:
            noise = 0

        if w_hh_mask is not None:
            effective_w_hh = weight_hh.t() * w_hh_mask
        else:
            effective_w_hh = weight_hh.t()

        activity = self.nonlinearity(
            input + hidden @ effective_w_hh + self.bias + noise
        )

        out = (1 - decay) * hidden + self.decay * activity
        return out


class RNNLayer(nn.Module):
    def __init__(self, rnncell, *args):
        super().__init__()
        self.rnncell = rnncell

    def forward(self, input, hidden_init):
        inputs = input.unbind(0)  # inputs has dimension [Time, batch, n_input]
        hidden = hidden_init[0].to(
            input
        )  # initial state has dimension [1, batch, n_input]
        outputs = []
        for i in range(len(inputs)):  # looping over the time dimension
            hidden = self.rnncell(inputs[i], hidden)
            hidden = hidden.to(input)
            outputs += [hidden]  # vanilla RNN directly outputs the hidden state
        return torch.stack(outputs), hidden
