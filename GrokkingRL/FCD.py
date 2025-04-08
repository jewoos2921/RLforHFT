import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class FCDuelingQ(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super().__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_value = nn.Linear(hidden_dims[-1], 1)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, state):
        x = state

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x,
                             device=self.device,
                             dtype=torch.float32)
            x = x.unsqueeze(0)

        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))

        a = self.output_layer(x)
        val = self.output_value(x).expand_as(a)

        q = val + a - a.mean(dim=1, keepdim=True).expand(a)
        return q
