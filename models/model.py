import torch
import torch.nn as nn


class pDNN(nn.Module):
    def __init__(self, layers, nodes, dropout, activation, input_size):
        super(pDNN, self).__init__()
        self.layers = layers
        self.nodes = nodes
        self.input_size = input_size
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU(True)

        if type(nodes) == list:
            assert(len(nodes) == layers, "Error in defining layers and nodes")
            self.modules = [
                nn.Linear(input_size, self.nodes[0]), self.activation_fn, nn.Dropout(p=dropout)]
            for i in range(layers-1):
                self.modules.extend([nn.Linear(
                    self.nodes[i], self.nodes[i+1]), self.activation_fn, nn.Dropout(p=dropout)])
            self.modules.append(nn.Linear(self.nodes[-1], 1))
        else:
            self.modules = [
                nn.Linear(input_size, self.nodes), self.activation_fn, nn.Dropout(p=dropout)]
            for i in range(layers-1):
                self.modules.extend(
                    [nn.Linear(self.nodes, self.nodes), self.activation_fn, nn.Dropout(p=dropout)])
            self.modules.append(nn.Linear(self.nodes, 1))

        self.model = nn.Sequential(**self.modules)

    def forward(self, x):
        return self.model(x)
