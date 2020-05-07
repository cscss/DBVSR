import torch.nn as nn


class FCNNET(nn.Module):
    def __init__(self, args):
        super(FCNNET, self).__init__()
        self.args = args
        self.kernel_size = self.args.kernel_size
        self.num_input_channels = self.kernel_size[0] * self.kernel_size[1]
        self.num_output_channels = self.kernel_size[0] * self.kernel_size[1]
        self.fcn_number = self.args.fcn_number

        if self.fcn_number == 1:
            self.lineLayer1 = nn.Linear(self.num_input_channels, self.num_output_channels, bias=True)
            self.softLayer = nn.Softmax()

        elif self.fcn_number == 2:
            self.num_hidden = 1000
            self.lineLayer1 = nn.Linear(self.num_input_channels, self.num_hidden, bias=True)
            self.reluLayer = nn.ReLU()
            self.lineLayer2 = nn.Linear(self.num_hidden, self.num_output_channels, bias=True)
            self.softLayer = nn.Softmax()

    def forward(self, x):

        if self.fcn_number == 1:
            lineLayer1_out = self.lineLayer1(x)
            output = self.softLayer(lineLayer1_out)

        elif self.fcn_number == 2:
            lineLayer1_out = self.lineLayer1(x)
            reluLayer_out = self.reluLayer(lineLayer1_out)
            lineLayer2_out = self.lineLayer2(reluLayer_out)
            output = self.softLayer(lineLayer2_out)

        out_k = output.view(-1, 1, self.kernel_size[0], self.kernel_size[1])

        return out_k

