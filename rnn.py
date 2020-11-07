import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class MyModel(nn.Module):
    def __init__(self, device, input_size=27, hidden_size=256, output_size_rnn=128, num_classes=27):
        super(MyModel, self).__init__()

        self.device = device
        self.rnn_src = RNN(input_size, hidden_size, output_size_rnn)
        self.rnn_targ = RNN(input_size, hidden_size, output_size_rnn)
        self.dense = nn.Linear(output_size_rnn*2, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input1, input2):
        hidden1 = self.rnn_src.initHidden().to(self.device)
        hidden2 = self.rnn_targ.initHidden().to(self.device)
        for i in range(input1.size()[0]):
            output1, hidden1 = self.rnn_src(input1[i], hidden1)
        for i in range(input2.size()[0]):
            output2, hidden2 = self.rnn_targ(input2[i], hidden2)
        output = self.dense(torch.cat((output1, output2), -1))
        output = self.softmax(output)

        return output
