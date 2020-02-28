import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class AttnEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, time_step):
        super(AttnEncoder, self).__init__()
        # feature shape
        self.input_size = input_size
        # hidden state shape/embedding shape
        self.hidden_size = hidden_size
        # window size/lag
        self.T = time_step

        # original LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        # inner tanh linear weighting of first part
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=self.T)
        # inner tanh linear weighting of second part
        self.attn2 = nn.Linear(in_features=self.T, out_features=self.T)
        # tanh layer
        self.tanh = nn.Tanh()
        # outer tanh linear weighting
        self.attn3 = nn.Linear(in_features=self.T, out_features=1)


    def forward(self, driving_x):
        batch_size = driving_x.size(0)

        # batch_size * time_step * hidden_size
        code = self.init_variable(batch_size, self.T, self.hidden_size)
        # initialize hidden state
        # single layer LSTM
        h = self.init_variable(1, batch_size, self.hidden_size)
        # initialize cell state
        s = self.init_variable(1, batch_size, self.hidden_size)
        for t in range(self.T):
            # batch_size * input_size * (2 * hidden_size + time_step)
            # h and s from last step t-1
            x = torch.cat((self.embedding_hidden(h), self.embedding_hidden(s)), 2)
            z1 = self.attn1(x)
            # .permute change dimensions
            # driving x: batch_size * time_step * feature_number
            z2 = self.attn2(driving_x.permute(0, 2, 1))
            # x: batch_size * feature_number * time_step
            x = z1 + z2 
            # batch_size * input_size * 1
            z3 = self.attn3(self.tanh(x))
            #print("z3.shape",z3.shape)
            if batch_size > 1:
                attn_w = F.softmax(z3.view(batch_size, self.input_size), dim=1)
            else:
                attn_w = self.init_variable(batch_size, self.input_size) + 1
            # batch_size * input_size
            weighted_x = torch.mul(attn_w, driving_x[:, t, :])
            # renew lstm hidden state
            _, states = self.lstm(weighted_x.unsqueeze(0), (h, s))
            h = states[0]
            s = states[1]
            # encoding result
            # batch_size * time_step * encoder_hidden_size
            code[:, t, :] = h

        return code

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.input_size, 1, 1).permute(1, 0, 2)


class AttnDecoder(nn.Module):

    def __init__(self, code_hidden_size, hidden_size, time_step, case_number,embedding_size,alpha=0.01):
        super(AttnDecoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step
        self.case_number = case_number
        self.embedding_size = embedding_size
        # inner tanh first part
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        # inner tanh second part
        self.attn2 = nn.Linear(in_features=code_hidden_size, out_features=code_hidden_size)
        # tanh
        self.tanh = nn.Tanh()
        # outer tanh 
        self.attn3 = nn.Linear(in_features=code_hidden_size, out_features=1)
        # lstm layer
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        # eq.15
        self.tilde = nn.Linear(in_features=self.code_hidden_size + 1, out_features=1)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        
        self.embedding = nn.Embedding(self.case_number,self.embedding_size)
        self.leakyrelu = nn.LeakyReLU(alpha)
        #self.fc2 = nn.Linear(in_features=hidden_size + embedding_size, out_features=1,bias=False)
        self.fc2 = nn.Linear(in_features=hidden_size + embedding_size, out_features=1,bias=False)

    def forward(self, h, y_seq, x_id):
        batch_size = h.size(0)
        d = self.init_variable(1, batch_size, self.hidden_size)
        s = self.init_variable(1, batch_size, self.hidden_size)
        ct = self.init_variable(batch_size, self.hidden_size)

        for t in range(self.T):
            # batch_size * time_step * (encoder_hidden_size + decoder_hidden_size)
            x = torch.cat((self.embedding_hidden(d), self.embedding_hidden(s)), 2)
            # batch_size * time_step * encoder_hidden_size
            z1 = self.attn1(x)
            # batch_size * time_step * encoder_hidden_size
            z2 = self.attn2(h)
            # batch_size * time_step * encoder_hidden_size
            x = z1 + z2
            # batch_size * time_step * 1
            z3 = self.attn3(self.tanh(x))
            # print("z3.shape",z3.shape)
            if batch_size > 1:
                beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            else:
                beta_t = self.init_variable(batch_size, self.code_hidden_size) + 1
            #beta_t = F.softmax(z3.view(batch_size, -1), dim=1)
            # batch_size * time_step
            # beta_t
            # batch_size * encoder_hidden_size
            #print("beta_t is {} ".format(beta_t.unsqueeze(1)))
            #print("the h is {} ".format(h))
            ct = torch.bmm(beta_t.unsqueeze(1), h).squeeze(1)
            if t < self.T :#- 1:
                yc = torch.cat((y_seq[:, t].unsqueeze(1), ct), dim=1)
                # batch_size * 1
                y_tilde = self.tilde(yc)
                # print("y_tilde.shape",y_tilde.shape)
                _, states = self.lstm(y_tilde.unsqueeze(0), (d, s))
                d = states[0]
                s = states[1]
        # batch_size * 1
        #print("d.shape:{}, d.squeeze.shape: {}, ct.shape:{}".format(d.shape,s.squeeze(0).shape,ct.shape))
        decoder_out = self.fc1(torch.cat((d.squeeze(0), ct), dim=1))
        embed = self.embedding(x_id)
        embed = self.leakyrelu(embed)
        #print("decoder_out.shape: {}, embed.shape: {}".format(decoder_out.shape,embed.shape))
        linear_in = torch.cat([decoder_out,embed],1)
        y_res = self.fc2(linear_in)
        return y_res

    def init_variable(self, *args):
        zero_tensor = torch.zeros(args)
        if torch.cuda.is_available():
            zero_tensor = zero_tensor.cuda()
        return Variable(zero_tensor)

    def embedding_hidden(self, x):
        return x.repeat(self.T, 1, 1).permute(1, 0, 2)