import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#--------------------------------#
sentences = ['How may I help you?',
             'Can I be of assistance?',
             'May I help you with something?',
             'May I assist you?']

words = dict()
reverse = dict()
i = 0
for s in sentences:
    s = s.replace('?',' <unk>')
    for w in s.split():
        if w.lower() not in words:
            words[w.lower()] = i
            reverse[i] = w.lower()
            i = i + 1

def get_next(model,word_):
    word = word_.lower()
    out = model(Variable(torch.LongTensor([words[word_]])))
    return reverse[int(out.max(dim=1)[1].data)]

def get_next_n(model,word_, n=3):
    print(word_)
    for i in range(0, n):
        word_ = get_next(model,word_)
        print(word_)

class BotBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(words), 10)
        self.rnn = nn.LSTM(10, 20, 2, dropout=0.5)
        self.h = (Variable(torch.zeros(2, 1, 20)), Variable(torch.zeros(2, 1, 20)))
        self.l_out = nn.Linear(20, len(words))
        
    def forward(self, cs):
        inp = self.embedding(cs).view(-1,1,10)
        outp,h = self.rnn(inp, self.h)
        out = F.log_softmax(self.l_out(outp), dim=-1).view(-1, len(words))
        return out

if __name__ == '__main__':
        
    model = BotBrain()

    model.load_state_dict(torch.load('pretrained'))
    get_next_n(model,'<unk>',5)
