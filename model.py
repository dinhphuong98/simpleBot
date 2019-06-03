import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#---------------------------------#
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

class DataGenerator():
    def __init__(self, dset):
        self.dset = dset
        self.len = len(self.dset)
        self.idx = 0
    def __len__(self):
        return self.len
    def __iter__(self):
        return self
    def __next__(self):
        x = Variable(torch.LongTensor([[self.dset[self.idx]]]), requires_grad=False)
        if self.idx == self.len - 1:
            raise StopIteration
        y = Variable(torch.LongTensor([self.dset[self.idx+1]]), requires_grad=False)
        self.idx = self.idx + 1
        return (x, y)

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
    m = BotBrain()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(m.parameters(), lr=0.01)
    plot_loss = []
    plot_epoch = []

    print("training...")
    for epoch in range(0,1000):
        s_loss = 0.0
        gen = DataGenerator([words[word.lower()] for word in ' '.join(sentences).replace('?',' <unk>').split(' ')])
        for x, y in gen:
            m.zero_grad()
            output = m(x)
            loss = criterion(output, y)
            loss.backward()
            s_loss += loss
            optimizer.step()
        
        plot_epoch.append(epoch)
        plot_loss.append(s_loss)

    plt.plot(plot_epoch,plot_loss)
    plt.xlabel("Loss")
    plt.ylabel("epoch")
    plt.show()

    torch.save(m.state_dict(),'pretrained')
