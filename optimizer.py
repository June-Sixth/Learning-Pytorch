import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable

#hyper parameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x=torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(*x.size()))#add random noise

# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

#put dataset into torch dataset
troch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(dataset=troch_dataset,batch_size = BATCH_SIZE,shuffle = True,num_workers = 2)

#def network
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

if __name__=='__main__':
    #nets
    net_SGD = Net(1,20,1)
    net_Momentum = Net(1,20,1)
    net_RMSprop = Net(1,20,1)
    net_Adam = Net(1,20,1)
    nets = [net_SGD,net_Momentum,net_RMSprop,net_Adam]

    #optimizers
    opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    #记录损失
    losses_his = [[],[],[],[]]

    for epoch in range(EPOCH):
        print(epoch)
        for step,(batch_x,batch_y) in enumerate(loader):
            b_x = Variable(x)
            b_y = Variable(y)
            for net, opt, l_his in zip(nets,optimizers,losses_his):
                prediction = net(b_x)
                loss = loss_func(prediction,b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

        lables = ['SGD','Momentum','RMSprop','Adam']
        for i,l_his in enumerate(losses_his):
            plt.plot(l_his,label=lables[i])
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.ylim((0,0.2))
        plt.show()