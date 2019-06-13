import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
#generate data
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(3)+0.2*torch.rand(x.size())#add random noise

#draw
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_feature=1,n_hidden=50,n_output=1)

print(net)

optimizer = torch.optim.SGD(net.parameters(),lr=0.2)
loss_func = torch.nn.MSELoss()

plt.ion() 

for t in range(500):
    prediction = net(x)
    loss = loss_func(prediction,y)
    optimizer.zero_grad()#清空上一步残余的更新参数
    loss.backward()
    optimizer.step()
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0.5, 0.5, 'generation=%d' % t, fontdict={'size': 20, 'color':  'blue'})       
        plt.pause(0.1)
plt.ioff()
plt.show()