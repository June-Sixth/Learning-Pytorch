import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.out = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):
        x = F.relu(self.hidden(x))     
        x = self.out(x)
        return x

net1 = Net(2, 10, 2)     
print(net1)  
net2 = torch.nn.Sequential(
    torch.nn.Linear(2,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10,2)
)
print(net2)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
# loss_func = torch.nn.CrossEntropyLoss()

# plt.ion()

# for t in range(4):
#     out = net(x)
#     loss = loss_func(out, y)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     # if t == 0:

#     plt.cla()
#     prediction = torch.max(out, 1)[1]
#     pred_y = prediction.data.numpy()
#     target_y = y.data.numpy()
#     plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#     accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#     plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#     plt.pause(0.1)

# plt.ioff()
# plt.show()

