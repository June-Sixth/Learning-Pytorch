import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,# 是否在训练时打乱数据顺序 true：打乱
    num_workers=2,# 用几个线程跑 2个
)
def show_batch():
    for epoch in range(3):#training 3 times
        for step,(batch_x,batch_y) in enumerate(loader):
            #training
            print('Epoch',epoch,'|Step',step,'|batch x',batch_x.numpy(),
                '|batch y',batch_y.numpy())

if __name__ == '__main__':
    show_batch()