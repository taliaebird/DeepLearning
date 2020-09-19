import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from torch.nn.parameter import Parameter
import pdb

assert torch.cuda.is_available(),

class CrossEntropyLoss(nn.Module):
  def __init__(self, size_average = None, ignore_index = -100, reduce=None, reduction = 'mean'):
    #defining variables
    self.__dict__.update(locals())
    super(CrossEntropyLoss, self).__init__()

  def forward(self, y_hat, y_truth):
    #defining the losses  
    losses = (-1 * y_hat[np.array(range(len(y_hat))), y_truth]) + torch.log(torch.sum((torch.exp(y_hat)), dim = 1))
    
    #finding the average loss
    loss = torch.sum(losses)/len(y_truth)
    
    return loss

class Conv2d(nn.Module): 
  def __init__(self, n_channels, out_channels, kernel_size, initialization = 'none', stride = 1,
               padding = 0, dilation = 1, groups = 1, bias = True):
    
    #define the items in the constructor
    self.__dict__.update(locals())
    super(Conv2d, self).__init__()

    # (out, in, k, k)
    self.weight = Parameter(torch.Tensor(out_channels, 
                               n_channels, 
                               *kernel_size))
    
    self.bias = Parameter(torch.Tensor(out_channels))

    #convolution intialization
    if(initialization == 'orthogonal'):   #setting weights in the orthogonal method
      X = np.random.random((out_channels, n_channels * kernel_size[0] * kernel_size[0]))
      U, _, Vt = np.linalg.svd(X, full_matrices = False)
      np.allclose(np.dot(Vt, Vt.T), np.eye(Vt.shape[0]))
      W = Vt.reshape((out_channels, n_channels, kernel_size[0], kernel_size[0]))

      self.weight.data = torch.from_numpy(W).float()   #setting weights
      self.bias.data.uniform_(0, 0)   #setting bias

    elif(initialization == 'uniform'):    #setting weights with the uniform method
      self.weight.data.uniform_(-1, 1)    #setting weights
      self.bias.data.uniform_(0, 0)   #setting bias

    else: #using Xavier initialization
      W = np.full((out_channels, n_channels, kernel_size[0], kernel_size[0]), (1/n_channels), dtype = float)
      self.weight.data = torch.from_numpy(W).float()   #setting weights
      self.bias.data.uniform_(0, 0)   #setting bias
    
  # initialize these
  def forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)

class ConvNetwork(nn.Module):
  def __init__(self, dataset):
    super(ConvNetwork, self).__init__() 
    x, y = dataset[0]
    c, h, w = x.size()
    output = 10

    self.net = nn.Sequential(    
        Conv2d(c, 125, (3, 3), initialization = 'uniform', padding = (1,1)),
        nn.ReLU(),  
        Conv2d(125, output, (28, 28), initialization = 'orthogonal', padding = (0, 0)),
    )

  def forward(self, x):
    #returns (n, 10, 1, 1)
    #then returns (n, 10, 1)
    #now returns (n, 10)
    return self.net(x).squeeze(2).squeeze(2)

class FashinoMNISTProcessedDataset(Dataset):
  def __init__(self, root, train=True):
    self.data = datasets.FashionMNIST(root, 
                                      train=train,
                                      transform=transforms.ToTensor(),
                                      download=True)
  def __getitem__(self,i): 
    x, y = self.data[i]
    return x, y

  def __len__(self):
    return len(self.data)

train_dataset = FashinoMNISTProcessedDataset('/tmp/fashionmnist', train=True)
val_dataset = FashinoMNISTProcessedDataset('/tmp/fashionmnist', train=False)
num_epochs = 10

model = ConvNetwork(train_dataset)    
model = model.cuda()
objective = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)
train_loader = DataLoader(train_dataset, 
                          batch_size = 42, 
                          pin_memory = True)
val_loader = DataLoader(val_dataset, 
                               batch_size = 42)

losses = []
validations = []    
accuracies = []

for epoch in range(1):
  loop = tqdm(total = len(train_loader), position = 0, leave = False)

  for batch, (x, y_truth) in enumerate(train_loader):
    x, y_truth = x.cuda(async = True), y_truth.cuda(async = True)

    optimizer.zero_grad()
    y_hat = model(x)
    loss = objective(y_hat, y_truth)
    loss.backward()

    losses.append(loss.item())

    #updating the accuracies
    accuracy = (torch.softmax(y_hat, 1).argmax(1) == y_truth).float().mean()    #FIXME
    accuracies.append(accuracy.item())

    loop.set_description('epoch:{}, loss:{:.4f}, accuracy:{:.3f}'.format(epoch, loss, accuracy))
    loop.update(1)

    optimizer.step()

    if batch % 100 == 0:  

      #returns mean of a list of loss model, objective finds loss, model(x.cuda()) finds output 
      val = np.mean([objective(model(x.cuda()), y.cuda()).item() for x, y in val_loader])
      validations.append((len(losses), val))

  loop.close()

#Plotting the training and validation loss
train_plot = plt.subplot(121)
a, b = zip(*validations)
train_plot.plot(losses, label = 'train')
train_plot.plot(a, b, label = 'val')
train_plot.legend()
plt.title("Training and Validation Loss")

#Plotting the accuracy
acc_plot = plt.subplot(122)
acc_plot.plot(accuracies, 'm', label = 'accuracy')
acc_plot.axis([0, len(accuracies), 0, 1])
acc_plot.legend()
plt.title("Accuracy")

plt.show()

#initalizing the number of parameters to 0
total = 0

#iterate through model parameters
#for each p, we want to get the product of all its entries then sum all products
for p in model.parameters():
  product = 1   #initializing the product to 1
  for s in p.size():
      product = product * s   #getting the product of all entries

  total = total + product + 1   #updating the total to include product of p, add one for bias

#printing the total number of parameters
print('Total number of parameters:', total)
