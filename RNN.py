import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from time import time
import datetime
import h5py

# Define Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def forward(self, x, y):
        return self.rel(x, y)

    def __call__(self, x, y):
        return self.forward(x, y)

# Define data reader
class MatRead(object):
    def __init__(self, file_path):
        super(MatRead).__init__()

        self.file_path = file_path
        self.data = h5py.File(self.file_path)

    def get_strain(self):
        strain = np.array(self.data['strain']).transpose(2,0,1)
        return torch.tensor(strain, dtype=torch.float32)

    def get_stress(self):
        stress = np.array(self.data['stress']).transpose(2,0,1)
        return torch.tensor(stress, dtype=torch.float32)

# Define normalizer
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

# Define RNN architecture
class RNN(nn.Module):
    def __init__(self, hidden_size, layer_hidden, layer_input):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.hidden_layers = nn.ModuleList()
        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 2:
                self.hidden_layers.append(nn.SELU())
                
        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 2:
                self.layers.append(nn.SELU())

    def forward(self, input, last, hidden, dt):
        h0 = hidden
        h = torch.cat((last, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)
        h = h*dt + h0

        x = torch.cat((input, h), 1)
        for _, l in enumerate(self.layers):
            x = l(x)

        output = x
        hidden = h
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

############################# Data processing #############################
# Read data from mat
path = 'Material.mat'
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# Split data into train and test
ntrain = 1000
ntest = strain.shape[0] - ntrain
train_strain = strain[:ntrain, :, :]
train_stress = stress[:ntrain, :, :]
test_strain = strain[ntrain:, :, :]
test_stress = stress[ntrain:, :, :]

# Normalize data
strain_normalizer = UnitGaussianNormalizer(train_strain)
train_strain_encode = strain_normalizer.encode(train_strain)
test_strain_encode = strain_normalizer.encode(test_strain)

stress_normalizer = UnitGaussianNormalizer(train_stress)
train_stress_encode = stress_normalizer.encode(train_stress)
test_stress_encode = stress_normalizer.encode(test_stress)

ndim = strain.shape[2]  # Number of components
nstep = strain.shape[1] # Number of time steps
dt = 1/(nstep-1)

# Create data loader
batch_size = 50
train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

test_set = Data.TensorDataset(test_strain_encode, test_stress_encode)
test_loader = Data.DataLoader(test_set, ntest, shuffle=False)

############################# Define and train network #############################

nhidden = 3
layer_hidden = [ndim + nhidden, 10, 20, 10, nhidden]
layer_input = [ndim + nhidden, 10, 20, 10, ndim]
net = RNN(nhidden, layer_hidden, layer_input)

loss_func = LpLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Number of parameters: %d' % n_params)    

# Train network
epochs = 100
print("Start training RNN for {} epochs...".format(epochs))
start_time = time()

loss_train_list = []
loss_test_list = []
x = []

for epoch in range(epochs):
    net.train(True)
    trainloss = 0
    testloss = 0

    for step, (input, target) in enumerate(train_loader):
        hidden = net.initHidden(batch_size)
        output = torch.zeros(batch_size, nstep, ndim)
        output[:, 0, :] = target[:, 0, :]
        for t in range(1, nstep):
            output[:, t, :], hidden = net(input[:, t, :], input[:, t-1, :], hidden, dt)
        loss = loss_func(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()       
        scheduler.step()

        trainloss += loss.item()

    # Test
    net.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            test_hidden = net.initHidden(ntest)
            test_output = torch.zeros(ntest, nstep, ndim)
            test_output[:, 0, :] = target[:, 0, :]
            for t in range(1, nstep):
                test_output[:, t, :], test_hidden = net(input[:, t, :], input[:, t-1, :], test_hidden, dt)
            loss = loss_func(test_output, target)
            testloss += loss.item()

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss/len(test_loader)))

    # Save loss
    loss_train_list.append(trainloss/len(train_loader))
    loss_test_list.append(testloss/len(test_loader))
    x.append(epoch)

total_time = time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))

print('Traing time: {}'.format(total_time_str))
print("Train loss:{}".format(trainloss/len(train_loader)))
print("Test loss:{}".format(testloss/len(test_loader)))


############################# Plot #############################
plt.figure(1)
plt.plot(x, loss_train_list, label='Train loss')
plt.plot(x, loss_test_list, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Test Losses')
plt.legend()
plt.grid()

plt.figure(2)
t = np.linspace(0, 1, train_strain.shape[1])
nsample = 12
ndim = 1
test_output = stress_normalizer.decode(test_output)
plt.plot(t, test_strain[nsample, :, ndim-1].detach().numpy(), label='Strain_xx')
plt.plot(t, test_stress[nsample, :, ndim-1].detach().numpy(), label='True Stress_xx')
plt.plot(t, test_output[nsample, :, ndim-1].detach().numpy(), label='Approximate Stress_xx')
plt.xlabel('Time')
plt.title('True Stress_xx vs Approximate Stress_xx for Sample {}'.format(nsample))
plt.legend()
plt.grid()

plt.show()