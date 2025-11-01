import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import h5py

# Define your loss function here
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

# This reads the matlab data from the .mat file provided
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

# Define data normalizer
class DataNormalizer:
    def __init__(self, data):
        """
        Compute mean and standard deviation for normalization.

        """
        self.mean = data.mean(dim=0, keepdim=True)  
        self.std = data.std(dim=0, keepdim=True)  

        # Prevent division by zero in case of constant features
        self.std[self.std == 0] = 1  

    def normalize(self, data):

        return (data - self.mean) / self.std

    def decode(self, normalized_data):

        return (normalized_data * self.std) + self.mean


# Define network your neural network for the constitutive model below
class Dense(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(Dense, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x 

######################### Data processing #############################
# Read data from .mat file
path = 'Material.mat'
data_reader = MatRead(path)
strain = data_reader.get_strain()
stress = data_reader.get_stress()

# Flatten the data
strain_flat = strain.reshape(-1, 6)
stress_flat = stress.reshape(-1, 6)

# Split data into train and test
total_samples = strain_flat.shape[0]
ntrain = 50000
ntest = 5000

# Generate shuffled indices
indices = torch.randperm(total_samples)

# Split data
train_indices = indices[:ntrain] 
test_indices = indices[ntrain:ntrain + ntest] 

# Apply shuffling
train_strain = strain_flat[train_indices]
train_stress = stress_flat[train_indices]
test_strain = strain_flat[test_indices]
test_stress = stress_flat[test_indices]

# Normalize your data
strain_normalizer   = DataNormalizer(train_strain)
train_strain_encode = strain_normalizer.normalize(train_strain)
test_strain_encode  = strain_normalizer.normalize(test_strain)

stress_normalizer   = DataNormalizer(train_stress)
train_stress_encode = stress_normalizer.normalize(train_stress)
test_stress_encode  = stress_normalizer.normalize(test_stress)

# Create data loader
batch_size = 1000
train_set = Data.TensorDataset(train_strain_encode, train_stress_encode)
train_loader = Data.DataLoader(train_set, batch_size, shuffle=True)

test_set = Data.TensorDataset(test_strain_encode, test_stress_encode)
test_loader = Data.DataLoader(test_set, batch_size, shuffle=False)

############################# Define and train network #############################
# Create Nueral network, define loss function and optimizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameter configuration
input_size = 6
output_size = 6
num_epochs = 100
FCNN_arch = [input_size, 20, 40, 20, output_size] 
non_linearity = nn.ReLU 
net = Dense(FCNN_arch, non_linearity).to(device)
print(net)

loss_func = LpLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2) 

n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) #Calculate the number of training parameters
print('Number of parameters: %d' % n_params)

# Train network
epochs = num_epochs
print("Start training for {} epochs...".format(epochs))

loss_train_list = []
loss_test_list = []

for epoch in range(epochs):
    net.train(True)
    trainloss = 0


    for i, data in enumerate(train_loader):
        input, target = data

        optimizer.zero_grad()

        output_encode = net(input)
        output        = stress_normalizer.decode(output_encode)
        loss = loss_func(output_encode, target)

        loss.backward()
        optimizer.step()

        trainloss += loss.item()

    net.eval()
    with torch.no_grad():
        testloss = 0
        for i, data in enumerate(test_loader):
            input, target = data
            output        = net(input)
            loss = loss_func(output, target)
            testloss += loss.item()

    scheduler.step()

    # Print train loss every 10 epochs
    if epoch % 10 == 0:
        print("epoch:{}, train loss:{}, test loss:{}".format(epoch, trainloss/len(train_loader), testloss/len(test_loader)))

    # Save loss
    loss_train_list.append(trainloss/len(train_loader))
    loss_test_list.append(testloss/len(test_loader))


print("Train loss:{}".format(trainloss/len(train_loader)))
print("Test loss:{}".format(testloss)/len(test_loader))

############################# Plot your result below using Matplotlib #############################
plt.figure(1)
epochs = range(1, len(loss_test_list) + 1)

# Plot loss over epochs
plt.plot(epochs, loss_train_list, marker='o', linestyle='-')
plt.plot(epochs, loss_test_list, marker='o', linestyle='-')
plt.title('Train and Test Losses')

output = stress_normalizer.decode(net(strain_normalizer.normalize(strain.reshape(-1, 6)[-5000:])))

# Extract first column of input and output
x_values = input[:, 0].detach().cpu().numpy()
y_values = output[:, 0].detach().cpu().numpy()

plt.figure(2)

# Plot input vs. output
plt.plot(strain.reshape(-1, 6)[-5000:,0].detach().cpu().numpy(), output.reshape(-1, 6)[:,0].detach().cpu().numpy(), marker='o', linestyle='None')
plt.plot(strain.reshape(-1, 6)[-5000:,0].detach().cpu().numpy(), stress.reshape(-1, 6)[-5000:,0].detach().cpu().numpy(), marker='o', linestyle='None')
plt.title('Truth Stresses vs Approximate Stresses for Sample {}')
plt.grid(True)
plt.show()
