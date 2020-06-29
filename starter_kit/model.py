import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
from torch_geometric import transforms
import torch_geometric

class Net(nn.Module):

    def __init__(self, num_features=100, num_classes=1):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 64, cached=True,
                             normalize=False)
        self.conv2 = GCNConv(64, 32, cached=True,
                             normalize=False)
        self.conv3 = GCNConv(32, 16, cached=True,
                             normalize=False)
        self.conv4 = GCNConv(16, num_classes, cached=True,
                             normalize=False)
        self.lin = nn.Linear(16, num_classes)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

        self.criterion = nn.BCELoss()

        self.params = {'batch_size': 1,
                       'shuffle': True,
                       'num_workers': 1}

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        #x = x[0] # select weights from root node
        #x = x.view(-1, 64)
        #x = self.lin(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = torch.sigmoid(x)
        return x

    def train(self, train_set, learning_rate, epochs):
        training_set = Dataset(train_set)
        training_generator = torch_geometric.data.DataLoader(training_set, **self.params)

        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.2)
        # create a loss function

        # run the main training loop
        for epoch in range(epochs):
            loss_avg = 0
            i = 0
            for batch_idx, (data, target) in enumerate(training_generator):
                #data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                try:
                    net_out = self(data)
                except:
                    import traceback
                    traceback.print_exc()
                    print("FAIL")
                    continue
                pred = net_out[0]
                #loss = self.criterion(pred.to(torch.float32), target.to(torch.float32))
                loss = F.binary_cross_entropy(pred.to(torch.float32), target.to(torch.float32))

                loss.backward()
                loss_avg += loss.data
                i += 1
                optimizer.step()
                #if batch_idx / len(training_generator) % 10 == 0:
                #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #            epoch, batch_idx * len(data), len(training_generator.dataset),
                #                   100. * batch_idx / len(training_generator), loss.data))
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss_avg/i))

    def test(self, test_set):
        test_set = Dataset(test_set)
        test_generator = torch_geometric.data.DataLoader(test_set, **self.params)
        # run a test loop
        test_loss = 0
        correct = 0
        for data, target in test_generator:
            data, target = Variable(data, volatile=True), Variable(target)
            net_out = self(data)
            # sum up batch loss
            test_loss += self.criterion(net_out, target).data[0]
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_generator.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_generator.dataset),
            100. * correct / len(test_generator.dataset)))


class Dataset(torch_geometric.data.Dataset):

    def __init__(self, path_list):
        self.path_list = path_list
        #self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need.

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        X, y = self.path_list[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]
        #return np.zeros(100), 0


        # word in token embedding: 1x100 [-1,1]
        # print(token_embedding.words)  # list of words in dictionary
        return X, y
