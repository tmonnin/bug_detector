import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
from torch.autograd import Variable
#from torch_geometric.nn import GCNConv
#from torch_geometric import transforms
#import torch_geometric

class Net(nn.Module):

    def __init__(self, num_features=13, num_classes=1):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(num_features, 50)

        # self.conv1 = GCNConv(num_features, 128, cached=True,
        #                      normalize=False)
        # self.conv2 = GCNConv(128, , cached=True,
        #                      normalize=False)
        # self.conv3 = GCNConv(9, 5, cached=True,
        #                      normalize=False)
        # self.conv4 = GCNConv(5, num_classes, cached=True,
        #                      normalize=False)

        #self.lstm = nn.LSTM(150, 128, num_layers=1, bidirectional=False)

        self.conv1 = nn.Conv2d(150, 32, (1, 5), stride=(1,1))#, padding=(0,2))
        self.conv2 = nn.Conv2d(32, 64, (1, 3), stride=(1,1))
        self.conv3 = nn.Conv2d(64, 64, (1, 3), stride=(1,1))
        self.conv4 = nn.Conv2d(64, 64, (1, 3), stride=(1,1))
        # TODO 1D
        self.lin1 = nn.Linear(64*1*5, 128)
        self.lin2 = nn.Linear(128, num_classes)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

        self.criterion = nn.BCELoss()

        self.params = {'batch_size': 32,
                       'shuffle': True,
                       'num_workers': 0}

    def forward(self, type_oh, property_ft):
        #[type_oh, property] = data #, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        type_embedded = self.embedding(type_oh)
        type_embedded = torch.transpose(type_embedded, 1, 2)
        type_embedded = type_embedded.unsqueeze(2)

        x = torch.cat((type_embedded, property_ft), dim=1)

        #lstm_out, _ = self.lstm(x.view(len(property_ft.size()[3]), 1, -1))


        x = F.relu(self.conv1(x))#, edge_index, edge_weight))
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x))#, edge_index, edge_weight))
        x = F.relu(self.conv3(x))#, edge_index, edge_weight))
        x = F.relu(self.conv4(x))#, edge_index, edge_weight))

        #x = x[0] # select weights from root node
        x = x.view(-1, 64*1*5)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

    def train(self, train_set, learning_rate, epochs):
        training_set = Dataset(train_set)
        training_generator = torch.utils.data.DataLoader(training_set, **self.params) # torch_geometric.data.DataLoader(training_set, **self.params)

        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.Adam(self.parameters())# SGD(self.parameters(), lr=learning_rate, momentum=0.2)
        # create a loss function

        # run the main training loop
        for epoch in range(epochs):
            loss_avg = 0
            count_correct = 0.0
            count_wrong = 0.0
            i = 0
            for batch_idx, (type_oh, property_ft, data, target) in enumerate(training_generator):
                #data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                try:
                    net_out = self(type_oh, property_ft)
                except:
                    import traceback
                    traceback.print_exc()
                    print("FAIL")
                    raise
                pred = net_out.to(torch.float32)
                loss = self.criterion(pred, target)
                correct_pred = torch.eq(pred.round(), target)
                count_correct += (correct_pred == True).sum()
                count_wrong += (correct_pred == False).sum()
                #loss = F.binary_cross_entropy(pred.to(torch.float32), target.to(torch.float32))

                loss.backward()
                loss_avg += loss.data
                i += 1
                optimizer.step()
                if batch_idx * self.params['batch_size'] % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * self.params['batch_size'], len(training_generator.dataset),
                                   100. * batch_idx / len(training_generator), loss.data))
            print('Train Epoch: {} \tLoss: {:.4f} \tAcc: {:.4f}'.format(epoch, loss_avg/i, count_correct / (count_correct+count_wrong)))

            torch.save(self.state_dict(), "model")

    def classify(self, data_set):
        test_set = Dataset(data_set)
        test_generator = torch.utils.data.DataLoader(test_set, **self.params) # torch_geometric.data.DataLoader(test_set, **self.params)
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

    def test(self, test_set):
        test_set = Dataset(test_set)
        test_generator = torch.utils.data.DataLoader(test_set, **self.params) # torch_geometric.data.DataLoader(test_set, **self.params)
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


class Dataset(torch.utils.data.Dataset):# torch_geometric.data.Dataset):

    def __init__(self, data_lst):
        self.data_lst = data_lst

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, index):
        data_dict = self.data_lst[index]
        type_oh = data_dict['type_oh']
        property_ft = data_dict['property_ft']
        data = []# data_dict['data']
        target = data_dict['label']
        return type_oh, property_ft, data, target
