import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.autograd import Variable
#from torch_geometric.nn import GCNConv
#from torch_geometric import transforms
#import torch_geometric

class Net(nn.Module):

    def __init__(self, num_features=13, num_classes=1):
        super(Net, self).__init__()
        # TODO check if necessary
        # 4 special characters: < pad >, EOS, < unk >, N
        # https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
        self.embedding = nn.Embedding(num_features, 50)
        strategy = "lstm"
        if strategy == "graph_conv":
            self.conv1 = GCNConv(num_features, 128, cached=True, normalize=False)
            self.conv2 = GCNConv(128, 9, cached=True, normalize=False)
            self.conv3 = GCNConv(9, 5, cached=True, normalize=False)
            self.conv4 = GCNConv(5, num_classes, cached=True, normalize=False)

            self.reg_params = self.conv1.parameters()
            self.non_reg_params = self.conv2.parameters()

        elif strategy == "conv":
            # TODO 1D conv
            self.conv1 = nn.Conv2d(150, 32, (1, 5), stride=(1,1))#, padding=(0,2))
            self.conv2 = nn.Conv2d(32, 64, (1, 3), stride=(1,1))
            self.conv3 = nn.Conv2d(64, 64, (1, 3), stride=(1,1))
            self.conv4 = nn.Conv2d(64, 64, (1, 3), stride=(1,1))

        elif strategy == "lstm":
            self.lstm_condition = nn.LSTM(150, 256, num_layers=1, bidirectional=False, batch_first=True) # TODO potentially increase e.g. 512
            self.lstm_context = nn.LSTM(100, 128, num_layers=1, bidirectional=False, batch_first=True) # TODO potentially increase e.g. 512
            # TODO padding
            # 5 prev and post lines
            # Fancy: attention
            # Type context: Vergrößern oder eigenes Embedding
            # BUT TYPE NOT RELEVANT for wrong operator task
            # Output: 10x lstm_context, 1x lstm_condition
            # Concat, linear, auf 128

        self.lin_condition = nn.Linear(256, 128)
        self.lin_context = nn.Linear(128, 64)
        self.lin = nn.Linear(192, num_classes)

        self.criterion = nn.BCELoss()

        self.params = {'batch_size': 32,
                       'shuffle': False,
                       'num_workers': 0}

    def forward(self, type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens):
        #outputs, outputs_len = torch.nn.utils.rnn.pad_packed_sequence(type_pack, batch_first=True)

        strategy = "lstm"
        #for type_len, type_pad in zip(type_lens, type_batch_pad):
        type_embedded = self.embedding(torch.tensor(type_batch_pad).to(torch.int64))
        x_condition = torch.cat([type_embedded, property_batch_pad], dim=2) # B,S,C

        if strategy == "conv":
            type_embedded = torch.transpose(type_embedded, 1, 2)
            type_embedded = type_embedded.unsqueeze(2)

            x_condition = torch.cat((type_embedded, property_embedded), dim=1)

            x_condition = F.relu(self.conv1(x_condition))
            #x = F.dropout(x, training=self.training)
            x_condition = F.relu(self.conv2(x_condition))
            x_condition = F.relu(self.conv3(x_condition))
            x_condition = F.relu(self.conv4(x_condition))
            #x = x[0] # select weights from root node
            x_condition = x_condition.view(-1, 64*1*5)
            x_condition = F.relu(self.lin1(x_condition))

        if strategy == "graph_conv":
            [type_int, property] = data  # , edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x_condition = F.relu(self.conv1(x_condition), edge_index, edge_weight)
            # x = F.dropout(x, training=self.training)
            x_condition = F.relu(self.conv2(x_condition), edge_index, edge_weight)
            x_condition = F.relu(self.conv3(x_condition), edge_index, edge_weight)
            x_condition = F.relu(self.conv4(x_condition), edge_index, edge_weight)
            # x = x[0] # select weights from root node
            x_condition = x_condition.view(-1, 64 * 1 * 5)
            x_condition = F.relu(self.lin1(x_condition))

        if strategy == "lstm":
            packed_input = torch.nn.utils.rnn.pack_padded_sequence(x_condition, pad_lens, enforce_sorted=False, batch_first=True)
            packed_outputs, _ = self.lstm_condition(packed_input)#, input_memory)
            x_condition, outputs_len = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            # (batch_size) -> (batch_size, 1, 1)
            lengths = pad_lens.unsqueeze(1).unsqueeze(2)
            # (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
            lengths = lengths.expand((-1, 1, x_condition.size(2)))
            # (batch_size, seq, hidden_size) -> (batch_size, 1, hidden_size)
            x_condition = torch.gather(x_condition, 1, lengths - 1)
            # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
            x_condition = x_condition.squeeze(1)


            packed_context_in = torch.nn.utils.rnn.pack_padded_sequence(token_batch_pad, pad_token_lens, enforce_sorted=False, batch_first=True)
            packed_context_out, _ = self.lstm_context(packed_context_in)
            x_context, outputs_len = torch.nn.utils.rnn.pad_packed_sequence(packed_context_out, batch_first=True)
            # (batch_size) -> (batch_size, 1, 1)
            lengths = pad_token_lens.unsqueeze(1).unsqueeze(2)
            # (batch_size, 1, 1) -> (batch_size, 1, hidden_size)
            lengths = lengths.expand((-1, 1, x_context.size(2)))
            # (batch_size, seq, hidden_size) -> (batch_size, 1, hidden_size)
            x_context = torch.gather(x_context, 1, lengths - 1)
            # (batch_size, 1, hidden_size) -> (batch_size, hidden_size)
            x_context = x_context.squeeze(1)


        x_condition = F.relu(self.lin_condition(x_condition))
        x_context = F.relu(self.lin_context(x_context))
        x = torch.cat([x_condition, x_context], dim=1)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

    def train(self, train_set, learning_rate, epochs, weights):
        training_set = TrainLoader(train_set)
        assert len(training_set) == len(weights)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        training_generator = torch.utils.data.DataLoader(training_set, **self.params, collate_fn=training_set.pad_collate, sampler=weighted_sampler) # torch_geometric.data.DataLoader(training_set, **self.params)

        # create a stochastic gradient descent optimizer
        optimizer = torch.optim.Adam(self.parameters())# SGD(self.parameters(), lr=learning_rate, momentum=0.2)
        # create a loss function

        # run the main training loop
        for epoch in range(epochs):
            loss_avg = 0
            count_correct = 0.0
            count_wrong = 0.0
            i = 0
            for batch_idx, (type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens, label_batch) in enumerate(training_generator):
                # TODO why Variable? Necessary?
                #data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                net_out = self(type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens)
                pred = net_out.to(torch.float32).squeeze(1)
                target_batch = torch.where(label_batch == 0, torch.zeros_like(label_batch, dtype=torch.float), torch.ones_like(label_batch, dtype=torch.float))
                loss = self.criterion(pred, target_batch)
                correct_pred = torch.eq(pred.round(), target_batch)
                count_correct += (correct_pred == True).sum()
                count_wrong += (correct_pred == False).sum()
                loss.backward()
                loss_avg += loss.data
                i += 1
                optimizer.step()
                if batch_idx * self.params['batch_size'] % 1000 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * self.params['batch_size'], len(training_generator.dataset),
                                   100. * batch_idx / len(training_generator), loss.data))
            accuracy = count_correct / (count_correct+count_wrong)
            loss_avg = loss_avg/i
            print('Train Epoch: {} \tLoss: {:.4f} \tAcc: {:.4f}'.format(epoch, loss_avg, accuracy))

            torch.save(self.state_dict(), 'model_{}_{:d}_{:d}'.format(epoch+1, int(loss_avg*100), int(accuracy*100)))

    def classify(self, data_set):
        classify_set = ClassifyLoader(data_set)
        test_generator = torch.utils.data.DataLoader(classify_set, **self.params, collate_fn=classify_set.pad_collate) # torch_geometric.data.DataLoader(test_set, **self.params)
        is_bug = []
        for batch_idx, (type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens) in enumerate(test_generator):
            net_out = self(type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens)
            pred = net_out.to(torch.float32).squeeze(1)
            pred = (pred >= 0.5).tolist()
            is_bug += pred  # TODO finetune for tradeoff precision and recall
        return is_bug

    def test(self, test_set):
        test_set = TrainLoader(test_set)
        test_generator = torch.utils.data.DataLoader(test_set, **self.params, collate_fn=test_set.pad_collate) # torch_geometric.data.DataLoader(test_set, **self.params)
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


class TrainLoader(torch.utils.data.Dataset):# torch_geometric.data.Dataset):

    def __init__(self, data_lst):
        self.data_lst = data_lst

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, index):
        data_dict = self.data_lst[index]
        type_int_lst = data_dict['type_int_lst']
        type_tensor = torch.stack(type_int_lst)
        property_emb_lst = data_dict['property_emb_lst']
        property_tensor = torch.stack(property_emb_lst)
        token_emb_lst = data_dict['code_adjacent_emb_lst']
        token_tensor = torch.stack(token_emb_lst)
        target = data_dict['label']
        return type_tensor, property_tensor, token_tensor, target

    def pad_collate(self, batch):
        (type_batch, property_batch, token_batch, target_batch) = zip(*batch)
        pad_lens = torch.tensor([len(type) for type in type_batch])

        type_batch_pad = pad_sequence(type_batch, batch_first=True, padding_value=0)
        property_batch_pad = pad_sequence(property_batch, batch_first=True, padding_value=0)
        token_batch_pad = pad_sequence(token_batch, batch_first=True, padding_value=0)
        pad_token_lens = torch.tensor([len(token) for token in token_batch])

        return type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens, torch.tensor(target_batch)


class ClassifyLoader(TrainLoader):

    def __getitem__(self, index):
        data_dict = self.data_lst[index]
        type_int_lst = data_dict['type_int_lst']
        type_tensor = torch.stack(type_int_lst)
        property_emb_lst = data_dict['property_emb_lst']
        property_tensor = torch.stack(property_emb_lst)
        token_emb_lst = data_dict['code_adjacent_emb_lst']
        token_tensor = torch.stack(token_emb_lst)
        return type_tensor, property_tensor, token_tensor

    def pad_collate(self, batch):
        (type_batch, property_batch, token_batch) = zip(*batch)
        pad_lens = torch.tensor([len(type) for type in type_batch])

        type_batch_pad = pad_sequence(type_batch, batch_first=True, padding_value=0)
        property_batch_pad = pad_sequence(property_batch, batch_first=True, padding_value=0)

        token_batch_pad = pad_sequence(token_batch, batch_first=True, padding_value=0)
        pad_token_lens = torch.tensor([len(token) for token in token_batch])

        return type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens