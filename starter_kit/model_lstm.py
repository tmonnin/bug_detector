import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils

class LSTMNet(nn.Module):

    def __init__(self, num_features=13, num_classes=1):
        super(LSTMNet, self).__init__()

        self.embedding = nn.Embedding(num_features, 50)

        self.lstm_condition = nn.LSTM(150, 256, num_layers=1, bidirectional=False, batch_first=True) # TODO potentially increase e.g. 512
        self.lstm_context = nn.LSTM(100, 128, num_layers=1, bidirectional=False, batch_first=True) # TODO potentially increase e.g. 512

        self.lin_condition = nn.Linear(256, 128)
        self.lin_context = nn.Linear(128, 64)
        self.lin = nn.Linear(192, num_classes)

        self.criterion = nn.BCELoss()

        self.params = {'batch_size': 32,
                       'shuffle': False,
                       'num_workers': 0}

    def forward(self, type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens):

        type_embedded = self.embedding(torch.tensor(type_batch_pad).to(torch.int64))
        x_condition = torch.cat([type_embedded, property_batch_pad], dim=2) # B,S,C

        packed_input = pack_padded_sequence(x_condition, pad_lens, enforce_sorted=False, batch_first=True)
        packed_outputs, _ = self.lstm_condition(packed_input)#, input_memory)
        x_condition, outputs_len = pad_packed_sequence(packed_outputs, batch_first=True)
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

    def train(self, train_set, learning_rate, epochs, distribution):
        training_set = TrainLoader(train_set)
        labels = []
        [labels.append(data["label"]) for data in train_set]
        weights = utils.weighted_distribution(labels, distribution)
        assert len(training_set) == len(weights)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
        training_loader = torch.utils.data.DataLoader(training_set, **self.params, collate_fn=training_set.pad_collate, sampler=weighted_sampler)

        # Use Adam optimizer as suggested in literature
        optimizer = torch.optim.Adam(self.parameters())

        # Training loop
        for epoch in range(1, epochs+1):
            loss_avg = 0
            count_correct = 0.0
            count_wrong = 0.0
            i = 0
            for batch_idx, (type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens, label_batch) in enumerate(training_loader):
                optimizer.zero_grad()
                # Inference
                net_out = self(type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens)
                pred = net_out.to(torch.float32).squeeze(1)
                zero_target = torch.zeros_like(label_batch, dtype=torch.float)
                one_target = torch.ones_like(label_batch, dtype=torch.float)
                target_batch = torch.where(label_batch == 0, zero_target, one_target)
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
                            epoch, batch_idx * self.params['batch_size'], len(training_loader.dataset),
                                   100. * batch_idx / len(training_loader), loss.data))
            accuracy = count_correct / (count_correct+count_wrong)
            loss_avg = loss_avg/i
            print('Train Epoch: {} \tLoss: {:.4f} \tAcc: {:.4f}'.format(epoch, loss_avg, accuracy))

            torch.save(self.state_dict(), 'model_lstm_d{:d}_e{}_l{:d}_a{:d}'.format(int(distribution[0]*100), epoch, int(loss_avg*100), int(accuracy*100)))

    def classify(self, data_set):
        classify_set = ClassifyLoader(data_set)
        classify_loader = torch.utils.data.DataLoader(classify_set, **self.params, collate_fn=classify_set.pad_collate)
        is_bug = []
        for batch_idx, (type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens) in enumerate(classify_loader):
            net_out = self(type_batch_pad, property_batch_pad, token_batch_pad, pad_lens, pad_token_lens)
            pred = net_out.to(torch.float32).squeeze(1)
            # Choose hyperparameter as tradeoff between precision and recall
            pred = (pred >= 0.5).tolist()
            is_bug += pred
        return is_bug

    def test(self, test_set):
        test_set = TrainLoader(test_set)
        test_loader = torch.utils.data.DataLoader(test_set, **self.params, collate_fn=test_set.pad_collate)
        # run a test loop
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            net_out = self(data)
            # sum up batch loss
            test_loss += self.criterion(net_out, target).data[0]
            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


class TrainLoader(torch.utils.data.Dataset):

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