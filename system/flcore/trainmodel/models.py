from numpy.lib.histograms import histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

batch_size = 16

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, batch_size, 2, 1)
#         self.conv2 = nn.Conv2d(batch_size, 32, 2, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(18432, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = nn.ReLU()(x)
#         x = nn.MaxPool2d(2, 1)(x)
#         x = self.dropout2(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = nn.ReLU()(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

# ==========================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*28*28, num_labels=10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class Mclr_CrossEntropy(nn.Module):
    def __init__(self, input_dim=1*28*28, num_labels=10):
        super(Mclr_CrossEntropy, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        outputs = self.linear(x)
        return outputs

# ==========================================================

class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_labels=10):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, num_labels)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

class DNNbase(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100):
        super(DNNbase, self).__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return x

class DNNClassifier(nn.Module):
    def __init__(self, mid_dim=100, num_labels=10):
        super(DNNClassifier, self).__init__()
        self.fc2 = nn.Linear(mid_dim, num_labels)

    def forward(self, x):
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# ==========================================================

class CifarNet(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

# class CifarNetHead(nn.Module):
#     def __init__(self):
#         super(CifarNetHead, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         return x

# class CifarNetBase(nn.Module):
#     def __init__(self):
#         super(CifarNetBase, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
 
#     def forward(self, x):
#         x = self.pool(x)
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

class CifarNetBase(nn.Module):
    def __init__(self):
        super(CifarNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class CifarNetClassifier(nn.Module):
    def __init__(self, num_labels=10):
        super(CifarNetClassifier, self).__init__()
        self.fc = nn.Linear(84, num_labels)

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ==========================================================

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGGbatch_size': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 512),
#             nn.ReLU(True),
#             nn.Linear(512, 10)
#         )

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         output = F.log_softmax(out, dim=1)
#         return output

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

# ==========================================================

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class LeNet(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256, num_labels=10, iswn=None):
        super(LeNet, self).__init__()

        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class LeNetBase(nn.Module):
    def __init__(self, feature_dim=50*4*4, bottleneck_dim=256):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class LeNetClassifier(nn.Module):
    def __init__(self, num_labels=10, bottleneck_dim=256, iswn=None):
        super(LeNetClassifier, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, num_labels)
        if iswn == "wn":
            self.fc = nn.utils.weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

# class LeNetBase(nn.Module):
#     def __init__(self):
#         super(LeNetBase, self).__init__()
#         self.conv_params = nn.Sequential(
#                 nn.Conv2d(1, 20, kernel_size=5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 nn.Conv2d(20, 50, kernel_size=5),
#                 nn.Dropout2d(p=0.5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 )
#         self.in_features = 50*4*4

#     def forward(self, x):
#         x = self.conv_params(x)
#         x = x.view(x.size(0), -1)
#         return x

# class LeNet_bootleneck(nn.Module):
#     def __init__(self, feature_dim, bottleneck_dim=256, iswn="bn"):
#         super(LeNet_bootleneck, self).__init__()
#         self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
#         self.dropout = nn.Dropout(p=0.5)
#         self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
#         # self.bottleneck.apply(init_weights)
#         self.iswn = iswn

#     def forward(self, x):
#         x = self.bottleneck(x)
#         if self.iswn == "bn":
#             x = self.bn(x)
#             x = self.dropout(x)
#         return x

# ==========================================================

# class CNNCifar(nn.Module):
#     def __init__(self, num_labels=10):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, batch_size, 5)
#         self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 100)
#         self.fc3 = nn.Linear(100, num_labels)

#         # self.weight_keys = [['fc1.weight', 'fc1.bias'],
#         #                     ['fc2.weight', 'fc2.bias'],
#         #                     ['fc3.weight', 'fc3.bias'],
#         #                     ['conv2.weight', 'conv2.bias'],
#         #                     ['conv1.weight', 'conv1.bias'],
#         #                     ]
                            
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, batch_size * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = F.log_softmax(x, dim=1)
#         return x

# ==========================================================

class ResNetClassifier(nn.Module):
    def __init__(self, input_dim=512, num_labels=10):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


# ==========================================================

class LSTMNet(nn.Module):
    def __init__(self, hidden_dim, num_layers, bidirectional=True, dropout=0.5, 
                padding_idx=0, vocab_size=98635, num_labels=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, num_labels)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted = False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # output = self.fc1(hidden)
        # output = self.dropout(self.fc2(output))
                
        #hidden = [batch size, hid dim * num directions]

        # # Each sequence "x" is passed through an embedding layer
        # out = self.embedding(x) 

        # Feed LSTMs
        # out, (hidden, cell) = self.lstm(out)
        out = self.dropout(out)

        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
            
        return out

class LSTMNetBase(nn.Module):
    def __init__(self, hidden_dim, num_layers, bidirectional, dropout=0.5, 
                padding_idx=0, vocab_size=98635, num_labels=10):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim*2)

    def forward(self, x):
        text, text_lengths = x
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        out, (hidden, cell) = self.lstm(packed_embedded)
        
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        # output = self.fc1(hidden)
        # output = self.dropout(self.fc2(output))
                
        #hidden = [batch size, hid dim * num directions]

        # # Each sequence "x" is passed through an embedding layer
        # out = self.embedding(x)

        # Feed LSTMs
        # out, (hidden, cell) = self.lstm(out)
        out = self.dropout(out)

        # The last hidden state is taken
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
            
        return out

class LSTMNetClassifier(nn.Module):
    def __init__(self, hidden_dim, num_labels=10):
        self.fc2 = nn.Linear(hidden_dim*2, num_labels)

    def forward(self, out):
        out = torch.sigmoid(self.fc2(out))

        return out

# ==========================================================

# class linear(Function):
#   @staticmethod
#   def forward(ctx, input):
#     return input
  
#   @staticmethod
#   def backward(ctx, grad_output):
#     return grad_output