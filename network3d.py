from heartcontour.models.lvhyp.loader import DataType
from torch.nn.functional import pad as padding_func
import torch.nn as nn
import torch

# ---- Building blocks for the networks ----
# The used networks for the training 
# will be built from these.


class Lhyp3DModelBase(nn.Module):
    def __init__(self):
        super(Lhyp3DModelBase, self).__init__()

    def forward(self, x):
        return x

    def freeze(self):
        # freeze the layers
        for param in self.parameters():
            param.requires_grad = False

    def load(self, state_dict, device):
        self.load_state_dict(state_dict)
        self.to(device)


class Residual3DBlock(nn.Module):
    def __init__(self, channel_in_num, channel_out_num, kernel_size):
        super(Residual3DBlock, self).__init__() 
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        pad_d = (kernel_size[2] - 1) // 2
        pad = (pad_h, pad_w, pad_d)
        self.batchnorm1 = nn.BatchNorm3d(channel_in_num)
        self.conv1 = nn.Conv3d(channel_in_num, channel_out_num, kernel_size, stride=1, padding=pad)
        self.batchnorm2 = nn.BatchNorm3d(channel_out_num)
        self.conv2 = nn.Conv3d(channel_out_num, channel_out_num, kernel_size, stride=1, padding=pad)
        self.batchnorm_skip = nn.BatchNorm3d(channel_in_num)
        self.conv_skip = nn.Conv3d(channel_in_num, channel_out_num, kernel_size, stride=1, padding=pad)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_skip = self.batchnorm_skip(x)
        y_skip = self.relu(y_skip)
        y_skip = self.conv_skip(y_skip)
        return y + y_skip


class Residual3DBlockPooling(nn.Module):
    def __init__(self, channel_in_num, channel_out_num, kernel_size):
        super(Residual3DBlockPooling, self).__init__()
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        pad_d = (kernel_size[2] - 1) // 2
        pad = (pad_h, pad_w, pad_d)
        # main branch
        self.batchnorm1 = nn.BatchNorm3d(channel_in_num)
        self.conv1 = nn.Conv3d(channel_in_num, channel_out_num, kernel_size, stride=2)
        self.batchnorm2 = nn.BatchNorm3d(channel_out_num)
        self.conv2 = nn.Conv3d(channel_out_num, channel_out_num, kernel_size, stride=1, padding=pad)
        # skip branch
        self.batchnorm_skip = nn.BatchNorm3d(channel_in_num)
        self.conv_skip = nn.Conv3d(channel_in_num, channel_out_num, kernel_size, stride=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.batchnorm1(x)
        y = self.relu(y)
        y = self.conv1(y)
        y = self.batchnorm2(y)
        y = self.relu(y)
        y = self.conv2(y)
        y_skip = self.batchnorm_skip(x)
        y_skip = self.relu(y_skip)
        y_skip = self.conv_skip(y_skip)
        return y + y_skip


class Extractor3DLA(Lhyp3DModelBase):
    def __init__(self):
        super(Extractor3DLA, self).__init__()
        # creating the network architecture
        self.block1 = Residual3DBlock(1, 8, (5, 5, 3))
        self.block2 = Residual3DBlock(8, 16, (5, 5, 3))
        self.blockpool1 = Residual3DBlockPooling(16, 16, (3, 3, 3))
        self.block3 = Residual3DBlock(16, 24, (3, 3, 3))
        self.block4 = Residual3DBlock(24, 24, (3, 3, 3))
        self.blockpool2 = Residual3DBlockPooling(24, 24, (3, 3, 3))
        self.block5 = Residual3DBlock(24, 12, (3, 3, 3))
        self.block6 = Residual3DBlock(12, 6, (3, 3, 1))

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.blockpool1(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.blockpool2(y)
        y = self.block5(y)
        y = self.block6(y)
        return y


class Extractor3DSA(Lhyp3DModelBase):
    def __init__(self):
        super(Extractor3DSA, self).__init__()
        # creating the network architecture
        self.block1 = Residual3DBlock(1, 8, (5, 5, 3))
        self.block2 = Residual3DBlock(8, 16, (5, 5, 3))
        self.blockpool1 = Residual3DBlockPooling(16, 16, (3, 3, 3))
        self.block3 = Residual3DBlock(16, 24, (3, 3, 3))
        self.block4 = Residual3DBlock(24, 24, (3, 3, 3))
        self.blockpool2 = Residual3DBlockPooling(24, 24, (3, 3, 3))
        self.block5 = Residual3DBlock(24, 12, (3, 3, 3))
        self.block6 = Residual3DBlock(12, 6, (3, 3, 3))

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.blockpool1(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.blockpool2(y)
        y = self.block5(y)
        y = self.block6(y)
        return y


class Extractor3DLALE(Lhyp3DModelBase):
    def __init__(self):
        super(Extractor3DLALE, self).__init__()
        # creating the network architecture
        self.block1 = Residual3DBlock(1, 8, (5, 5, 3))
        self.block2 = Residual3DBlock(8, 16, (5, 5, 3))
        self.blockpool1 = Residual3DBlockPooling(16, 16, (3, 3, 3))
        self.block3 = Residual3DBlock(16, 24, (3, 3, 3))
        self.block4 = Residual3DBlock(24, 24, (3, 3, 3))
        self.blockpool2 = Residual3DBlockPooling(24, 24, (3, 3, 1))
        self.block5 = Residual3DBlock(24, 12, (3, 3, 1))
        self.block6 = Residual3DBlock(12, 6, (3, 3, 1))

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.blockpool1(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.blockpool2(y)
        y = self.block5(y)
        y = self.block6(y)
        return y


class Extractor3DSALE(Lhyp3DModelBase):
    def __init__(self):
        super(Extractor3DSALE, self).__init__()
        # creating the network architecture
        self.block1 = Residual3DBlock(1, 8, (5, 5, 3))
        self.block2 = Residual3DBlock(8, 16, (5, 5, 3))
        self.blockpool1 = Residual3DBlockPooling(16, 16, (3, 3, 3))
        self.block3 = Residual3DBlock(16, 24, (3, 3, 3))
        self.block4 = Residual3DBlock(24, 24, (3, 3, 3))
        self.blockpool2 = Residual3DBlockPooling(24, 24, (3, 3, 1))
        self.block5 = Residual3DBlock(24, 12, (3, 3, 1))
        self.block6 = Residual3DBlock(12, 6, (3, 3, 1))

    def forward(self, x):
        y = self.block1(x)
        y = self.block2(y)
        y = self.blockpool1(y)
        y = self.block3(y)
        y = self.block4(y)
        y = self.blockpool2(y)
        y = self.block5(y)
        y = self.block6(y)
        return y


class Individual3DClassifier(Lhyp3DModelBase):
    def __init__(self, channel_in, mode):
        super(Individual3DClassifier, self).__init__()
        self.blockpool = Residual3DBlockPooling(channel_in, 6, (3, 3, 1))
        self.length = 1734
        if mode == 'sa':
            self.length = 3468

        self.linear = nn.Linear(self.length, 2)  # 2 classes at the output

    def forward(self, x):
        y = self.blockpool(x)
        y = y.view(-1, self.length)
        y = self.linear(y)
        return y


class Ensemble3DClassifier(Lhyp3DModelBase):
    def __init__(self):
        super(Ensemble3DClassifier, self).__init__()
        self.block1 = Residual3DBlock(18, 18, (3, 3, 2))  # original: 104, 82
        self.block2 = Residual3DBlock(18, 6, (3, 3, 1))
        self.blockpool = Residual3DBlockPooling(6, 6, (3, 3, 1))
        self.linear = nn.Linear(1734, 2)

    def forward(self, x, y, z):#, v):
        w = torch.cat([x, y, z], dim=1)#, v], dim=1)
        w = self.block1(w)
        w = self.block2(w)
        w = self.blockpool(w)
        w = w.view(-1, 1734)
        w = self.linear(w)
        return w


class Ensemble3DClassifierWithLE(Lhyp3DModelBase):
    def __init__(self):
        super(Ensemble3DClassifierWithLE, self).__init__()
        self.block1 = Residual3DBlock(36, 18, (3, 3, 2))
        self.block2 = Residual3DBlock(18, 6, (3, 3, 1))
        self.blockpool = Residual3DBlockPooling(6, 6, (3, 3, 1))
        self.linear = nn.Linear(1734, 2)

    def forward(self, x, y, z, u, v, w):
        f = torch.cat([x, y, z, u, v, w], dim=1)
        f = self.block1(f)
        f = self.block2(f)
        f = self.blockpool(f)
        f = f.view(-1, 1734)
        f = self.linear(f)
        return f


class Individual3DModelBase:
    def __init__(self, extractor, classifier, dtype):
        self.dtype = dtype
        self.extractor = extractor
        self.classifier = classifier

    def __call__(self, sample):
        image = sample[self.dtype]
        features = self.extractor(image)
        return self.classifier(features)

    def parameters(self):
        for prm in self.extractor.parameters():
            yield prm
        for prm in self.classifier.parameters():
            yield prm
    
    def to(self, device):
        self.extractor.to(device)
        self.classifier.to(device)
        return self
    
    def save(self, path):
        checkpoint = {
            'extractor_state': self.extractor.state_dict(),
            'classifier_state': self.classifier.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load(self, path, device):
        # load in saved file
        checkpoint = torch.load(path)
        extractor_state = checkpoint['extractor_state']
        classifier_state = checkpoint['classifier_state']
        # load in the state dictionaries
        self.extractor.load(extractor_state, device)
        self.classifier.load(classifier_state, device)


class LA2CHmodel3D(Individual3DModelBase):
    def __init__(self):
        super(LA2CHmodel3D, self).__init__(Extractor3DLA(), Individual3DClassifier(6, 'la'), DataType.LA2CH)


class LA4CHmodel3D(Individual3DModelBase):
    def __init__(self):
        super(LA4CHmodel3D, self).__init__(Extractor3DLA(), Individual3DClassifier(6, 'la'), DataType.LA4CH)


class LALVOTmodel3D(Individual3DModelBase):
    def __init__(self):
        super(LALVOTmodel3D, self).__init__(Extractor3DLA(), Individual3DClassifier(6, 'la'), DataType.LALVOT)


class SAmodel3D(Individual3DModelBase):
    def __init__(self):
        super(SAmodel3D, self).__init__(Extractor3DSA(), Individual3DClassifier(6, 'sa'), DataType.SA)


class LALEmodel3D(Individual3DModelBase):
    def __init__(self):
        super(LALEmodel3D, self).__init__(Extractor3DLALE(), Individual3DClassifier(6, 'lale'), DataType.LALE)


class SALEmodel3D(Individual3DModelBase):
    def __init__(self):
        super(SALEmodel3D, self).__init__(Extractor3DSALE(), Individual3DClassifier(6, 'sale'), DataType.SALE)


class Lhyp3DModel:
    def __init__(self):
        self.extractor_2ch = Extractor3DLA()
        self.extractor_4ch = Extractor3DLA()
        self.extractor_lvot = Extractor3DLA()
        self.extractor_sa = Extractor3DSA()
        self.extractor_sale = Extractor3DSALE()
        self.extractor_lale = Extractor3DLALE()
        self.classifier = Ensemble3DClassifierWithLE()
    
    def __call__(self, sample):
        img_2ch = sample[DataType.LA2CH]
        img_4ch = sample[DataType.LA4CH]
        img_lvot = sample[DataType.LALVOT]
        img_sa = sample[DataType.SA]
        img_sale = sample[DataType.SALE]
        img_lale = sample[DataType.LALE]
        # go through the network
        ftr_2ch = padding_func(self.extractor_2ch(img_2ch), (0, 1), value=0)
        ftr_4ch = padding_func(self.extractor_4ch(img_4ch), (0, 1), value=0)
        ftr_lvot = padding_func(self.extractor_lvot(img_lvot), (0, 1), value=0)
        ftr_sa = self.extractor_sa(img_sa)
        ftr_sale = padding_func(self.extractor_sale(img_sale), (1, 1), value=0)
        ftr_lale = padding_func(self.extractor_lale(img_lale), (1, 1), value=0)
        return self.classifier(ftr_2ch, ftr_4ch, ftr_lvot, ftr_sa, ftr_sale, ftr_lale)

    def freeze_extractors(self):
        self.extractor_2ch.freeze()
        self.extractor_4ch.freeze()
        self.extractor_lvot.freeze()
        self.extractor_sa.freeze()
        self.extractor_sale.freeze()
        self.extractor_lale.freeze()
    
    def parameters(self):
        for prm in self.extractor_2ch.parameters():
            yield prm
        for prm in self.extractor_4ch.parameters():
            yield prm
        for prm in self.extractor_lvot.parameters():
            yield prm
        for prm in self.extractor_sa.parameters():
            yield prm
        for prm in self.extractor_sale.parameters():
            yield prm
        for prm in self.extractor_lale.parameters():
            yield prm
        for prm in self.classifier.parameters():
            yield prm
    
    def to(self, device):
        self.extractor_2ch.to(device)
        self.extractor_4ch.to(device)
        self.extractor_lvot.to(device)
        self.extractor_sa.to(device)
        self.extractor_sale.to(device)
        self.extractor_lale.to(device)
        self.classifier.to(device)
        return self

    def save(self, path):
        checkpoint = {
            'extractor_2ch_state': self.extractor_2ch.state_dict(),
            'extractor_4ch_state': self.extractor_4ch.state_dict(),
            'extractor_lvot_state': self.extractor_lvot.state_dict(),
            'extractor_sa_state': self.extractor_sa.state_dict(),
            'extractor_sale_state': self.extractor_sale.state_dict(),
            'extractor_lale_state': self.extractor_lale.state_dict(),
            'classifier_state': self.classifier.state_dict()
        }
        torch.save(checkpoint, path)
    
    def load(self, path, device):
        # load in saved file
        checkpoint = torch.load(path)
        extractor_2ch_state = checkpoint['extractor_2ch_state']
        extractor_4ch_state = checkpoint['extractor_4ch_state']
        extractor_lvot_state = checkpoint['extractor_lvot_state']
        extractor_sa_state = checkpoint['extractor_sa_state']
        extractor_sale_state = checkpoint['extractor_sale_state']
        extractor_lale_state = checkpoint['extractor_lale_state']
        classifier_state = checkpoint['classifier_state']
        # load in the state dictionaries
        self.extractor_2ch.load(extractor_2ch_state, device)
        self.extractor_4ch.load(extractor_4ch_state, device)
        self.extractor_lvot.load(extractor_lvot_state, device)
        self.extractor_sa.load(extractor_sa_state, device)
        self.extractor_sale.load(extractor_sale_state, device)
        self.extractor_lale.load(extractor_lale_state, device)
        self.classifier.load(classifier_state, device)
    
    def load_pretrained_extractors(self, paths, device):
        checkpoint = torch.load(paths['2ch'])
        self.extractor_2ch.load(checkpoint['extractor_state'], device)
        checkpoint = torch.load(paths['4ch'])
        self.extractor_4ch.load(checkpoint['extractor_state'], device)
        checkpoint = torch.load(paths['lvot'])
        self.extractor_lvot.load(checkpoint['extractor_state'], device)
        checkpoint = torch.load(paths['sa'])
        self.extractor_sa.load(checkpoint['extractor_state'], device)
        checkpoint = torch.load(paths['sale'])
        self.extractor_sale.load(checkpoint['extractor_state'], device)
        checkpoint = torch.load(paths['lale'])
        self.extractor_lale.load(checkpoint['extractor_state'], device)
