import torch
import glob
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from tqdm import tqdm
import torchvision.transforms as T
import random
import string
import lpips
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter
import kornia as k
import utils

#loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

class Flickr(Dataset):
    def __init__(self,path = './data/mirflickr/*.jpg'):
        self.data_list = glob.glob(path)[:100]
        self.trans = T.Compose([T.Resize((400,400)), T.ToTensor()])
        self.totensor = T.Compose([T.ToTensor()])
        #bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
        #data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
        #ecc = bch.encode(data)
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        secret = torch.tensor(random.choices([0,1], k=100))
        img = Image.open(self.data_list[idx])
        img = self.trans(img) - 0.5
        #secret = self.totensor(int(secret))
        return img, secret

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(100,7500)
        self.conv1 = nn.Conv2d(6, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride = 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride = 2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride = 2)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride = 2)
        
        
        self.conv6 = nn.Conv2d(384, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(192, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(160, 64, 3, padding =1)
        self.conv9 = nn.Conv2d(96, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, 32, 2)  # Downsampling??? TODO: check the paddin
        self.conv11 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv12 = nn.Conv2d(32, 32, 2)  # Downsampling??? TODO: check the paddin
        
        self.conv13 = nn.Conv2d(70, 32, 3, padding=1)
        self.conv14 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.conv15 = nn.Conv2d(64, 3, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') 
        self.upsample8 = nn.Upsample(scale_factor=8, mode='nearest') 
    def forward(self, x, secret):
        
        secret = F.relu(self.linear(secret))
        b,seq=secret.size()
        secret = secret.view(b,3,50,50)
        secret= secret.contiguous()

        secret = self.upsample8(secret)
        inp = torch.cat((x, secret), 1)
        #downsample phase
        conv1 = F.relu(self.conv1(inp))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv4 = F.relu(self.conv4(conv3))
        conv5 = F.relu(self.conv5(conv4))
        conv5=self.upsample(conv5) 
        #upsample phase
        merge = torch.cat( (conv4 , conv5), 1)

        conv6 = self.upsample( F.relu(self.conv6(merge)))
        merge = torch.cat([conv3, conv6], 1)
        conv7 = self.upsample( F.relu(self.conv7(merge)))
        merge = torch.cat([conv2, conv7], 1)
        conv8 = self.upsample( F.relu(self.conv8(merge)))
        merge = torch.cat([conv1, conv8], 1)
        conv9 = F.relu(self.conv9(merge))
        residual = self.conv15(conv9)
        return residual

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        pad = 1
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding=pad)
        self.conv2 = nn.Conv2d(32, 32, 3,  padding=pad)
        self.conv3 = nn.Conv2d(32, 64, 3,stride = 2,  padding=pad)
        self.conv4 = nn.Conv2d(64, 64, 3,  padding=pad)
        self.conv5 = nn.Conv2d(64, 64, 3,stride = 2,  padding=pad)
        self.conv6 = nn.Conv2d(64, 128, 3,stride = 2, padding=pad)
        self.conv7 = nn.Conv2d(128, 128, 3,stride= 2, padding=pad)
        self.dense8 = nn.Linear(21632, 512)
        self.dense9 = nn.Linear(512, 100)
        
    def forward(self, input):
        
        x = nn.functional.relu(self.conv1(input))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        x = nn.functional.relu(self.conv5(x))
        x = nn.functional.relu(self.conv6(x))
        x = nn.functional.relu(self.conv7(x))
        x= torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.dense8(x))
        x = self.dense9(x)
        return x 

loss_fn_vgg = lpips.LPIPS(net='vgg')

enc = Encoder()
dec = Decoder()
data = Flickr()

device = 'cuda'
enc = enc.to(device)
dec = dec.to(device)
loss_fn_vgg = loss_fn_vgg.to(device)
bce = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(list( dec.parameters()) + list(enc.parameters()), lr=0.0001 )

INTERVAL = 100
BATCH_SIZE = 4
TURN_ON_WEIGHTS = 1800

data = Flickr()
train_dataloader = DataLoader(data, batch_size=BATCH_SIZE , shuffle=True)

trans = torch.nn.Sequential(
    k.augmentation.RandomPerspective(0.5, p=1),
    k.augmentation.RandomMotionBlur(5, 1.0,1.0, p=1, resample = 'bilinear'),
    k.augmentation.RandomGaussianBlur((3, 3), (1, 3.0), p=1.),
    k.augmentation.RandomGaussianNoise(mean=0., std=0.2, p=1.),
    k.augmentation.ColorJitter( 0.3, ( 0.5, 1.5), 0.1, 0.1, p=1.),
)


def train(writer, epoch=0):
    
    MAX = len(train_dataloader)
    
    for idx, data in enumerate(tqdm(train_dataloader)):
        step = (MAX * epoch) + idx
        optimizer.zero_grad()
        img = data[0].to(device)
        secret = data[1].to(device).float()

        residual = enc(img, secret.float())
        
        out_img = img + residual 
        out_img = (torch.clip(out_img, min = 0, max = 1) - 0.5)

        out_secret = dec(out_img)

        bce_loss = bce(out_secret, secret)

        if(TURN_ON_WEIGHTS <= step):
            residual_loss = (residual**2).mean()
            percep = loss_fn_vgg(out_img, img).sum()
            total_loss =  (bce_loss * 10) + percep + residual_loss
        else:
            total_loss =  (bce_loss * 10)

        total_loss.backward()
        optimizer.step()

    step = (MAX * epoch) + idx
    acc, _, _ = utils.get_similarity(out_secret, secret, sigmoid=False)

    print('Total loss:',total_loss.item())
    print('BCE:',bce_loss.item())
    print('Acc:',acc)
    writer.add_scalar('loss/total_loss', total_loss.item(), step)
    writer.add_scalar('loss/bce', bce_loss.item(), step)
    if(TURN_ON_WEIGHTS <= step):
        writer.add_scalar('weight/percep', percep.item(), step)
        writer.add_scalar('weight/residual', residual_loss.item(), step)
    torch.save(enc.state_dict(), './checkpoints/encoder-{}-{}.pth'.format(epoch,idx))
    torch.save(dec.state_dict(), './checkpoints/decoder-{}-{}.pth'.format(epoch,idx))

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


dec.apply(init_weights)
enc.apply(init_weights)
for e in range(10000):
    writer = SummaryWriter(log_dir='./logs/stegastamp')
    train(writer, e)