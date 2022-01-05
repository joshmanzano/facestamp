import argparse
from torch import nn
import torch

class ChannelDecoder(nn.Module):
    def __init__(self, secret_size, model_size):
        super().__init__()
        if(model_size == 'small'):
            layers = nn.Sequential(
                nn.Linear(100, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, secret_size * 3),
                nn.ReLU(),
                nn.Linear(secret_size * 3, secret_size),
                nn.ReLU(),
            )
        elif(model_size == 'big'):
            layers = nn.Sequential(
                nn.Linear(100, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, secret_size * 4),
                nn.ReLU(),
                nn.Linear(secret_size * 4, secret_size * 2),
                nn.ReLU(),
                nn.Linear(secret_size * 2, secret_size),
                nn.ReLU(),
            )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output

class ChannelEncoder(nn.Module):
    def __init__(self, secret_size, model_size):
        super().__init__()
        if(model_size == 'small'):
            layers = nn.Sequential(
                nn.Linear(secret_size, secret_size * 3),
                nn.ReLU(),
                nn.Linear(secret_size * 3, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, 100),
                nn.ReLU(),
            )
        elif(model_size == 'big'):
            layers = nn.Sequential(
                nn.Linear(secret_size, secret_size * 2),
                nn.ReLU(),
                nn.Linear(secret_size * 2, secret_size * 4),
                nn.ReLU(),
                nn.Linear(secret_size * 4, secret_size * 6),
                nn.ReLU(),
                nn.Linear(secret_size * 6, 100),
                nn.ReLU(),
            )
        self.main = layers

    def forward(self, secret):
        output = self.main(secret)
        return output

channel_encoder = ChannelEncoder(14, 'small')
channel_decoder = ChannelDecoder(14, 'small')
channel_encoder.load_state_dict(torch.load('checkpoints/channel_encoder_s22'))
channel_decoder.load_state_dict(torch.load('checkpoints/channel_decoder_s22'))
channel_encoder.eval()
channel_decoder.eval()
channel_encoder = channel_encoder.cuda()
channel_decoder = channel_decoder.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify file to analyze')
    parser.add_argument('mode')
    parser.add_argument('secret')

    args = parser.parse_args()

    mode = args.mode == 'encode'
    secret = args.secret.strip()

    secret_array = [int(s) for s in secret]
    if(mode):
        encoded = channel_encoder(torch.Tensor(secret_array).cuda())
        encoded = torch.round(torch.clip(encoded,0,1))
        encoded_string = ''
        for i in encoded:
            encoded_string += str(int(i))
        print(encoded_string)
    else:
        decoded = channel_decoder(torch.Tensor(secret_array).cuda())
        decoded = torch.round(torch.clip(decoded,0,1))
        decoded_string = ''
        for i in decoded :
            decoded_string += str(int(i))
        print(decoded_string)


