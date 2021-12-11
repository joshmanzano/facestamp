from bch_coding import encode_secret, decode_secret
import torch
import utils
import numpy as np
import model

secret_size = 14
def generate_random_secret(size):
    return torch.rand(size)

ch_enc = './checkpoints/channel_encoder_current_0.2'
ch_dec = './checkpoints/channel_decoder_current_0.2'
channel_encoder = model.ChannelEncoder(secret_size)
channel_decoder = model.ChannelDecoder(secret_size)
channel_encoder.load_state_dict(torch.load(ch_enc))
channel_decoder.load_state_dict(torch.load(ch_dec))
channel_encoder.eval()
channel_decoder.eval()
channel_encoder = channel_encoder.cuda()
channel_decoder = channel_decoder.cuda()
scale_factor = 4

if __name__ == '__main__':

    print('BCH')
    for error in range(1,10):
        success = 0 
        total = 10000
        similarity = []
        for _ in range(total):
            cos = torch.nn.CosineSimilarity(dim=0)

            tensor_secret = torch.round(generate_random_secret(secret_size))
            tensor_secret = tensor_secret.cuda()
            bch_secret, secret = encode_secret(tensor_secret)
            bch_secret = bch_secret.cuda()

            ## transit
            # bch_secret = utils.channel_noise(bch_secret[None], 0.05, False)[0]
            bch_secret = utils.channel_noise_limit(bch_secret[None], error)[0]
            bch_secret = torch.round(bch_secret)

            ## end_transit

            decoded, status = decode_secret(bch_secret)

            similarity.append(cos(tensor_secret, decoded).item())
            if(status):
                success += 1
            
        print(f'## {error} ##')
        print(success / total)
        print(np.mean(similarity))
        print(np.std(similarity))

    print('Channel Coding')
    for error in range(1,10):
        success = 0 
        total = 10000
        similarity = []
        for _ in range(total):
            cos = torch.nn.CosineSimilarity(dim=0)

            tensor_secret = torch.round(generate_random_secret(secret_size))
            tensor_secret = tensor_secret.cuda()
            bch_secret = channel_encoder(tensor_secret)
            bch_secret = bch_secret.cuda()

            ## transit
            # bch_secret = utils.channel_noise(bch_secret[None], 0.05, False)[0]
            bch_secret = utils.channel_noise_limit(bch_secret[None], error)[0]
            bch_secret = torch.round(bch_secret)

            ## end_transit

            decoded = channel_decoder(bch_secret)

            tensor_secret = torch.round(tensor_secret)
            decoded = torch.round(decoded)

            similarity.append(cos(tensor_secret, decoded).item())
            if(cos(tensor_secret, decoded).item == 1.):
                success += 1
            
        print(f'## {error} ##')
        print(success / total)
        print(np.mean(similarity))
        print(np.std(similarity))