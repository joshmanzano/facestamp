import torch
import utils
import numpy as np
import model
from dataset import Data 
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

secret_size = 14

iterations = ['1', '2']
model_sizes = ['big', 'small']
probs = ['0.1', '0.2']

results = {}

for iter in iterations:
    for model_size in model_sizes:
        for prob in probs:
            ch_enc = f'./checkpoints/{iter}/channel_encoder_{model_size}_{prob}'
            ch_dec = f'./checkpoints/{iter}/channel_decoder_{model_size}_{prob}'
            channel_encoder = model.ChannelEncoder(secret_size, model_size)
            channel_decoder = model.ChannelDecoder(secret_size, model_size)
            channel_encoder.load_state_dict(torch.load(ch_enc))
            channel_decoder.load_state_dict(torch.load(ch_dec))
            channel_encoder.eval()
            channel_decoder.eval()
            channel_encoder = channel_encoder.cuda()
            channel_decoder = channel_decoder.cuda()
            scale_factor = 4

            dataset = Data('train', secret_size, size=(400, 400))
            dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)

            test_result = {}

            for error in range(0,4):
                pbar = tqdm(dataloader)
                similarity = []
                for _, data in enumerate(pbar):
                    cos = torch.nn.CosineSimilarity(dim=1)

                    _, _, secret, _, _= data
                    tensor_secret = secret 
                    tensor_secret = tensor_secret.cuda()
                    bch_secret = channel_encoder(tensor_secret)
                    bch_secret = bch_secret.cuda()

                    ## transit
                    # bch_secret = utils.channel_noise(bch_secret[None], 0.05, False)[0]
                    bch_secret = utils.channel_noise_limit(bch_secret, error)
                    bch_secret = torch.round(bch_secret)

                    ## end_transit

                    decoded = channel_decoder(bch_secret)

                    tensor_secret = torch.round(tensor_secret)
                    decoded = torch.round(decoded)

                    similarity.append(torch.mean(cos(tensor_secret, decoded)).item())
                test_result[error] = np.mean(similarity)
                    
                print(f'## err-{error} {iter} {model_size} {prob} ## ')
                print(np.mean(similarity))
                print(np.std(similarity))
            
            results[f'{iter} {model_size} {prob}'] = test_result

pickle.dump(results,open('channel_test_results.bin', 'wb'))