import model
from dataset import Data
from torch.utils.data import DataLoader, RandomSampler
import torch
from tqdm import tqdm
import numpy as np

def main(ch_enc, ch_dec, source):
    dataset = Data('./data/celebatest', 14, size=(128, 128), dataset_size=0)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    # enc_model_path = './evaluate/encoder_current_rw_distortion_no_channel'
    # dec_model_path = './evaluate/decoder_current_rw_distortion_no_channel'
    ch_enc_model_path = f'./{source}/{ch_enc}'
    ch_dec_model_path = f'./{source}/{ch_dec}'

    channel_encoder = model.ChannelEncoder(14)
    channel_decoder = model.ChannelDecoder(14)
    channel_encoder.load_state_dict(torch.load(ch_enc_model_path))
    channel_decoder.load_state_dict(torch.load(ch_dec_model_path))
    channel_encoder.cuda()
    channel_decoder.cuda()
    channel_encoder.eval()
    channel_decoder.eval()

    results = []
    for data in tqdm(dataloader):
        _, gt_secret, _, _ = data
        gt_secret = gt_secret.cuda()
        secret_input = channel_encoder(gt_secret)
        recovered = channel_decoder(secret_input)
        recovered = torch.clip(recovered, 0, 1)
        recovered = torch.round(recovered)
        similarity = torch.nn.CosineSimilarity(dim=1)(gt_secret, recovered)
        results.append(similarity.item())
    print(np.mean(results))
    print(np.std(results))

if __name__ == '__main__':
    main('channel_encoder_current_0.2','channel_decoder_current_0.2','checkpoints')

