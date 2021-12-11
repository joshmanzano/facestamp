import bchlib
import hashlib
import torch
import numpy as np

import utils

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
secret_size = 14
padding = 2

def generate_random_secret(size):
    return torch.rand(size)

## assumed multiple by 8
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')

## default fill 16 bits
def bytes_to_bitstring(b):
    return ''.join(format(x, '08b') for x in b)
    # return format(int.from_bytes(b, byteorder='big'), 'b').zfill(fill)

def bitstring_to_intarray(s):
    return [int(i) for i in s]

## input of tensor, output tensor
def encode_secret(secret):
    assert len(secret) == secret_size
    if(type(secret) != list):
        secret = torch.round(secret).int().tolist()
    secret = ''.join(['0' for _ in range(padding)] + [str(int(i)) for i in secret])
    byte_secret = bitstring_to_bytes(secret)
    ecc = bch.encode(byte_secret)
    packet = byte_secret + ecc
    packet_string = bytes_to_bitstring(packet) 
    bch_secret = torch.Tensor(bitstring_to_intarray(packet_string))
    secret = torch.Tensor(bitstring_to_intarray(secret))
    assert len(bch_secret) == secret_size + (BCH_BITS * 8) + padding
    return bch_secret, secret

## input of tensor, returns tensor and status
def decode_secret(decoded):
    assert len(decoded) == secret_size + (BCH_BITS * 8) + padding
    if(type(decoded) != list):
        decoded = torch.round(decoded).int().tolist()
    decoded = ''.join([str(int(i)) for i in decoded])
    packet = bytes(int(decoded[i : i + 8], 2) for i in range(0, len(decoded), 8))
    packet = bytearray(packet)
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
    bitflips = bch.decode_inplace(data, ecc)
    code = bytes_to_bitstring(data) 
    tensor_code = torch.Tensor(bitstring_to_intarray(code)).cuda()
    tensor_code = tensor_code[padding:]
    if bitflips != -1:
        return tensor_code, True
    return tensor_code, False

if __name__ == '__main__':

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