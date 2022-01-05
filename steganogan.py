from steganogan.models import SteganoGAN
import random
from numpy import dot
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm

def get_steganogan(architecture):

    steganogan_kwargs = {
        'cuda': True,
        'verbose': False,
        # 'depth': 4,
    }

    steganogan_kwargs['path'] = f'./steganogan/pretrained/{architecture}.steg'
    # steganogan_kwargs['architecture'] = 'dense'

    return SteganoGAN.load(**steganogan_kwargs)


def encode():
    """Given loads a pretrained pickel, encodes the image with it."""
    for architecture in ['basic', 'residual', 'dense']:
        similarities = []
        steganogan = get_steganogan(architecture)
        for _ in tqdm(range(100)):
            gt = [random.randint(0,1) for _ in range(100)]
            steganogan.encode('test.png', 'test-enc.png', gt)
            dec = steganogan.decode('test-enc.png')
            dec = [int(i) for i in dec]

            try:
                cos_sim = dot(gt, dec)/(norm(gt)*norm(dec))
            except:
                continue
            # if np.isnan(cos_sim):
            #     print(gt, dec)
            #     print(cos_sim)
            #     breakpoint()

            similarities.append(cos_sim)
        similarities = np.array(similarities)
        print(architecture)
        print(similarities.mean())

if __name__ == '__main__':
    encode()


