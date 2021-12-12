from deepface import DeepFace
from PIL import Image, ImageDraw, ImageFont
from torch.nn.modules import flatten
from torchvision import transforms
import torch
from scipy import spatial
import subprocess
import numpy as np
import random
import time
import io
import matplotlib.pyplot as plt
import lpips
import pandas as pd

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
races = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
genders = ['Man', 'Woman']
loss_fn = lpips.LPIPS(net='alex')
loss_fn = loss_fn.cuda()

def lpips_loss(first_image, second_image):
    first_image = (first_image - 0.5) * 2
    second_image = (second_image - 0.5) * 2
    # get perceptual loss (used to retain similarity between input image and encoded image)
    lpips_loss = loss_fn.forward(first_image, second_image).mean()
    # lpips_loss = loss_fn(image_input, encoded_image).sum()
    return lpips_loss

def random_secret(secret_size):
    return torch.round(torch.rand(secret_size)).tolist()

def get_summary_stat(tensor):
    stats = [torch.max(tensor), torch.min(tensor), torch.mean(tensor), torch.std(tensor)]
    stats = [stat.item() for stat in stats]
    return stats

def display_summary_stat(stats):
    for stat in stats:
        print('%.2f' % stat)

def to_secret_string(secret):
    return ''.join([str(int(i)) for i in secret.tolist()])

def get_similarity(secret_input, decoded_secret, sigmoid=True):
    with torch.no_grad():
        cosine_similarity = []
        cosine_similarity_class = []
        if(sigmoid):
            decoded_secret = torch.sigmoid(decoded_secret)
        decoded_secret_rounded = torch.round(decoded_secret)


        for idx, _ in enumerate(secret_input):
            cosine_similarity.append(1 - spatial.distance.cosine(secret_input.tolist()[idx], decoded_secret.tolist()[idx]))
            cosine_similarity_class.append(1 - spatial.distance.cosine(secret_input.tolist()[idx], decoded_secret_rounded.tolist()[idx]))
        
        similarity_avg = sum(cosine_similarity) / len(cosine_similarity)
        class_similarity_avg = sum(cosine_similarity_class) / len(cosine_similarity_class)


        return similarity_avg, class_similarity_avg, cosine_similarity 

def get_class_accuracy(secret_input, decoded_secret):
    with torch.no_grad():
        cosine_similarity = []
        cosine_similarity_class = []
        decoded_secret = torch.sigmoid(decoded_secret)
        decoded_secret_rounded = torch.round(decoded_secret)
        classes = {
            'emotion': None,
            'race': None,
            'gender': None,
        }
        age = None

        for idx, _ in enumerate(secret_input):
            cosine_similarity.append(1 - spatial.distance.cosine(secret_input.tolist()[idx], decoded_secret.tolist()[idx]))
            cosine_similarity_class.append(1 - spatial.distance.cosine(secret_input.tolist()[idx], decoded_secret_rounded.tolist()[idx]))
            gt_classes = parse_string(to_secret_string(secret_input[idx]))
            out_classes = parse_string(to_secret_string(decoded_secret_rounded[idx]))
        
            for idx, cat in enumerate(classes):
                if(gt_classes[idx] == out_classes[idx]):
                    classes[cat] = True
                else:
                    classes[cat] = False
            
            age = abs(int(gt_classes[3]) - int(out_classes[3]))
        class_similarity_avg = sum(cosine_similarity_class) / len(cosine_similarity_class)

        return class_similarity_avg, classes, age 

# def create_mask_input(image_input, region_input, region_transform=True):
#     with torch.no_grad():
#         mask_input = torch.Tensor(size=(image_input.shape)).cuda()

#         for idx, img in enumerate(image_input):

#             if(region_transform):
#                 region = []
#                 for r in region_input:
#                     region.append(r[idx].int())
#             else:
#                 region = region_input
            
#             x = int(region[0])
#             y = int(region[1])
#             w = int(region[2])
#             h = int(region[3])

#             mask = torch.ones(image_input.shape[1], image_input.shape[2], image_input.shape[3]).cuda()
#             mask[:,y:y+h,x:x+w] = 0

#             mask_input[idx] = mask 
    
#         return mask_input

def extract_face(image_input, region_input):
    with torch.no_grad():
        faces = []

        for idx, img in enumerate(image_input):

            region = []
            for r in region_input:
                region.append(r[idx].int())

            x = int(region[0])
            y = int(region[1])
            w = int(region[2])
            h = int(region[3])

            face = img[:,y:y+h,x:x+w]
            faces.append(face.squeeze())

        return faces 

def swap_face(image_input, faces, region_input):
    with torch.no_grad():
        trans_image_input = torch.Tensor(size=(image_input.shape[0],3,128,128)).cuda()
        for idx, img in enumerate(image_input):

            region = []
            for r in region_input:
                region.append(r[idx].int())

            x = int(region[0])
            y = int(region[1])
            w = int(region[2])
            h = int(region[3])

            face_image = transforms.ToPILImage()(faces[idx].squeeze())

            face_image = face_image.resize((w,h))

            face_tensor = transforms.ToTensor()(face_image)

            img[:,y:y+h,x:x+w] = face_tensor

            trans_image_input[idx] = img

        return trans_image_input 

def get_temperature(gpu):
    return subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader']).decode('utf-8').strip().split('\n')[gpu].split()[0]

# def get_gpu_memory(gpu):
#     used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader']).decode('utf-8').strip('\n')[gpu]
#     used = used.split()[0]
#     total = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader']).decode('utf-8').strip('\n')[gpu]
#     total = total.split()[0]
#     utilization = int(used) / int(total)
#     return 100 * utilization

def process_test_data(performance):
    graph_data_mean = []
    graph_data_std = []
    df = pd.DataFrame(performance)
    for d in df:
        graph_data_mean.append(df[d].mean())
        graph_data_std.append(df[d].std())
    return graph_data_mean, graph_data_std


def create_barh(writer, tag, title, xlabel, labels, performance, epoch):
    y_pos = np.arange(len(labels))
    min = 0
    max = 1
    xerr = [np.subtract(performance, min), np.subtract(max, performance)]
    plt.barh(y_pos, performance, xerr=xerr, align='center', capsize=10)
    plt.yticks(y_pos, labels)
    plt.xlabel(xlabel)
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = Image.open(buf)
    image = transforms.ToTensor()(image).squeeze()

    writer.add_image(tag, image, epoch)

    plt.close()


def channel_noise_limit(batch_bits, amount):
    cur_amount = 0
    batch_bits = torch.clip(batch_bits, 0, 1)
    size = batch_bits.shape
    probability = []
    while(cur_amount < amount):
        probability.append(1)
        cur_amount += 1
    while(len(probability) < size[1]):
        probability.append(0)
    random.shuffle(probability)
    probability = torch.FloatTensor(probability)
    probability = probability.cuda()
    change = (1 - batch_bits) - batch_bits
    change = probability * change
    batch_bits = batch_bits + change
    batch_bits = torch.clip(batch_bits, 0, 1)

    return batch_bits


def channel_noise(batch_bits, prob, rounding_clipping, verbose=False):
    if(verbose):
        start_time = time.time()

    if(rounding_clipping):
        batch_bits = torch.round(torch.clip(batch_bits, 0, 1))
    batch_bits = torch.clip(batch_bits, 0, 1)
    size = batch_bits.shape
    probability = (torch.rand(size[1]) < prob).float()
    probability = probability.cuda()
    change = (1 - batch_bits) - batch_bits
    change = probability * change
    batch_bits = batch_bits + change
    batch_bits = torch.clip(batch_bits, 0, 1)

    if(verbose):
        end_time = time.time()
        time_taken = (end_time - start_time)
        print(f'Channel noise time: {float(time_taken)} seconds')
    return batch_bits

    # batch_bits = batch_bits * random_bits
    # batch_bits = torch.clip(batch_bits, 0, 1)
    # for batch_idx, _ in enumerate(batch_bits):
    #     # batch_bits[batch_idx] = torch.round(batch_bits[batch_idx])
    #     for bit_idx, _ in enumerate(batch_bits[batch_idx]):
    #         if(random.random() < probability):
    #             if(batch_bits[batch_idx][bit_idx] < 0.5):
    #                 batch_bits[batch_idx][bit_idx] = batch_bits[batch_idx][bit_idx] + 1
    #             elif(batch_bits[batch_idx][bit_idx] > 0.5):
    #                 batch_bits[batch_idx][bit_idx] = batch_bits[batch_idx][bit_idx] * 0
    
def draw(img, strings, font_size=24):
    img = transforms.ToPILImage()(img)
    font = ImageFont.truetype('font.ttf', font_size)
    draw = ImageDraw.Draw(img)
    line = 1

    for string in strings:
        draw.text((1,line), string, 255, font=font)
        line += font_size

    return transforms.ToTensor()(img)

def compute_class_loss(decoded_secret, secret_input, cross_entropy):
    emotion = (decoded_secret[:,:3])
    gt_emotion = (secret_input[:,:3])
    race = (decoded_secret[:,3:6])
    gt_race = (secret_input[:,3:6])
    gender = (decoded_secret[:,6])
    gt_gender = (secret_input[:,6])
    age = (decoded_secret[:,7:])
    gt_age = (secret_input[:,7:])

    emotion_loss = cross_entropy(emotion, gt_emotion)
    race_loss = cross_entropy(race, gt_race)
    gender_loss = cross_entropy(gender, gt_gender)
    age_loss = cross_entropy(age, gt_age)

    class_loss = emotion_loss + race_loss + gender_loss + age_loss

    return class_loss

def parse_string(secret):
    emotion_binary = secret[:3]
    race_binary = secret[3:6]
    gender_binary = secret[6]
    age_binary = secret[7:]

    try:
        emotion = emotions[int(emotion_binary, 2)]
    except Exception as e:
        emotion = 'unknown'
    try:
        race = races[int(race_binary, 2)]
    except Exception as e:
        race = 'unknown'
    try:
        gender = genders[int(gender_binary, 2)]
    except:
        gender = 'unknown'
    try:
        age = str(int(age_binary, 2))
    except:
        age = 'unknown'

    return [emotion, race, gender, age]

def get_secret_string(filename):
    secret, region = analyzeFace(filename)
    secret = convert_secret(secret)
    return secret, region

def analyzeFace(filename):
    analysis = DeepFace.analyze(filename,actions=['age','gender', 'race', 'emotion'],detector_backend='ssd',prog_bar=False)
    gender = int(analysis['gender'] == 'Man')
    age = int(analysis['age'])
    race = [float(analysis['race'][r]) for r in analysis['race']]
    emotion = [float(analysis['emotion'][e]) for e in analysis['emotion']]
    region = [int(analysis['region'][r]) for r in analysis['region']]
    # embedding = DeepFace.represent(filename)
    return (age, gender, race, emotion), region

def convert_secret(secret):
    races = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    age = secret[0]
    gender = secret[1] 
    race = secret[2]
    emotion = secret[3]

    max_value = 0
    max_idx = 0
    for idx, e in enumerate(emotions):
        if emotion[idx] > max_value:
            max_value = emotion[idx]
            max_idx = idx
    emotion = emotions[max_idx]

    max_value = 0
    max_idx = 0
    for idx, r in enumerate(races):
        if race[idx] > max_value:
            max_value = race[idx]
            max_idx = idx
    race = races[max_idx]

    for idx, e in enumerate(emotions):
        if emotion == e:
            emotion = idx
    for idx, r in enumerate(races):
        if race == r:
            race = idx
    
    emotion_binary = []
    for b in '{0:b}'.format(emotion):
        emotion_binary.append(int(b))
    while len(emotion_binary) < 3:
        emotion_binary.insert(0,0)

    race_binary = []
    for b in '{0:b}'.format(race):
        race_binary.append(int(b))
    while len(race_binary) < 3:
        race_binary.insert(0,0)

    gender_binary = []
    for b in '{0:b}'.format(gender):
        gender_binary.append(int(b))
    while len(gender_binary) < 1:
        gender_binary.insert(0,0)

    age_binary = []
    for b in '{0:b}'.format(int(age)):
        age_binary.append(int(b))
    while len(age_binary) < 7:
        age_binary.insert(0,0)

    secret_string = emotion_binary + race_binary + gender_binary + age_binary

    return secret_string

# DiffJPEG
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = torch.nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = torch.nn.Parameter(torch.from_numpy(c_table))


def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
